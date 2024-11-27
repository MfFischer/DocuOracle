from flask import Blueprint, current_app, request
from flask_restful import Resource, Api, reqparse
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename
import werkzeug.datastructures
from docuoracle_app import db, csrf
from docuoracle_app.models import Document
from docuoracle_app.llama_handler import (
    process_document_with_llama,
    process_document_with_rag,
    get_llama_status,
    initialize_llama,
    initialize_rag
)
from docuoracle_app.graph_handler import (
    get_graph_handler,
    generate_plotly_graph,
    generate_matplotlib_graph,
    parse_excel,
    generate_visualizations,
    get_available_charts
)
import os
import logging
from functools import wraps

# Configure logging
logger = logging.getLogger(__name__)

# Define the API blueprint
api_blueprint = Blueprint('api', __name__)
api = Api(api_blueprint)


# CSRF exempt decorator for specific routes that need it
def csrf_exempt(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        return view(*args, **kwargs)

    return csrf.exempt(wrapped)


# Request parsers with CSRF token
def parse_with_csrf():
    parser = reqparse.RequestParser()
    parser.add_argument('csrf_token', type=str, location=['form', 'headers', 'json'])
    return parser


# Define request parsers
file_parser = parse_with_csrf()
file_parser.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files', required=True)

query_parser = parse_with_csrf()
query_parser.add_argument('document_id', type=int, required=True)
query_parser.add_argument('question', type=str, required=True)
query_parser.add_argument('use_rag', type=bool, default=True)

graph_parser = parse_with_csrf()
graph_parser.add_argument('document_id', type=int, required=True)
graph_parser.add_argument('chart_type', type=str, required=True)
graph_parser.add_argument('x_col', type=str, required=True)
graph_parser.add_argument('y_col', type=str, required=True)

model_parser = parse_with_csrf()
model_parser.add_argument('use_rag', type=bool, default=True)
model_parser.add_argument('deployment', type=str, default='production')
model_parser.add_argument('resources', type=str, default='limited')


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'xlsx', 'xls', 'csv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Base API class with CSRF handling
class CSRFProtectedResource(Resource):
    method_decorators = [login_required]

    def dispatch_request(self, *args, **kwargs):
        if request.method != 'GET':
            csrf_token = request.headers.get('X-CSRFToken') or \
                         request.form.get('csrf_token') or \
                         request.json.get('csrf_token') if request.json else None

            if not csrf_token:
                return {'error': 'CSRF token missing'}, 400

            if not csrf.validate_csrf(csrf_token):
                return {'error': 'Invalid CSRF token'}, 400

        return super(CSRFProtectedResource, self).dispatch_request(*args, **kwargs)


class DocumentAPI(CSRFProtectedResource):
    def get(self, document_id=None):
        try:
            if document_id:
                document = Document.query.filter_by(
                    id=document_id,
                    user_id=current_user.id
                ).first()

                if not document:
                    return {'error': 'Document not found'}, 404

                return {
                    'id': document.id,
                    'filename': document.filename,
                    'uploaded_at': document.uploaded_at.isoformat(),
                    'file_type': document.file_type,
                    'processed': document.processed,
                    'rag_processed': document.rag_processed if hasattr(document, 'rag_processed') else False
                }, 200
            else:
                documents = Document.query.filter_by(user_id=current_user.id).all()
                return {
                    'documents': [{
                        'id': doc.id,
                        'filename': doc.filename,
                        'uploaded_at': doc.uploaded_at.isoformat(),
                        'rag_processed': doc.rag_processed if hasattr(doc, 'rag_processed') else False
                    } for doc in documents]
                }, 200

        except Exception as e:
            logger.error(f"Error in DocumentAPI.get: {str(e)}")
            return {'error': 'Internal server error'}, 500

    def post(self):
        try:
            args = file_parser.parse_args()
            file = args['file']

            if not file:
                return {'error': 'No file provided'}, 400

            if not allowed_file(file.filename):
                return {'error': 'File type not allowed'}, 400

            filename = secure_filename(file.filename)
            file_type = filename.rsplit('.', 1)[1].lower()

            upload_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], str(current_user.id))
            os.makedirs(upload_dir, exist_ok=True)

            filepath = os.path.join(upload_dir, filename)
            file.save(filepath)

            new_doc = Document(
                filename=filename,
                filepath=filepath,
                user_id=current_user.id,
                file_type=file_type,
                rag_processed=False
            )
            db.session.add(new_doc)
            db.session.commit()

            return {
                'message': 'File uploaded successfully',
                'document': {
                    'id': new_doc.id,
                    'filename': new_doc.filename,
                    'uploaded_at': new_doc.uploaded_at.isoformat()
                }
            }, 201

        except Exception as e:
            logger.error(f"Error in DocumentAPI.post: {str(e)}")
            return {'error': 'Internal server error'}, 500


class QueryAPI(CSRFProtectedResource):
    def post(self):
        try:
            args = query_parser.parse_args()
            document = Document.query.filter_by(
                id=args['document_id'],
                user_id=current_user.id
            ).first()

            if not document:
                return {'error': 'Document not found'}, 404

            try:
                if document.file_type in ['xlsx', 'xls', 'csv']:
                    df = parse_excel(document.filepath)
                    document_text = df.to_string()
                else:
                    with open(document.filepath, 'r') as doc_file:
                        document_text = doc_file.read()

                if args.get('use_rag', True):
                    result = process_document_with_rag(
                        document_text,
                        args['question'],
                        is_data_analysis=(document.file_type in ['xlsx', 'xls', 'csv'])
                    )
                else:
                    result = process_document_with_llama(
                        document_text,
                        args['question'],
                        is_data_analysis=(document.file_type in ['xlsx', 'xls', 'csv'])
                    )

                document.processed = True
                if args.get('use_rag', True):
                    document.rag_processed = True
                db.session.commit()

                return {
                    'success': True,
                    'result': result,
                    'document': {
                        'id': document.id,
                        'filename': document.filename
                    }
                }, 200

            except Exception as e:
                logger.error(f"Error processing document: {str(e)}")
                return {'error': 'Error processing document'}, 500

        except Exception as e:
            logger.error(f"Error in QueryAPI.post: {str(e)}")
            return {'error': 'Internal server error'}, 500


class GraphAPI(CSRFProtectedResource):
    def post(self):
        try:
            args = graph_parser.parse_args()
            document = Document.query.filter_by(
                id=args['document_id'],
                user_id=current_user.id
            ).first()

            if not document:
                return {'error': 'Document not found'}, 404

            if document.file_type not in ['xlsx', 'xls', 'csv']:
                return {'error': 'Document is not a spreadsheet'}, 400

            try:
                df = parse_excel(document.filepath)

                if args['x_col'] not in df.columns or args['y_col'] not in df.columns:
                    return {'error': 'Specified columns not found in document'}, 400

                if args.get('library', 'plotly') == 'plotly':
                    graph = generate_plotly_graph(
                        df,
                        args['chart_type'],
                        args['x_col'],
                        args['y_col']
                    )
                else:
                    graph = generate_matplotlib_graph(
                        df,
                        args['chart_type'],
                        args['x_col'],
                        args['y_col']
                    )

                return {
                    'graph': graph,
                    'document': {
                        'id': document.id,
                        'filename': document.filename
                    }
                }, 200

            except Exception as e:
                logger.error(f"Error generating graph: {str(e)}")
                return {'error': 'Error generating graph'}, 500

        except Exception as e:
            logger.error(f"Error in GraphAPI.post: {str(e)}")
            return {'error': 'Internal server error'}, 500


class ModelAPI(CSRFProtectedResource):
    def get(self):
        """Get current model status"""
        try:
            status = get_llama_status()
            return {
                'success': True,
                'status': status
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }, 500

    def post(self):
        """Initialize or connect to the model (RAG or traditional)"""
        try:
            args = model_parser.parse_args()

            # Check current status
            status = get_llama_status()
            if status['initialized']:
                return {
                    'success': True,
                    'message': 'Model already connected',
                    'status': status
                }

            # Initialize based on configuration
            if args.get('use_rag', True):
                requirements = {
                    'deployment': args.get('deployment', 'production'),
                    'resources': args.get('resources', 'limited')
                }
                success, message = initialize_rag(requirements)
            else:
                success, message = initialize_llama()

            if success:
                return {
                    'success': True,
                    'message': message,
                    'status': get_llama_status()
                }
            else:
                return {
                    'success': False,
                    'error': message,
                    'status': get_llama_status()
                }, 500

        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'status': get_llama_status()
            }, 500

class FileUploadAPI(CSRFProtectedResource):
    def post(self):
        try:
            # Get file from request
            if 'file' not in request.files:
                return {'success': False, 'error': 'No file part'}, 400

            file = request.files['file']
            if file.filename == '':
                return {'success': False, 'error': 'No selected file'}, 400

            if not allowed_file(file.filename):
                return {'success': False, 'error': 'File type not allowed'}, 400

            try:
                filename = secure_filename(file.filename)
                upload_folder = current_app.config['UPLOAD_FOLDER']
                user_folder = os.path.join(upload_folder, str(current_user.id))
                os.makedirs(user_folder, exist_ok=True)

                filepath = os.path.join(user_folder, filename)
                file.save(filepath)

                # Save document in database
                doc = Document(
                    filename=filename,
                    filepath=filepath,
                    user_id=current_user.id,
                    file_type=filename.rsplit('.', 1)[1].lower()
                )
                db.session.add(doc)
                db.session.commit()

                return {
                    'success': True,
                    'message': 'File uploaded successfully',
                    'document': {
                        'id': doc.id,
                        'filename': doc.filename
                    }
                }

            except Exception as e:
                logger.error(f"Error saving file: {str(e)}")
                return {'success': False, 'error': f'Error saving file: {str(e)}'}, 500

        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            return {'success': False, 'error': f'Upload error: {str(e)}'}, 500


# Add resources to the API
api.add_resource(DocumentAPI, '/documents', '/documents/<int:document_id>')
api.add_resource(QueryAPI, '/query')
api.add_resource(GraphAPI, '/graph')
api.add_resource(ModelAPI, '/model')
api.add_resource(FileUploadAPI, '/upload_document')