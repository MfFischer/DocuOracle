from flask import Blueprint, request, jsonify, current_app
from flask_restful import Resource, Api, reqparse
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename
import werkzeug.datastructures
from docuoracle_app import db
from docuoracle_app.models import Document
from docuoracle_app.llama_handler import process_document_with_llama
from docuoracle_app.graph_handler import parse_excel, generate_plotly_graph, generate_matplotlib_graph
import os
from datetime import datetime
import json

# Define the API blueprint
api_blueprint = Blueprint('api', __name__)
api = Api(api_blueprint)

# Define request parsers
file_parser = reqparse.RequestParser()
file_parser.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files', required=True)

query_parser = reqparse.RequestParser()
query_parser.add_argument('document_id', type=int, required=True)
query_parser.add_argument('question', type=str, required=True)

graph_parser = reqparse.RequestParser()
graph_parser.add_argument('document_id', type=int, required=True)
graph_parser.add_argument('chart_type', type=str, required=True)
graph_parser.add_argument('x_col', type=str, required=True)
graph_parser.add_argument('y_col', type=str, required=True)


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'xlsx', 'xls', 'csv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class DocumentAPI(Resource):
    method_decorators = [login_required]  # Require login for all methods

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
                    'processed': document.processed
                }, 200
            else:
                # List all documents for current user
                documents = Document.query.filter_by(user_id=current_user.id).all()
                return {
                    'documents': [{
                        'id': doc.id,
                        'filename': doc.filename,
                        'uploaded_at': doc.uploaded_at.isoformat()
                    } for doc in documents]
                }, 200

        except Exception as e:
            current_app.logger.error(f"Error in DocumentAPI.get: {str(e)}")
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

            # Create upload directory if it doesn't exist
            upload_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], str(current_user.id))
            os.makedirs(upload_dir, exist_ok=True)

            filepath = os.path.join(upload_dir, filename)
            file.save(filepath)

            new_doc = Document(
                filename=filename,
                filepath=filepath,
                user_id=current_user.id,
                file_type=file_type
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
            current_app.logger.error(f"Error in DocumentAPI.post: {str(e)}")
            return {'error': 'Internal server error'}, 500


class QueryAPI(Resource):
    method_decorators = [login_required]

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

                answer = process_document_with_llama(args['question'], document_text)

                return {
                    'answer': answer,
                    'document': {
                        'id': document.id,
                        'filename': document.filename
                    }
                }, 200

            except Exception as e:
                current_app.logger.error(f"Error processing document: {str(e)}")
                return {'error': 'Error processing document'}, 500

        except Exception as e:
            current_app.logger.error(f"Error in QueryAPI.post: {str(e)}")
            return {'error': 'Internal server error'}, 500


class GraphAPI(Resource):
    method_decorators = [login_required]

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

                # Validate columns exist
                if args['x_col'] not in df.columns or args['y_col'] not in df.columns:
                    return {'error': 'Specified columns not found in document'}, 400

                # Generate graph
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
                current_app.logger.error(f"Error generating graph: {str(e)}")
                return {'error': 'Error generating graph'}, 500

        except Exception as e:
            current_app.logger.error(f"Error in GraphAPI.post: {str(e)}")
            return {'error': 'Internal server error'}, 500


# Add the resources to the API
api.add_resource(DocumentAPI, '/documents', '/documents/<int:document_id>')
api.add_resource(QueryAPI, '/query')
api.add_resource(GraphAPI, '/graph')