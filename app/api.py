# app/api.py

from flask import request, jsonify
from flask_restful import Resource, Api
from app import app, db
from app.models import Document
from app.llama_handler import generate_response
from app.graph_handler import parse_excel, generate_matplotlib_graph, generate_plotly_graph
from flask_swagger_ui import get_swaggerui_blueprint

api = Api(app)

# Swagger configuration
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "DocuOracle API"
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# API for handling document uploads
class DocumentAPI(Resource):
    def get(self, document_id):
        document = Document.query.get(document_id)
        if document:
            return {'filename': document.filename, 'uploaded_at': document.uploaded_at}
        return {'error': 'Document not found'}, 404

    def post(self):
        uploaded_file = request.files.get('file')
        if uploaded_file:
            filename = uploaded_file.filename
            filepath = os.path.join('documents', filename)
            uploaded_file.save(filepath)

            new_doc = Document(filename=filename, filepath=filepath, user_id=1)  # Replace with actual user ID
            db.session.add(new_doc)
            db.session.commit()

            return {'message': 'File uploaded successfully', 'document_id': new_doc.id}, 201
        return {'error': 'No file uploaded'}, 400

# API for asking questions
class QueryAPI(Resource):
    def post(self):
        data = request.get_json()
        document_id = data.get('document_id')
        question = data.get('question')

        document = Document.query.get(document_id)
        if document:
            # Read the document and pass it to LLaMA (Ollama)
            if document.filename.endswith('.xls') or document.filename.endswith('.xlsx') or document.filename.endswith('.csv'):
                df = parse_excel(document.filepath)
                document_text = df.to_string()
            else:
                with open(document.filepath, 'r') as doc_file:
                    document_text = doc_file.read()

            answer = generate_response(question, document_text)
            return {'answer': answer}, 200
        return {'error': 'Document not found'}, 404

# API for generating graphs
class GraphAPI(Resource):
    def post(self):
        data = request.get_json()
        document_id = data.get('document_id')
        chart_type = data.get('chart_type')  # e.g., 'line', 'bar', 'scatter'
        x_col = data.get('x_col')
        y_col = data.get('y_col')

        document = Document.query.get(document_id)
        if document and (document.filename.endswith('.xls') or document.filename.endswith('.xlsx') or document.filename.endswith('.csv')):
            df = parse_excel(document.filepath)
            if chart_type and x_col and y_col:
                # Generate graph (choose Matplotlib or Plotly)
                graph = generate_plotly_graph(df, chart_type, x_col, y_col)
                return {'graph': graph}, 200
            else:
                return {'error': 'Missing graph parameters'}, 400
        return {'error': 'Document not found or not an Excel file'}, 404

api.add_resource(DocumentAPI, '/api/documents/<int:document_id>', '/api/documents')
api.add_resource(QueryAPI, '/api/query')
api.add_resource(GraphAPI, '/api/graph')
