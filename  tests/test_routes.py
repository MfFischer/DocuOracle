import unittest
from app import app, db
from app.models import User, Document
from io import BytesIO


class RoutesTestCase(unittest.TestCase):

    def setUp(self):
        """
        Set up the testing environment.
        """
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'  # In-memory database for tests
        self.client = app.test_client()  # Create test client
        self.ctx = app.app_context()
        self.ctx.push()
        db.create_all()  # Create database tables

        # Add a test user
        self.test_user = User(username='testuser', password='testpassword')
        db.session.add(self.test_user)
        db.session.commit()

    def tearDown(self):
        """
        Tear down the testing environment.
        """
        db.session.remove()
        db.drop_all()
        self.ctx.pop()

    def test_home_route(self):
        """
        Test the home page route.
        """
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Welcome to DocuOracle', response.data)

    def test_upload_route_get(self):
        """
        Test the GET method for the upload route.
        """
        response = self.client.get('/upload')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Upload Document', response.data)

    def test_upload_route_post(self):
        """
        Test the POST method for the upload route with a PDF file.
        """
        data = {
            'file': (BytesIO(b"Dummy PDF content"), 'test.pdf')
        }
        response = self.client.post('/upload', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'test.pdf', response.data)

    def test_api_documents_post(self):
        """
        Test the API for uploading a document via POST.
        """
        data = {
            'file': (BytesIO(b"Dummy PDF content"), 'testapi.pdf')
        }
        response = self.client.post('/api/documents', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 201)
        self.assertIn(b'File uploaded successfully', response.data)

    def test_api_documents_get(self):
        """
        Test the API for retrieving a document by ID.
        """
        # Add a document to the database
        document = Document(filename='testdoc.pdf', filepath='documents/testdoc.pdf', user_id=self.test_user.id)
        db.session.add(document)
        db.session.commit()

        response = self.client.get(f'/api/documents/{document.id}')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'testdoc.pdf', response.data)

    def test_api_query_post(self):
        """
        Test the API for asking a question about a document.
        """
        # Add a document to the database
        document = Document(filename='testdoc.pdf', filepath='documents/testdoc.pdf', user_id=self.test_user.id)
        db.session.add(document)
        db.session.commit()

        data = {
            'document_id': document.id,
            'question': 'What is the document about?'
        }
        response = self.client.post('/api/query', json=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'answer', response.data)  # Check for 'answer' in the JSON response


if __name__ == '__main__':
    unittest.main()
