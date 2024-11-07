from flask import Blueprint, render_template, flash, redirect, url_for, request, current_app
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
import os

# Import from your package
from docuoracle_app import db  # Import db instance from __init__.py
from docuoracle_app.models import User, Document  # Import models
from docuoracle_app.utils import parse_document  # Import utilities
from docuoracle_app.llama_handler import process_document_with_llama  # Import LLaMA handler
from docuoracle_app.graph_handler import generate_analysis_graphs  # Import graph handler

# Create Blueprint
routes_blueprint = Blueprint('routes', __name__)


# Home route
@routes_blueprint.route('/')
@routes_blueprint.route('/home')
def home():
    """Home page route."""
    return render_template('home.html')


# Login route
@routes_blueprint.route('/login', methods=['GET', 'POST'])
def login():
    """User login route."""
    if current_user.is_authenticated:
        return redirect(url_for('routes.home'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            flash('Login successful!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('routes.home'))

        flash('Invalid username or password', 'error')

    return render_template('login.html')


# Register route
@routes_blueprint.route('/register', methods=['GET', 'POST'])
def register():
    """User registration route."""
    if current_user.is_authenticated:
        return redirect(url_for('routes.home'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('routes.register'))

        user = User(username=username)
        user.set_password(password)

        try:
            db.session.add(user)
            db.session.commit()
            flash('Registration successful!', 'success')
            return redirect(url_for('routes.login'))
        except Exception as e:
            db.session.rollback()
            flash('Registration failed. Please try again.', 'error')
            print(f"Registration error: {e}")

    return render_template('register.html')


# Logout route
@routes_blueprint.route('/logout')
@login_required
def logout():
    """User logout route."""
    logout_user()
    flash('Logged out successfully', 'success')
    return redirect(url_for('routes.home'))


# Upload route
@routes_blueprint.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    """Document upload route."""
    if request.method == 'POST':
        if 'document' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)

        file = request.files['document']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Parse document content
                content = parse_document(filepath)

                # Create document record
                document = Document(
                    filename=filename,
                    filepath=filepath,
                    content=content,
                    user_id=current_user.id
                )

                db.session.add(document)
                db.session.commit()

                flash('Document uploaded successfully!', 'success')
                return redirect(url_for('routes.analyze_document', doc_id=document.id))

            except Exception as e:
                db.session.rollback()
                flash('Error uploading document. Please try again.', 'error')
                print(f"Upload error: {e}")
        else:
            flash('Invalid file type', 'error')

    return render_template('upload.html')


# Document analysis route
@routes_blueprint.route('/analyze/<int:doc_id>', methods=['GET', 'POST'])
@login_required
def analyze_document(doc_id):
    """Document analysis route."""
    document = Document.query.get_or_404(doc_id)

    # Check document ownership
    if document.user_id != current_user.id:
        flash('Unauthorized access', 'error')
        return redirect(url_for('routes.home'))

    if request.method == 'POST':
        query = request.form.get('query')
        if query:
            try:
                # Process with LLaMA
                analysis_result = process_document_with_llama(document.content, query)

                # Generate graphs if needed
                graphs = generate_analysis_graphs(analysis_result)

                return render_template('analysis_result.html',
                                       document=document,
                                       analysis=analysis_result,
                                       graphs=graphs)

            except Exception as e:
                flash('Error analyzing document. Please try again.', 'error')
                print(f"Analysis error: {e}")

    return render_template('analysis.html', document=document)


# User documents route
@routes_blueprint.route('/my-documents')
@login_required
def user_documents():
    """View user's documents route."""
    documents = Document.query.filter_by(user_id=current_user.id).all()
    return render_template('my_documents.html', documents=documents)


# Helper functions
def allowed_file(filename):
    """Check if file type is allowed."""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']