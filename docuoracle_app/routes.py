from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, send_file, \
    send_from_directory, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from . import db
from .models import User, Document
from .utils import parse_pdf, parse_word, parse_excel
from .llama_handler import process_document_with_llama, llama_handler, initialize_llama
from .graph_handler import generate_plotly_graph, get_available_charts
import os

# Add debug prints for template directory
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
print("=" * 50)
print("Routes.py initialization")
print(f"Current working directory: {os.getcwd()}")
print(f"Template directory: {template_dir}")
print(f"Templates folder exists: {os.path.exists(template_dir)}")
if os.path.exists(template_dir):
    print(f"Files in templates folder: {os.listdir(template_dir)}")
    base_html = os.path.join(template_dir, 'base.html')
    if os.path.exists(base_html):
        print(f"base.html exists and permissions: {oct(os.stat(base_html).st_mode)[-3:]}")
        try:
            with open(base_html, 'r', encoding='utf-8') as f:
                print("First line of base.html:", f.readline().strip())
        except Exception as e:
            print(f"Error reading base.html: {e}")
print("=" * 50)

routes_blueprint = Blueprint('routes', __name__)

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'xlsx', 'xls', 'csv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@routes_blueprint.route('/')
def home():
    try:
        documents = []
        if current_user.is_authenticated:
            documents = Document.query.filter_by(user_id=current_user.id) \
                .order_by(Document.created_at.desc()) \
                .limit(5) \
                .all()

        return render_template('home.html', documents=documents)
    except Exception as e:
        print(f"Error in home route: {e}")
        flash(f"An error occurred: {str(e)}", "danger")
        return render_template('error.html', error=str(e))


@routes_blueprint.route('/view_document/<int:doc_id>')
@login_required
def view_document(doc_id):
    try:
        document = Document.query.get_or_404(doc_id)

        # Check if the document belongs to the current user
        if document.user_id != current_user.id:
            flash('You do not have permission to view this document.', 'danger')
            return redirect(url_for('routes.home'))

        # Read and parse the document based on its type
        document_text = None
        df = None

        if document.filepath.endswith('.pdf'):
            document_text = parse_pdf(document.filepath)
        elif document.filepath.endswith('.docx'):
            document_text = parse_word(document.filepath)
        elif document.filepath.endswith(('.xls', '.xlsx', '.csv')):
            df = parse_excel(document.filepath)
            document_text = df.to_string() if df is not None else None

        return render_template('view_document.html',
                               document=document,
                               document_text=document_text,
                               dataframe=df)
    except Exception as e:
        flash(f'Error viewing document: {str(e)}', 'danger')
        return redirect(url_for('routes.home'))


@routes_blueprint.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('routes.home'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        try:
            user = User.query.filter_by(username=username).first()

            if user and user.check_password(password):
                login_user(user)
                flash('Logged in successfully.', 'success')
                next_page = request.args.get('next')
                return redirect(next_page if next_page else url_for('routes.upload'))

            flash('Invalid username or password.', 'danger')
        except Exception as e:
            flash(f'Login error: {str(e)}', 'danger')

    return render_template('login.html')


@routes_blueprint.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('routes.home'))

    if request.method == 'POST':
        try:
            username = request.form.get('username')
            password = request.form.get('password')
            email = request.form.get('email')

            if not all([username, password, email]):
                flash('All fields are required.', 'danger')
                return redirect(url_for('routes.register'))

            if User.query.filter_by(username=username).first():
                flash('Username already exists.', 'danger')
                return redirect(url_for('routes.register'))

            if User.query.filter_by(email=email).first():
                flash('Email already registered.', 'danger')
                return redirect(url_for('routes.register'))

            user = User(username=username, email=email)
            user.set_password(password)

            db.session.add(user)
            db.session.commit()

            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('routes.login'))
        except Exception as e:
            db.session.rollback()
            flash(f'Registration error: {str(e)}', 'danger')

    return render_template('register.html')


@routes_blueprint.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('routes.home'))


@routes_blueprint.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    try:
        # Get available chart types for the template
        chart_types = get_available_charts()

        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file part', 'danger')
                return redirect(request.url)

            file = request.files['file']
            if file.filename == '':
                flash('No selected file', 'danger')
                return redirect(request.url)

            if file and allowed_file(file.filename):
                try:
                    filename = secure_filename(file.filename)
                    upload_folder = os.path.join(current_app.config['UPLOAD_FOLDER'], str(current_user.id))
                    os.makedirs(upload_folder, exist_ok=True)
                    filepath = os.path.join(upload_folder, filename)
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

                    # Get the action type (ask or analyze)
                    action = request.form.get('action', 'ask')
                    question = request.form.get('question')

                    document_text = None
                    df = None
                    answer = None
                    graph = None

                    # Extract content based on file type
                    if filename.endswith('.pdf'):
                        document_text = parse_pdf(filepath)
                    elif filename.endswith('.docx'):
                        document_text = parse_word(filepath)
                    elif filename.endswith(('.xls', '.xlsx', '.csv')):
                        df = parse_excel(filepath)
                        document_text = df.to_string() if df is not None else None

                    # Handle Q&A if question is provided
                    if action == 'ask' and question and document_text:
                        answer = process_document_with_llama(question, document_text)

                    # Handle data analysis if requested
                    elif action == 'analyze' and df is not None:
                        chart_type = request.form.get('chart_type')
                        x_col = request.form.get('x_col')
                        y_col = request.form.get('y_col')

                        if all([chart_type, x_col, y_col]):
                            additional_options = {}

                            if chart_type in ['bubble', 'scatter']:
                                additional_options.update({
                                    'size_col': request.form.get('size_col'),
                                    'color_col': request.form.get('color_col')
                                })
                            elif chart_type == 'heatmap':
                                additional_options.update({
                                    'x_axis': request.form.get('x_axis'),
                                    'y_axis': request.form.get('y_axis'),
                                    'values': request.form.get('values')
                                })
                            elif chart_type == 'candlestick':
                                additional_options.update({
                                    'open_col': request.form.get('open_col'),
                                    'high_col': request.form.get('high_col'),
                                    'low_col': request.form.get('low_col'),
                                    'close_col': request.form.get('close_col')
                                })

                            try:
                                graph = generate_plotly_graph(
                                    df,
                                    chart_type=chart_type,
                                    x_col=x_col,
                                    y_col=y_col,
                                    title=f"{chart_type.title()} Chart",
                                    **additional_options
                                )
                            except Exception as e:
                                flash(f'Error generating graph: {str(e)}', 'danger')

                    return render_template('upload.html',
                                           answer=answer,
                                           graph=graph,
                                           filename=filename,
                                           chart_types=chart_types,
                                           columns=df.columns.tolist() if df is not None else None)

                except Exception as e:
                    flash(f'Error processing file: {str(e)}', 'danger')
                    return redirect(request.url)

        return render_template('upload.html', chart_types=chart_types)

    except Exception as e:
        flash(f'Upload error: {str(e)}', 'danger')
        return render_template('error.html', error=str(e))


@routes_blueprint.route('/debug')
def debug():
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
    debug_info = {
        'current_dir': os.getcwd(),
        'template_dir': template_dir,
        'template_dir_exists': os.path.exists(template_dir),
        'template_files': os.listdir(template_dir) if os.path.exists(template_dir) else [],
        'base_html_path': os.path.join(template_dir, 'base.html'),
        'base_html_exists': os.path.exists(os.path.join(template_dir, 'base.html'))
    }
    return '<pre>' + '\n'.join(f'{k}: {v}' for k, v in debug_info.items()) + '</pre>'


@routes_blueprint.route('/static/<path:filename>')
def static_files(filename):
    print(f"Requesting static file: {filename}")
    return send_from_directory('static', filename)


@routes_blueprint.route('/debug/static')
def debug_static():
    static_dir = os.path.join(current_app.root_path, 'static')
    css_dir = os.path.join(static_dir, 'css')

    debug_info = {
        'static_folder': current_app.static_folder,
        'static_dir_exists': os.path.exists(static_dir),
        'css_dir_exists': os.path.exists(css_dir),
        'static_files': os.listdir(static_dir) if os.path.exists(static_dir) else [],
        'css_files': os.listdir(css_dir) if os.path.exists(css_dir) else [],
        'css_url': url_for('static', filename='css/styles.css')
    }

    return f'<pre>{debug_info}</pre>'

@routes_blueprint.route('/api/initialize_llama', methods=['GET'])
@login_required
def initialize_llama_endpoint():
    success, message = initialize_llama()
    return jsonify({
        'success': success,
        'message': message
    })


@routes_blueprint.route('/delete_document/<int:doc_id>', methods=['POST'])
@login_required
def delete_document(doc_id):
    try:
        document = Document.query.get_or_404(doc_id)

        # Check if document belongs to current user
        if document.user_id != current_user.id:
            flash('You do not have permission to delete this document.', 'danger')
            return redirect(url_for('routes.home'))

        # Delete the physical file
        if os.path.exists(document.filepath):
            os.remove(document.filepath)

        # Delete from database
        db.session.delete(document)
        db.session.commit()

        flash('Document deleted successfully.', 'success')
        return redirect(url_for('routes.home'))

    except Exception as e:
        flash(f'Error deleting document: {str(e)}', 'danger')
        return redirect(url_for('routes.home'))

# Add this to your routes.py

@routes_blueprint.route('/process_document', methods=['POST'])
@login_required
def process_document():
    try:
        document_id = request.form.get('document_id')
        question = request.form.get('question')
        action = request.form.get('action')

        if not document_id:
            flash('Please select a document.', 'danger')
            return redirect(url_for('routes.home'))

        document = Document.query.get_or_404(document_id)

        # Check if document belongs to current user
        if document.user_id != current_user.id:
            flash('You do not have permission to access this document.', 'danger')
            return redirect(url_for('routes.home'))

        # Extract document content based on file type
        document_text = None
        df = None

        if document.filepath.endswith('.pdf'):
            document_text = parse_pdf(document.filepath)
        elif document.filepath.endswith('.docx'):
            document_text = parse_word(document.filepath)
        elif document.filepath.endswith(('.xls', '.xlsx', '.csv')):
            df = parse_excel(document.filepath)
            document_text = df.to_string() if df is not None else None

        answer = None
        graph = None

        # Process based on action type
        if action == 'ask' and question and document_text:
            answer = process_document_with_llama(question, document_text)
        elif action == 'analyze' and df is not None:
            # Handle data analysis similar to upload route
            chart_type = request.form.get('chart_type')
            if chart_type:
                try:
                    graph = generate_plotly_graph(
                        df,
                        chart_type=chart_type,
                        x_col=request.form.get('x_col'),
                        y_col=request.form.get('y_col'),
                        title=f"{chart_type.title()} Chart"
                    )
                except Exception as e:
                    flash(f'Error generating graph: {str(e)}', 'danger')

        documents = Document.query.filter_by(user_id=current_user.id).order_by(Document.created_at.desc()).limit(5).all()
        return render_template('home.html',
                             documents=documents,
                             answer=answer,
                             graph=graph,
                             question=question)

    except Exception as e:
        flash(f'Error processing document: {str(e)}', 'danger')
        return redirect(url_for('routes.home'))