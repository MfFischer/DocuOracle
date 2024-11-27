from flask import Blueprint, render_template, flash, redirect, url_for, request, current_app, \
    send_from_directory, jsonify, session
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
import os
import logging
import pandas as pd
from urllib.parse import urlsplit
import plotly.express as px
from docuoracle_app import db
from docuoracle_app.models import User, Document
from docuoracle_app.utils import parse_pdf, parse_word, parse_excel, parse_document, get_document_columns
from docuoracle_app.llama_handler import (
    initialize_llama,
    initialize_rag,
    process_document_with_llama,
    process_document_with_rag,
    analyze_excel_data,
    llama_handler
)
from docuoracle_app.graph_handler import (
    generate_visualizations
)
from itsdangerous import URLSafeTimedSerializer
from flask_wtf.csrf import CSRFError

# Create Blueprint
routes_blueprint = Blueprint('routes', __name__)

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'xlsx', 'xls', 'csv'}

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global variables for model management
language_model = None
_model_initialized = False


def get_model_initialization_status():
    """Helper function to get model initialization status."""
    global _model_initialized
    return {
        'initialized': _model_initialized,
        'status': 'Connected' if _model_initialized else 'Not Connected'
    }


def allowed_file(filename):
    """Check if file type is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_uploaded_file(file, user_id):
    """Helper function to save uploaded file."""
    try:
        filename = secure_filename(file.filename)
        upload_folder = current_app.config['UPLOAD_FOLDER']
        user_folder = os.path.join(upload_folder, str(user_id))
        os.makedirs(user_folder, exist_ok=True)

        filepath = os.path.join(user_folder, filename)
        file.save(filepath)

        # Save document in database
        doc = Document(
            filename=filename,
            filepath=filepath,
            user_id=user_id,
            file_type=filename.rsplit('.', 1)[1].lower()
        )
        db.session.add(doc)
        db.session.commit()

        return True, doc, None
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        return False, None, str(e)


@routes_blueprint.route('/')
def index():
    """Landing page route."""
    return render_template('index.html')


@routes_blueprint.route('/home')
@login_required  # Make sure this is protected
def home():
    """Dashboard/home page route."""
    try:
        documents = []
        model_status = {'initialized': False, 'status': 'Not Connected'}
        columns = []

        if current_user.is_authenticated:
            documents = Document.query.filter_by(user_id=current_user.id) \
                .order_by(Document.created_at.desc()) \
                .all()

            # Get model status
            model_status = get_model_initialization_status()

            # Get columns if there's a selected document
            selected_doc_id = session.get('selected_document_id')
            if selected_doc_id:
                doc = Document.query.get(selected_doc_id)
                if doc and doc.filepath.endswith(('.xlsx', '.xls', '.csv')):
                    try:
                        df = parse_excel(doc.filepath)
                        if df is not None:
                            columns = df.columns.tolist()
                    except Exception as e:
                        print(f"Error getting columns: {e}")

        return render_template('home.html',
                               documents=documents,
                               model_status=model_status,
                               columns=columns)

    except Exception as e:
        print(f"Error in home route: {e}")
        return render_template('error.html', error=str(e))


@routes_blueprint.route('/login', methods=['GET', 'POST'])
def login():
    """User login route."""
    if request.method == 'POST':
        try:
            username = request.form.get('username')
            password = request.form.get('password')

            user = User.query.filter_by(username=username).first()

            if user and user.check_password(password):
                login_user(user)
                flash('Welcome back!', 'success')

                # Get the next page from query parameters
                next_page = request.args.get('next')

                # Validate the next page URL
                if not next_page or urlsplit(next_page).netloc != '':
                    next_page = url_for('routes.home')

                return redirect(next_page)

            flash('Invalid username or password', 'error')

        except Exception as e:
            flash(f'Login error: {str(e)}', 'error')

    return render_template('login.html')


@routes_blueprint.route('/register', methods=['GET', 'POST'])
def register():
    """User registration route."""
    if request.method == 'POST':
        try:
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')

            if User.query.filter_by(username=username).first():
                flash('Username already exists', 'error')
                return redirect(url_for('routes.register'))

            if User.query.filter_by(email=email).first():
                flash('Email already registered', 'error')
                return redirect(url_for('routes.register'))

            user = User(username=username, email=email)
            user.set_password(password)

            db.session.add(user)
            db.session.commit()

            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('routes.login'))

        except Exception as e:
            db.session.rollback()
            flash(f'Registration error: {str(e)}', 'error')

    return render_template('register.html')


@routes_blueprint.route('/logout')
@login_required
def logout():
    """User logout route."""
    logout_user()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('routes.home'))


@routes_blueprint.route('/profile')
@login_required
def profile():
    """User profile route."""
    try:
        # Get user's documents count
        documents_count = Document.query.filter_by(user_id=current_user.id).count()

        # Get user's recent documents
        recent_documents = Document.query.filter_by(user_id=current_user.id) \
            .order_by(Document.created_at.desc()) \
            .limit(5) \
            .all()

        return render_template('profile.html',
                               user=current_user,
                               documents_count=documents_count,
                               recent_documents=recent_documents)
    except Exception as e:
        flash(f'Error loading profile: {str(e)}', 'error')
        return redirect(url_for('routes.home'))


@routes_blueprint.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    """User settings route."""
    try:
        if request.method == 'POST':
            action = request.form.get('action')

            if action == 'update_profile':
                # Update email
                new_email = request.form.get('email')
                if new_email and new_email != current_user.email:
                    if User.query.filter_by(email=new_email).first():
                        flash('Email already exists', 'error')
                    else:
                        current_user.email = new_email
                        db.session.commit()
                        flash('Email updated successfully', 'success')

                # Update username
                new_username = request.form.get('username')
                if new_username and new_username != current_user.username:
                    if User.query.filter_by(username=new_username).first():
                        flash('Username already exists', 'error')
                    else:
                        current_user.username = new_username
                        db.session.commit()
                        flash('Username updated successfully', 'success')

            elif action == 'change_password':
                current_password = request.form.get('current_password')
                new_password = request.form.get('new_password')
                confirm_password = request.form.get('confirm_password')

                if not current_password or not new_password or not confirm_password:
                    flash('All password fields are required', 'error')
                elif not current_user.check_password(current_password):
                    flash('Current password is incorrect', 'error')
                elif new_password != confirm_password:
                    flash('New passwords do not match', 'error')
                else:
                    current_user.set_password(new_password)
                    db.session.commit()
                    flash('Password updated successfully', 'success')

        return render_template('settings.html', user=current_user)

    except Exception as e:
        flash(f'Error updating settings: {str(e)}', 'error')
        return redirect(url_for('routes.home'))


@routes_blueprint.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    """Combined upload endpoint for both web and API requests."""
    if request.method == 'POST':
        # Check if it's an API request (JSON or files in request)
        is_api_request = request.is_json or request.files

        if 'file' not in request.files:
            message = 'No file part'
            return jsonify({'success': False, 'error': message}) if is_api_request else \
                redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            message = 'No selected file'
            return jsonify({'success': False, 'error': message}) if is_api_request else \
                redirect(request.url)

        if file and allowed_file(file.filename):
            success, doc, error = save_uploaded_file(file, current_user.id)

            if success:
                if is_api_request:
                    return jsonify({
                        'success': True,
                        'message': 'File uploaded successfully',
                        'document': {
                            'id': doc.id,
                            'filename': doc.filename
                        }
                    })
                else:
                    flash('Document uploaded successfully!', 'success')
                    return redirect(url_for('routes.view_document', doc_id=doc.id))
            else:
                if is_api_request:
                    return jsonify({'success': False, 'error': error})
                else:
                    flash(f'Error processing file: {error}', 'danger')
                    return redirect(request.url)
        else:
            message = 'File type not allowed'
            return jsonify({'success': False, 'error': message}) if is_api_request else \
                redirect(request.url)

    # GET request - render upload form
    return render_template('upload.html')


@routes_blueprint.route('/view_document/<int:doc_id>')
@login_required
def view_document(doc_id):
    """View document details route."""
    try:
        document = Document.query.get_or_404(doc_id)

        if document.user_id != current_user.id:
            flash('You do not have permission to view this document.', 'danger')
            return redirect(url_for('routes.home'))

        document_text = None
        df = None
        visualizations = []

        if document.filepath.endswith('.pdf'):
            document_text = parse_pdf(document.filepath)
        elif document.filepath.endswith('.docx'):
            document_text = parse_word(document.filepath)
        elif document.filepath.endswith(('.xls', '.xlsx', '.csv')):
            df = parse_excel(document.filepath)
            document_text = df.to_string() if df is not None else None
            if df is not None:
                visualizations = generate_visualizations(df)

        return render_template('view_document.html',
                               document=document,
                               document_text=document_text,
                               dataframe=df,
                               visualizations=visualizations)
    except Exception as e:
        flash(f'Error viewing document: {str(e)}', 'danger')
        return redirect(url_for('routes.home'))


@routes_blueprint.route('/delete_document/<int:doc_id>', methods=['POST'])
@login_required
def delete_document(doc_id):
    """Delete document route."""
    try:
        document = Document.query.get_or_404(doc_id)

        if document.user_id != current_user.id:
            flash('You do not have permission to delete this document.', 'danger')
            return redirect(url_for('routes.home'))

        if os.path.exists(document.filepath):
            os.remove(document.filepath)

        db.session.delete(document)
        db.session.commit()

        flash('Document deleted successfully.', 'success')
        return redirect(url_for('routes.home'))

    except Exception as e:
        flash(f'Error deleting document: {str(e)}', 'danger')
        return redirect(url_for('routes.home'))


@routes_blueprint.route('/select_document', methods=['POST'])
@login_required
def select_document():
    """Select document for analysis."""
    try:
        data = request.get_json()
        doc_id = data.get('document_id')

        if not doc_id:
            return jsonify({'error': 'No document ID provided'})

        document = Document.query.get(doc_id)
        if not document or document.user_id != current_user.id:
            return jsonify({'error': 'Document not found'})

        session['selected_document_id'] = doc_id

        # If it's a data file, return the columns
        columns = []
        if document.filepath.endswith(('.xlsx', '.xls', '.csv')):
            try:
                df = parse_excel(document.filepath)
                if df is not None:
                    columns = df.columns.tolist()
            except:
                pass

        return jsonify({
            'success': True,
            'filename': document.filename,
            'columns': columns
        })

    except Exception as e:
        return jsonify({'error': f'Error selecting document: {str(e)}'})


def process_uploaded_document(document, form_data):
    """Helper function to process uploaded document."""
    try:
        df = None
        document_text = None

        # Extract content based on file type
        if document.filepath.endswith('.pdf'):
            document_text = parse_pdf(document.filepath)
        elif document.filepath.endswith('.docx'):
            document_text = parse_word(document.filepath)
        elif document.filepath.endswith(('.xls', '.xlsx', '.csv')):
            df = parse_excel(document.filepath)
            if df is not None:
                document_text = df.to_string()

        # Generate automatic visualizations for data files
        visualizations = []
        if df is not None:
            visualizations = generate_visualizations(df)

        return document_text, df, visualizations

    except Exception as e:
        print(f"Error processing document: {e}")
        return None, None, []


@routes_blueprint.route('/api/initialize_model', methods=['POST'])
@login_required
def initialize_model_endpoint():
    """Initialize the language model with RAG support."""
    try:
        global _model_initialized

        # Get initialization type from request
        data = request.get_json()
        use_rag = data.get('use_rag', False)

        if use_rag:
            # Initialize with RAG
            requirements = {
                'deployment': data.get('deployment', 'production'),
                'resources': data.get('resources', 'limited')
            }
            success, message = initialize_rag(requirements)
        else:
            # Initialize traditional Llama
            success, message = initialize_llama()

        if success:
            _model_initialized = True
            logger.info("Model initialized successfully")
            return jsonify({
                'success': True,
                'message': message,
                'status': get_model_initialization_status(),
                'mode': 'rag' if use_rag else 'traditional'
            })
        else:
            logger.error(f"Model initialization failed: {message}")
            return jsonify({
                'success': False,
                'error': message,
                'status': get_model_initialization_status()
            })

    except Exception as e:
        logger.exception("Error during model initialization")
        return jsonify({
            'success': False,
            'error': str(e),
            'status': get_model_initialization_status()
        })


@routes_blueprint.route('/api/llama/status', methods=['GET'])
@login_required
def get_llama_status_endpoint():
    """Get current status of model."""
    try:
        status = get_model_initialization_status()
        return jsonify({
            'success': True,
            'initialized': status['initialized'],
            'status': status['status']
        })
    except Exception as e:
        logger.exception("Error getting model status")
        return jsonify({
            'success': False,
            'error': str(e),
            'initialized': False,
            'status': 'Error'
        })


@routes_blueprint.route('/api/generate_visualization', methods=['POST'])
@login_required
def generate_visualization():
    """Generate visualization from document data."""
    try:
        data = request.get_json()
        document_id = data.get('document_id')
        config = data.get('config', {})

        if not document_id:
            return jsonify({
                'success': False,
                'error': 'Missing document ID'
            })

        document = Document.query.get_or_404(document_id)
        if not document or document.user_id != current_user.id:
            return jsonify({
                'success': False,
                'error': 'Document not found or unauthorized'
            })

        # Parse the data file using your utility function
        df = parse_excel(document.filepath)
        if df is None:
            return jsonify({
                'success': False,
                'error': 'Could not read data file'
            })

        # Generate the visualization
        try:
            chart_type = config.get('chartType', 'scatter')
            x_col = config.get('xColumn')
            y_col = config.get('yColumn')
            theme = config.get('colorTheme', 'dark')

            # Validate columns
            if not x_col or not y_col:
                return jsonify({
                    'success': False,
                    'error': 'Missing X or Y axis columns'
                })

            if x_col not in df.columns or y_col not in df.columns:
                return jsonify({
                    'success': False,
                    'error': 'Selected columns not found in data'
                })

            # Create the appropriate plot
            if chart_type == 'line':
                fig = px.line(df, x=x_col, y=y_col)
            elif chart_type == 'bar':
                fig = px.bar(df, x=x_col, y=y_col)
            elif chart_type == 'scatter':
                fig = px.scatter(df, x=x_col, y=y_col)
            elif chart_type == 'pie':
                fig = px.pie(df, names=x_col, values=y_col)
            else:
                return jsonify({
                    'success': False,
                    'error': f'Unsupported chart type: {chart_type}'
                })

            # Update layout based on theme
            fig.update_layout(
                template='plotly_dark' if theme == 'dark' else 'plotly_white',
                paper_bgcolor='rgba(0,0,0,0)' if theme == 'dark' else 'white',
                plot_bgcolor='rgba(0,0,0,0)' if theme == 'dark' else 'white'
            )

            # Convert to HTML
            visualization_html = fig.to_html(
                full_html=False,
                include_plotlyjs=True,
                config={'responsive': True}
            )

            return jsonify({
                'success': True,
                'visualization': visualization_html
            })

        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return jsonify({
                'success': False,
                'error': f'Error generating visualization: {str(e)}'
            })

    except Exception as e:
        logger.error(f"Error in visualization endpoint: {e}")
        return jsonify({
            'success': False,
            'error': f'Error processing request: {str(e)}'
        })


@routes_blueprint.route('/process_document', methods=['POST'])
@login_required
def process_document():
    """Process document with RAG or traditional approach."""
    try:
        data = request.get_json()
        document_id = data.get('document_id')
        question = data.get('question')
        use_rag = data.get('use_rag', False)  # New parameter

        logger.debug(f"Processing document {document_id} with question: {question}, RAG: {use_rag}")

        if not document_id or not question:
            return jsonify({
                'success': False,
                'error': 'Missing document or question'
            }), 400

        document = Document.query.get_or_404(document_id)

        if document.user_id != current_user.id:
            return jsonify({
                'success': False,
                'error': 'Unauthorized access'
            }), 403

        # Parse document
        document_content = parse_document(document.filepath)
        if document_content is None:
            return jsonify({
                'success': False,
                'error': 'Could not parse document'
            }), 400

        # Process based on content type and RAG preference
        if isinstance(document_content, pd.DataFrame):
            # Excel/CSV data
            result = analyze_excel_data(document_content, question)
            return jsonify(result)
        else:
            # Text document (PDF/DOCX)
            if use_rag:
                result = process_document_with_rag(document_content, question)
            else:
                result = process_document_with_llama(document_content, question)

            return jsonify(result)

    except Exception as e:
        logger.error(f"Document processing error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error processing document: {str(e)}'
        }), 500


@routes_blueprint.route('/api/get_columns/<int:doc_id>')
@login_required
def get_columns(doc_id):
    """Get columns from document for visualization."""
    try:
        document = Document.query.get_or_404(doc_id)
        logger.debug(f"Getting columns for document: {document.filename}")

        if document.user_id != current_user.id:
            return jsonify({
                'success': False,
                'error': 'Unauthorized access'
            }), 403

        if not document.filepath.endswith(('.xlsx', '.xls', '.csv')):
            return jsonify({
                'success': False,
                'error': 'File type not supported for visualization'
            }), 400

        # Use the get_document_columns function
        result = get_document_columns(document.filepath)

        if result['success']:
            return jsonify(result)
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Failed to parse file')
            }), 500

    except Exception as e:
        logger.error(f"Error getting columns: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Error handlers
@routes_blueprint.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return render_template('errors/404.html'), 404


@routes_blueprint.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    db.session.rollback()
    return render_template('errors/500.html'), 500


# Password reset functionality
@routes_blueprint.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Handle forgot password requests."""
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()

        if user:
            reset_token = generate_reset_token(user)
            # TODO: Implement email sending
            flash('Password reset instructions have been sent to your email.', 'info')
            return redirect(url_for('routes.login'))

        flash('Email address not found.', 'error')
    return render_template('forgot_password.html')


@routes_blueprint.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    """Handle password reset."""
    if request.method == 'POST':
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('routes.reset_password', token=token))

        user = verify_reset_token(token)
        if user:
            user.set_password(password)
            db.session.commit()
            flash('Your password has been updated.', 'success')
            return redirect(url_for('routes.login'))

        flash('Invalid or expired reset token.', 'error')
        return redirect(url_for('routes.forgot_password'))

    return render_template('reset_password.html')


# Utility functions
def generate_reset_token(user):
    """Generate a password reset token."""
    serializer = URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
    return serializer.dumps(user.email, salt='password-reset-salt')


def verify_reset_token(token, expiration=3600):
    """Verify the reset token."""
    serializer = URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
    try:
        email = serializer.loads(
            token,
            salt='password-reset-salt',
            max_age=expiration
        )
        return User.query.filter_by(email=email).first()
    except:
        return None


# Debug routes
@routes_blueprint.route('/debug')
def debug():
    """Debug information route."""
    if not current_app.debug:
        return redirect(url_for('routes.home'))

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


@routes_blueprint.route('/debug/static')
def debug_static():
    """Debug static files route."""
    if not current_app.debug:
        return redirect(url_for('routes.home'))

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


@routes_blueprint.route('/static/<path:filename>')
def static_files(filename):
    """Static files route."""
    print(f"Requesting static file: {filename}")
    return send_from_directory('static', filename)


@routes_blueprint.route('/api/llama/status', methods=['GET'])
@login_required
def get_llama_status():
    """Get current status of model."""
    return jsonify(get_model_initialization_status())


# Additional utility functions for document processing
def analyze_data_file(df, question):
    """Analyze data files with RAG support."""
    try:
        data_description = f"""
        The data contains {len(df)} rows and {len(df.columns)} columns.
        Columns: {', '.join(df.columns.tolist())}
        """

        prompt = f"""<s>[INST] Analyze the following data and answer the question.

Data Description: {data_description}
Sample Data (first few rows):
{df.head().to_string()}

Question: {question}

Provide a clear analysis with relevant statistics if applicable. [/INST]"""

        if hasattr(llama_handler, 'rag_config') and llama_handler.rag_config:
            result = process_document_with_rag(prompt, question, is_data_analysis=True)
        else:
            result = process_document_with_llama(prompt, question, is_data_analysis=True)

        visualizations = generate_visualizations(df)

        if result.get('success', False):
            return {
                'success': True,
                'answer': result.get('answer'),
                'sources': result.get('sources', []),
                'visualizations': visualizations,
                'type': result.get('type', 'traditional_analysis')
            }
        else:
            return {
                'success': False,
                'error': result.get('error', 'Analysis failed'),
                'visualizations': visualizations
            }

    except Exception as e:
        return {
            'success': False,
            'error': f'Error analyzing data: {str(e)}'
        }


@routes_blueprint.errorhandler(CSRFError)
def handle_csrf_error(e):
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':  # If AJAX request
        return jsonify({
            'success': False,
            'error': 'CSRF token is missing or invalid'
        }), 400
    else:  # If regular form submission
        flash('Session has expired. Please try again.', 'error')
        return redirect(url_for('routes.login')), 400
