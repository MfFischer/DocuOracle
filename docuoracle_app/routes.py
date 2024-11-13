from flask import Blueprint, render_template, flash, redirect, url_for, request, current_app, send_file, \
    send_from_directory, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
import os
from datetime import datetime
# Import from your package
from docuoracle_app import db
from docuoracle_app.models import User, Document
from docuoracle_app.utils import parse_pdf, parse_word, parse_excel
from docuoracle_app.llama_handler import process_document_with_llama, analyze_excel_data, initialize_llama
from docuoracle_app.graph_handler import generate_plotly_graph, get_available_charts
from itsdangerous import URLSafeTimedSerializer
import pandas as pd  # Add this import
import plotly.express as px
import plotly.graph_objects as go

# Create Blueprint
routes_blueprint = Blueprint('routes', __name__)

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'xlsx', 'xls', 'csv'}

# Global variable to track Llama status
_llama_initialized = False

def get_llama_initialization_status():
    """Helper function to get Llama initialization status."""
    global _llama_initialized
    return {
        'initialized': _llama_initialized,
        'status': 'Connected' if _llama_initialized else 'Not Connected'
    }


def generate_visualizations(df, question=None):
    """Generate visualizations based on dataframe content and optional question."""
    try:
        visualizations = []

        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns

        if len(numeric_cols) >= 1:
            # Time series plot if there's a date column
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                date_col = date_cols[0]
                for num_col in numeric_cols[:2]:  # Limit to first 2 numeric columns
                    fig = px.line(df, x=date_col, y=num_col,
                                  title=f'{num_col} over time')
                    visualizations.append({
                        'type': 'time_series',
                        'plot': fig.to_html(full_html=False),
                        'description': f'Time series plot of {num_col}'
                    })

            # Distribution plots for numeric columns
            for col in numeric_cols[:2]:  # Limit to first 2 columns
                fig = px.histogram(df, x=col,
                                   title=f'Distribution of {col}')
                visualizations.append({
                    'type': 'distribution',
                    'plot': fig.to_html(full_html=False),
                    'description': f'Distribution of {col}'
                })

            # Correlation heatmap if multiple numeric columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix,
                                title='Correlation Matrix')
                visualizations.append({
                    'type': 'correlation',
                    'plot': fig.to_html(full_html=False),
                    'description': 'Correlation matrix between numeric variables'
                })

        # Categorical analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            for cat_col in categorical_cols[:2]:  # Limit to first 2 categorical columns
                value_counts = df[cat_col].value_counts()
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                             title=f'Distribution of {cat_col}')
                visualizations.append({
                    'type': 'categorical',
                    'plot': fig.to_html(full_html=False),
                    'description': f'Distribution of {cat_col} categories'
                })

        return visualizations

    except Exception as e:
        print(f"Error generating visualizations: {e}")
        return []


def generate_plotly_graph(df, chart_type='line', x_col=None, y_col=None, title=None,
                          color_scheme='default', animation=None, show_grid=True,
                          enable_zoom=False, show_labels=False, legend_position='right'):
    """Generate a Plotly graph based on specified parameters."""
    try:
        if not x_col or not y_col:
            raise ValueError("X and Y columns must be specified")

        # Set default title if none provided
        if not title:
            title = f'{chart_type.title()} Chart: {y_col} vs {x_col}'

        # Create figure based on chart type
        if chart_type == 'line':
            fig = px.line(df, x=x_col, y=y_col, title=title)
        elif chart_type == 'bar':
            fig = px.bar(df, x=x_col, y=y_col, title=title)
        elif chart_type == 'scatter':
            fig = px.scatter(df, x=x_col, y=y_col, title=title)
        elif chart_type == 'pie':
            fig = px.pie(df, names=x_col, values=y_col, title=title)
        else:
            fig = px.line(df, x=x_col, y=y_col, title=title)  # Default to line chart

        # Update layout based on parameters
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h" if legend_position in ['top', 'bottom'] else "v",
                x=1 if legend_position == 'right' else 0,
                y=1 if legend_position == 'top' else 0,
                xanchor="right" if legend_position == 'right' else "left",
                yanchor="top" if legend_position == 'top' else "bottom"
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(showgrid=show_grid, gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(showgrid=show_grid, gridcolor='rgba(128,128,128,0.2)')
        )

        # Enable zoom if requested
        if not enable_zoom:
            fig.update_layout(
                xaxis=dict(fixedrange=True),
                yaxis=dict(fixedrange=True)
            )

        # Add data labels if requested
        if show_labels:
            fig.update_traces(textposition='top center')

        # Apply color scheme
        if color_scheme != 'default':
            if color_scheme == 'dark':
                fig.update_layout(template='plotly_dark')
            elif color_scheme == 'light':
                fig.update_layout(template='plotly')

        # Add animation
        if animation:
            fig.update_layout(
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    buttons=[dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 500, "redraw": True},
                                     "fromcurrent": True}]
                    )]
                )]
            )

        return fig.to_html(full_html=False, include_plotlyjs=True)

    except Exception as e:
        print(f"Error generating plotly graph: {e}")
        return None


def get_available_charts():
    """Return list of available chart types."""
    return [
        {'value': 'line', 'label': 'Line Chart'},
        {'value': 'bar', 'label': 'Bar Chart'},
        {'value': 'scatter', 'label': 'Scatter Plot'},
        {'value': 'pie', 'label': 'Pie Chart'},
        {'value': 'area', 'label': 'Area Chart'},
        {'value': 'bubble', 'label': 'Bubble Chart'}
    ]

def allowed_file(filename):
    """Check if file type is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@routes_blueprint.route('/')
@routes_blueprint.route('/home')
def home():
    try:
        documents = []
        if current_user.is_authenticated:
            documents = Document.query.filter_by(user_id=current_user.id) \
                .order_by(Document.created_at.desc()) \
                .limit(5) \
                .all()
        return render_template('home.html',
                             documents=documents,
                             current_year=datetime.now().year)
    except Exception as e:
        print(f"Error in home route: {e}")
        flash(f"An error occurred: {str(e)}", "danger")
        return render_template('error.html', error=str(e))


@routes_blueprint.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            username = request.form.get('username')
            password = request.form.get('password')

            user = User.query.filter_by(username=username).first()

            if user and user.check_password(password):
                login_user(user)
                flash('Welcome back!', 'success')
                return redirect(url_for('routes.home'))

            flash('Invalid username or password', 'error')

        except Exception as e:
            flash(f'Login error: {str(e)}', 'error')

    return render_template('login.html')


@routes_blueprint.route('/register', methods=['GET', 'POST'])
def register():
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


@routes_blueprint.route('/logout')
@login_required
def logout():
    """User logout route."""
    logout_user()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('routes.home'))


@routes_blueprint.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    """Document upload and processing route."""
    try:
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

                    # Process document based on action type
                    return process_uploaded_document(doc, request.form)

                except Exception as e:
                    flash(f'Error processing file: {str(e)}', 'danger')
                    return redirect(request.url)

        return render_template('upload.html', chart_types=chart_types)

    except Exception as e:
        flash(f'Upload error: {str(e)}', 'danger')
        return render_template('error.html', error=str(e))


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


@routes_blueprint.route('/generate_visualization', methods=['POST'])
@login_required
def generate_visualization():
    try:
        data = request.get_json()
        doc_id = data.get('document_id')
        config = data.get('config', {})

        if not doc_id:
            return jsonify({
                'success': False,
                'error': 'Missing document ID'
            }), 400

        document = Document.query.get_or_404(doc_id)
        if document.user_id != current_user.id:
            return jsonify({
                'success': False,
                'error': 'Unauthorized access'
            }), 403

        # Read the data
        try:
            if document.filepath.endswith('.csv'):
                df = pd.read_csv(document.filepath)
            else:
                df = pd.read_excel(document.filepath)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Error reading file: {str(e)}'
            }), 500

        # Generate visualization
        chart_html = generate_plotly_graph(
            df=df,
            chart_type=config.get('chartType'),
            x_col=config.get('xColumn'),
            y_col=config.get('yColumn'),
            title=config.get('title'),
            color_scheme=config.get('colorScheme', 'default'),
            animation=config.get('animation', 'none'),
            show_grid=config.get('showGridLines', True),
            enable_zoom=config.get('enableZoom', False),
            show_labels=config.get('showDataLabels', False),
            legend_position=config.get('legendPosition', 'right')
        )

        if not chart_html:
            return jsonify({
                'success': False,
                'error': 'Failed to generate visualization'
            }), 500

        return jsonify({
            'success': True,
            'visualization': chart_html
        })

    except Exception as e:
        print(f"Error generating visualization: {e}")  # Add this for debugging
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@routes_blueprint.route('/process_document', methods=['POST'])
@login_required
def process_document():
    """Process document and return JSON response."""
    try:
        data = request.get_json()  # Changed from form to json
        document_id = data.get('document_id')
        question = data.get('question')

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

        # Handle different file types
        if document.filepath.endswith(('.xls', '.xlsx', '.csv')):
            # For Excel/CSV files, use pandas
            df = parse_excel(document.filepath)
            if df is not None:
                result = analyze_excel_data(df, question)
                if result['success']:
                    # Add visualization data if available
                    try:
                        viz_data = generate_visualizations(df, question)
                        result['visualizations'] = viz_data
                    except Exception as e:
                        print(f"Visualization error: {e}")
                return jsonify(result)
            else:
                return jsonify({
                    'success': False,
                    'error': 'Could not parse Excel/CSV file'
                }), 400
        else:
            # For other documents (PDF, DOCX)
            document_text = None
            if document.filepath.endswith('.pdf'):
                document_text = parse_pdf(document.filepath)
            elif document.filepath.endswith('.docx'):
                document_text = parse_word(document.filepath)

            if not document_text:
                return jsonify({
                    'success': False,
                    'error': 'Could not extract document content'
                }), 400

            result = process_document_with_llama(document_text, question)
            return jsonify({
                'success': True,
                'answer': result
            })

    except Exception as e:
        print(f"Document processing error: {e}")
        return jsonify({
            'success': False,
            'error': f'Error processing document: {str(e)}'
        }), 500

# Helper functions
def process_uploaded_document(document, form_data):
    """Helper function to process uploaded or existing document."""
    action = form_data.get('action', 'ask')
    question = form_data.get('question')

    document_text = None
    df = None
    answer = None
    graph = None

    # Extract content based on file type
    if document.filepath.endswith('.pdf'):
        document_text = parse_pdf(document.filepath)
    elif document.filepath.endswith('.docx'):
        document_text = parse_word(document.filepath)
    elif document.filepath.endswith(('.xls', '.xlsx', '.csv')):
        df = parse_excel(document.filepath)
        document_text = df.to_string() if df is not None else None

    # Handle Q&A
    if action == 'ask' and question and document_text:
        answer = process_document_with_llama(question, document_text)

    # Handle data analysis
    elif action == 'analyze' and df is not None:
        chart_type = form_data.get('chart_type')
        if chart_type:
            try:
                graph = generate_plotly_graph(
                    df,
                    chart_type=chart_type,
                    x_col=form_data.get('x_col'),
                    y_col=form_data.get('y_col'),
                    title=f"{chart_type.title()} Chart"
                )
            except Exception as e:
                flash(f'Error generating graph: {str(e)}', 'danger')

    documents = Document.query.filter_by(user_id=current_user.id) \
        .order_by(Document.created_at.desc()) \
        .limit(5) \
        .all()

    return render_template('home.html',
                           documents=documents,
                           answer=answer,
                           graph=graph,
                           question=question)


# API endpoints
@routes_blueprint.route('/api/initialize_llama', methods=['GET'])
@login_required
def initialize_llama_endpoint():
    """Initialize LLaMA model endpoint."""
    global _llama_initialized
    try:
        # Call your actual Llama initialization function
        success = initialize_llama()
        if success:
            _llama_initialized = True
            return jsonify({
                'success': True,
                'message': 'Llama initialized successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to initialize Llama'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error initializing Llama: {str(e)}'
        }), 500


# Debug routes
@routes_blueprint.route('/debug')
def debug():
    """Debug information route."""
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
    """Get current status of Llama model."""
    return jsonify(get_llama_initialization_status())

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

            return redirect(url_for('routes.settings'))

        return render_template('settings.html')

    except Exception as e:
        flash(f'Error updating settings: {str(e)}', 'error')
        return redirect(url_for('routes.home'))


@routes_blueprint.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Handle forgot password requests."""
    if request.method == 'POST':
        email = request.form.get('email')

        # Check if email exists
        user = User.query.filter_by(email=email).first()
        if user:
            # Generate reset token
            reset_token = generate_reset_token(user)

            # Send password reset email (implement this function)
            # send_reset_email(user.email, reset_token)

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

        # Verify token and update password
        user = verify_reset_token(token)
        if user:
            user.set_password(password)
            db.session.commit()
            flash('Your password has been updated.', 'success')
            return redirect(url_for('routes.login'))

        flash('Invalid or expired reset token.', 'error')
        return redirect(url_for('routes.forgot_password'))

    return render_template('reset_password.html')


def generate_reset_token(user):
    """Generate a password reset token."""
    # Implementation depends on your token generation method
    # This is a simple example
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


@routes_blueprint.route('/api/get_columns/<int:doc_id>')
@login_required
def get_columns(doc_id):
    """Get columns from document for visualization."""
    try:
        # Get document
        document = Document.query.get_or_404(doc_id)

        # Check ownership
        if document.user_id != current_user.id:
            return jsonify({
                'success': False,
                'error': 'Unauthorized access'
            }), 403

        # Check file type
        if not document.filepath.endswith(('.xlsx', '.xls', '.csv')):
            return jsonify({
                'success': False,
                'error': 'File type not supported for visualization'
            }), 400

        # Read file
        try:
            if document.filepath.endswith('.csv'):
                df = pd.read_csv(document.filepath)
            else:
                df = pd.read_excel(document.filepath)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Error reading file: {str(e)}'
            }), 500

        # Get columns that can be used for visualization
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        return jsonify({
            'success': True,
            'columns': {
                'numeric': numeric_cols,
                'datetime': datetime_cols,
                'categorical': categorical_cols,
                'all': numeric_cols + datetime_cols + categorical_cols
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500