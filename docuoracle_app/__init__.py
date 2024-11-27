from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager, current_user
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
from flask_wtf.csrf import CSRFProtect

# Load environment variables
load_dotenv()

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
csrf = CSRFProtect()
logger = logging.getLogger(__name__)


# Initialize Flask-Login
@login_manager.user_loader
def load_user(user_id):
    """User loader for Flask-Login."""
    from .models import User
    try:
        return User.query.get(int(user_id))
    except Exception as e:
        logger.error(f"Error loading user: {e}")
        return None


def init_llama(app):
    """Initialize LLM based on configuration."""
    from .llama_handler import initialize_llama, initialize_rag, llama_handler

    try:
        if app.config['MODEL_PROVIDER'] == 'hf_space':
            # Initialize with RAG for Hugging Face
            requirements = {
                'deployment': 'production',
                'resources': 'limited' if app.config.get('RESOURCES_LIMITED', True) else 'full'
            }
            success, message = initialize_rag(requirements)
            if not success:
                app.logger.error(f"Failed to initialize RAG: {message}")
        else:
            # Initialize traditional local model
            success, message = initialize_llama()
            if not success:
                app.logger.error(f"Failed to initialize local model: {message}")

        return success, message
    except Exception as e:
        app.logger.error(f"Error initializing LLM: {e}")
        return False, str(e)


def create_app():
    """Application factory for creating the Flask app."""
    base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

    # Define app directories with absolute paths
    app_dir = os.path.join(base_dir, 'docuoracle_app')
    template_dir = os.path.join(app_dir, 'templates')
    static_dir = os.path.join(app_dir, 'static')
    upload_dir = os.path.join(base_dir, 'uploads')
    models_dir = os.path.join(base_dir, 'models')

    # Use AppData for database storage (more reliable on Windows)
    appdata_dir = os.path.join(os.environ.get('APPDATA', base_dir), 'DocuOracle')

    print(f"Database directory will be: {appdata_dir}")  # Debug print

    # Ensure required directories exist
    for directory in [
        template_dir,
        static_dir,
        upload_dir,
        models_dir,
        os.path.join(static_dir, 'css'),
        appdata_dir
    ]:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Created/verified directory: {directory}")
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")

    # Initialize Flask app
    app = Flask(__name__,
                template_folder=template_dir,
                static_folder=static_dir,
                static_url_path='/static')

    # Configure database with Windows-friendly path
    db_path = os.path.join(appdata_dir, 'docuoracle.db')

    # Convert Windows path to SQLAlchemy format
    db_uri = f'sqlite:///{db_path.replace(os.sep, "/")}'
    print(f"Database URI: {db_uri}")  # Debug print

    # Load configuration
    app.config.update(
        SECRET_KEY=os.getenv('SECRET_KEY', 'your-default-secret-key'),
        SQLALCHEMY_DATABASE_URI=db_uri,
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        UPLOAD_FOLDER=upload_dir,

        # Session Configuration
        PERMANENT_SESSION_LIFETIME=timedelta(minutes=30),  # 30 minutes session lifetime
        SESSION_COOKIE_SECURE=True,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE='Lax',

        # Model Configuration
        HF_TOKEN=os.getenv('HF_TOKEN'),
        MODEL_PROVIDER=os.getenv('MODEL_PROVIDER', 'hf_space'),
        RESOURCES_LIMITED=os.getenv('RESOURCES_LIMITED', 'true').lower() == 'true',
        MODEL_DEPLOYMENT=os.getenv('MODEL_DEPLOYMENT', 'production'),

        # RAG Configuration
        RAG_CHUNK_SIZE=int(os.getenv('RAG_CHUNK_SIZE', 500)),
        RAG_CHUNK_OVERLAP=int(os.getenv('RAG_CHUNK_OVERLAP', 50)),
        RAG_TOP_K=int(os.getenv('RAG_TOP_K', 3)),
        RAG_MODEL=os.getenv('RAG_MODEL', 'facebook/opt-350m'),
        RAG_EMBEDDINGS_MODEL=os.getenv('RAG_EMBEDDINGS_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),

        # Processing Configuration
        MAX_LENGTH=int(os.getenv('MAX_LENGTH', 512)),
        TEMPERATURE=float(os.getenv('TEMPERATURE', 0.7)),
        BATCH_SIZE=int(os.getenv('BATCH_SIZE', 8)),
        CONCURRENT_REQUESTS=int(os.getenv('CONCURRENT_REQUESTS', 4)),

        # Security Configuration
        MAX_CONTENT_LENGTH=int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024)),
        ALLOWED_EXTENSIONS=set(os.getenv('ALLOWED_EXTENSIONS', 'pdf,docx,xlsx,xls,csv').split(',')),

        WTF_CSRF_ENABLED=True,
        WTF_CSRF_SECRET_KEY=os.getenv('CSRF_SECRET_KEY', 'your-csrf-secret-key')
    )

    # Session handler
    @app.before_request
    def before_request():
        if current_user.is_authenticated:
            session.permanent = True
            app.permanent_session_lifetime = timedelta(minutes=30)
            session.modified = True

        # Check if session is expired for protected routes
        if not request.is_json:  # Skip for API requests
            if not current_user.is_authenticated and \
                    request.endpoint and \
                    request.endpoint not in ['routes.login', 'routes.register', 'static'] and \
                    not request.path.startswith('/static/'):
                flash('Your session has expired. Please login again.', 'info')
                return redirect(url_for('routes.login'))

    # Configure logging
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if os.getenv('ENABLE_FILE_LOGGING', 'false').lower() == 'true':
        file_handler = logging.FileHandler(os.getenv('LOG_FILE', 'app.log'))
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        app.logger.addHandler(file_handler)

    # Initialize extensions with app
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    csrf.init_app(app)

    # Configure login manager
    login_manager.login_view = 'routes.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'
    login_manager.refresh_view = 'routes.login'
    login_manager.needs_refresh_message = 'Please login again to confirm your identity.'
    login_manager.needs_refresh_message_category = 'info'
    login_manager.session_protection = "strong"

    # Create or update styles.css
    styles_css_path = os.path.join(static_dir, 'css', 'styles.css')
    with open(styles_css_path, 'w') as f:
        f.write("""/* Base styles */
:root {
    --primary-bg: #1a1a1a;
    --secondary-bg: #2d2d2d;
    --accent-color: #60a5fa;
    --text-color: #ffffff;
}

/* General body styling */
body {
    background-color: var(--primary-bg);
    color: var(--text-color);
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
}

/* Navbar styling */
.navbar {
    background-color: var(--secondary-bg);
    padding: 10px;
    text-align: center;
}

.navbar a {
    color: var(--accent-color);
    text-decoration: none;
    margin: 0 15px;
    font-size: 18px;
}

.navbar a:hover {
    text-decoration: underline;
}

/* Form styling */
form {
    max-width: 400px;
    margin: 50px auto;
    padding: 20px;
    background-color: var(--secondary-bg);
    border-radius: 8px;
}

form input[type="text"],
form input[type="password"],
form input[type="email"],
form input[type="file"] {
    width: 100%;
    padding: 10px;
    margin-bottom: 15px;
    border: none;
    border-radius: 4px;
    background-color: #333;
    color: #fff;
}

form button {
    background-color: var(--accent-color);
    border: none;
    color: #fff;
    padding: 10px;
    cursor: pointer;
    border-radius: 4px;
    width: 100%;
}

form button:hover {
    background-color: #4b9bff;
}

/* Flash messages */
.alert {
    padding: 10px;
    margin: 10px 0;
    border-radius: 4px;
}

.alert-success {
    background-color: #4CAF50;
    color: white;
}

.alert-danger {
    background-color: #f44336;
    color: white;
}

/* Utility classes */
.container {
    padding: 20px;
    max-width: 1200px;
    margin: 0 auto;
}

.fade-in {
    animation: fadeIn 0.3s ease-out;
}

.fade-out {
    animation: fadeOut 0.3s ease-in forwards;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(45, 45, 45, 0.5);
}

::-webkit-scrollbar-thumb {
    background: rgba(96, 165, 250, 0.5);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(96, 165, 250, 0.7);
}
""")

        # Register blueprints and initialize database
        with app.app_context():
            try:
                # Test database connection before creating tables
                with app.app_context():
                    try:
                        db.engine.connect()
                        print("Database connection successful")
                    except Exception as e:
                        print(f"Database connection failed: {e}")
                        raise

                    # Initialize database
                    db.create_all()
                    print("Database tables created successfully")

                # Initialize LLM
                init_llama(app)

                # Register blueprints
                from .routes import routes_blueprint
                app.register_blueprint(routes_blueprint)

                from .api import api_blueprint
                app.register_blueprint(api_blueprint, url_prefix='/api')

                try:
                    from .swagger_config import swaggerui_blueprint
                    app.register_blueprint(swaggerui_blueprint, url_prefix='/swagger')
                except ImportError:
                    app.logger.warning("Swagger UI blueprint not available")

            except Exception as e:
                app.logger.error(f"Error during app initialization: {e}")
                print(f"Detailed error during initialization: {str(e)}")
                raise

        # Add template context processors
        @app.context_processor
        def utility_processor():
            return {
                'current_year': datetime.now().year,
                'app_name': 'DocuOracle'
            }

        # Error handlers
        @app.errorhandler(404)
        def not_found_error():
            return render_template('errors/404.html'), 404

        @app.errorhandler(500)
        def internal_error():
            db.session.rollback()
            return render_template('errors/500.html'), 500

        return app


# Export components
__all__ = [
    'db',
    'migrate',
    'login_manager',
    'csrf',
    'create_app',
    'load_user'
]

# Make llama_handler available
try:
    from .llama_handler import llama_handler

    __all__.append('llama_handler')
except ImportError:
    logger.warning("Could not import llama_handler")
