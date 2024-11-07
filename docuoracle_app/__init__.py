from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
import os

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()


def create_app():
    """Application factory for creating the Flask app."""
    # Update template and static folder paths
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))

    print("=" * 50)
    print("Static Directory Configuration")
    print(f"Static directory path: {static_dir}")
    print(f"Static directory exists: {os.path.exists(static_dir)}")

    if os.path.exists(static_dir):
        print(f"Static files: {os.listdir(static_dir)}")
        css_dir = os.path.join(static_dir, 'css')
        if os.path.exists(css_dir):
            print(f"CSS directory exists: True")
            print(f"CSS files: {os.listdir(css_dir)}")
        else:
            print("CSS directory does not exist")
    print("=" * 50)

    app = Flask(__name__,
                template_folder=template_dir,
                static_folder=static_dir,
                static_url_path='/static')  # Explicitly set static URL path

    # Load configuration
    from config import Config
    app.config.from_object(Config)
    Config.init_app(app)

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)

    # Configure login manager
    login_manager.login_view = 'routes.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'

    # Create static directories and ensure styles.css exists
    css_dir = os.path.join(static_dir, 'css')
    os.makedirs(css_dir, exist_ok=True)

    # Create styles.css if it doesn't exist
    styles_css_path = os.path.join(css_dir, 'styles.css')
    if not os.path.exists(styles_css_path):
        with open(styles_css_path, 'w') as f:
            f.write("""/* Base styles */
.navbar {
    background-color: #2d2d2d;
}

/* Custom styles */
.fade-in {
    animation: fadeIn 0.5s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
""")
        print(f"Created styles.css at {styles_css_path}")

    with app.app_context():
        try:
            # Import and register routes blueprint
            from .routes import routes_blueprint
            app.register_blueprint(routes_blueprint)
            print("Main routes registered successfully")

            # Import and register API blueprint
            from .api import api_blueprint
            app.register_blueprint(api_blueprint, url_prefix='/api')
            print("API routes registered successfully")

            # Import and register Swagger UI blueprint
            from .swagger_config import swaggerui_blueprint
            app.register_blueprint(swaggerui_blueprint, url_prefix='/swagger')
            print("Swagger UI registered successfully")

            # Add debug route for static files
            @app.route('/debug/static')
            def debug_static():
                debug_info = {
                    'static_folder': app.static_folder,
                    'static_url_path': app.static_url_path,
                    'css_path': os.path.join(app.static_folder, 'css', 'styles.css'),
                    'css_exists': os.path.exists(os.path.join(app.static_folder, 'css', 'styles.css')),
                    'static_files': os.listdir(app.static_folder) if os.path.exists(app.static_folder) else [],
                    'css_files': os.listdir(os.path.join(app.static_folder, 'css')) if os.path.exists(
                        os.path.join(app.static_folder, 'css')) else []
                }
                return f'<pre>{str(debug_info)}</pre>'

        except Exception as e:
            print(f"Error registering blueprints: {e}")
            raise

    return app


@login_manager.user_loader
def load_user(user_id):
    from .models import User
    return User.query.get(int(user_id))