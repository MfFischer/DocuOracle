from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
import os
from datetime import datetime
from flask import Flask, render_template

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()


def create_app():
    """Application factory for creating the Flask app."""
    # Get absolute paths for templates and static directories
    app_dir = os.path.abspath(os.path.dirname(__file__))
    template_dir = os.path.join(app_dir, 'templates')
    static_dir = os.path.join(app_dir, 'static')

    # Ensure required directories exist
    os.makedirs(template_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(os.path.join(static_dir, 'css'), exist_ok=True)
    os.makedirs(os.path.join(static_dir, 'js'), exist_ok=True)
    os.makedirs(os.path.join(static_dir, 'img'), exist_ok=True)

    # Initialize Flask app
    app = Flask(__name__,
                template_folder=template_dir,
                static_folder=static_dir,
                static_url_path='/static')

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

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeOut {
    from { opacity: 1; transform: translateY(0); }
    to { opacity: 0; transform: translateY(-10px); }
}

.fade-in {
    animation: fadeIn 0.3s ease-out;
}

.fade-out {
    animation: fadeOut 0.3s ease-in forwards;
}

/* Custom scrollbar */
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

/* Utility classes */
.backdrop-blur {
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}
""")

    with app.app_context():
        try:
            # Register blueprints
            from .routes import routes_blueprint
            app.register_blueprint(routes_blueprint)

            from .api import api_blueprint
            app.register_blueprint(api_blueprint, url_prefix='/api')

            from .swagger_config import swaggerui_blueprint
            app.register_blueprint(swaggerui_blueprint, url_prefix='/swagger')

            # Add template context processors
            @app.context_processor
            def utility_processor():
                return {
                    'current_year': datetime.now().year,
                    'app_name': 'DocuOracle'
                }

            # Add error handlers
            @app.errorhandler(404)
            def not_found_error(error):
                return render_template('errors/404.html'), 404

            @app.errorhandler(500)
            def internal_error(error):
                db.session.rollback()
                return render_template('errors/500.html'), 500

            # Debug routes
            if app.debug:
                @app.route('/debug/static')
                def debug_static():
                    return {
                        'static_folder': app.static_folder,
                        'static_url_path': app.static_url_path,
                        'template_folder': app.template_folder,
                        'static_files': os.listdir(static_dir),
                        'template_files': os.listdir(template_dir),
                        'css_files': os.listdir(os.path.join(static_dir, 'css')),
                    }

                @app.route('/debug/config')
                def debug_config():
                    safe_config = {k: str(v) for k, v in app.config.items()
                                   if not k.upper().startswith(('SECRET', 'PASSWORD'))}
                    return safe_config

            # Verify template directory structure
            required_templates = ['base.html', 'home.html', 'login.html', 'register.html']
            missing_templates = [t for t in required_templates
                                 if not os.path.exists(os.path.join(template_dir, t))]

            if missing_templates:
                print(f"Warning: Missing templates: {missing_templates}")

        except Exception as e:
            print(f"Error during app initialization: {e}")
            raise

    return app


@login_manager.user_loader
def load_user(user_id):
    from .models import User
    try:
        return User.query.get(int(user_id))
    except Exception as e:
        print(f"Error loading user: {e}")
        return None


# Import models to ensure they're registered with SQLAlchemy
from . import models