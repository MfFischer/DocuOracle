import os
from datetime import timedelta

class Config:
    # Base directory for the application
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(BASE_DIR, 'instance', 'docuoracle_app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Security configurations
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your_secret_key'

    # Session configuration
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    SESSION_TYPE = 'filesystem'

    # Upload configuration
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'xlsx', 'xls', 'csv'}

    # Ollama configurations
    OLLAMA_PATH = r"C:\Users\k\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Ollama.lnk"
    OLLAMA_MODEL = 'llama2'
    OLLAMA_MAX_TOKENS = 2000
    OLLAMA_TEMPERATURE = 0.7

    @staticmethod
    def init_app(app):
        # Create necessary directories
        os.makedirs(os.path.join(Config.BASE_DIR, 'uploads'), exist_ok=True)
        os.makedirs(os.path.join(Config.BASE_DIR, 'instance'), exist_ok=True)

        # Ensure the directory for the database exists
        db_dir = os.path.dirname(app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', ''))
        os.makedirs(db_dir, exist_ok=True)