import os
from dotenv import load_dotenv

# Load environment variables from .env file
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))


class Config:
    """Base configuration class."""
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
                              'sqlite:///' + os.path.join(basedir, 'database', 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Upload and Security configuration
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or 'uploads'
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB
    ALLOWED_EXTENSIONS = set(os.environ.get('ALLOWED_EXTENSIONS', 'pdf,docx,xlsx,xls,csv').split(','))
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'true').lower() == 'true'
    SESSION_COOKIE_HTTPONLY = True
    PERMANENT_SESSION_LIFETIME = int(os.environ.get('SESSION_LIFETIME', 3600))

    # Hugging Face and Model configuration
    HF_TOKEN = os.environ.get('HF_TOKEN')
    MODEL_PROVIDER = os.environ.get('MODEL_PROVIDER', 'hf_space')
    RESOURCES_LIMITED = os.environ.get('RESOURCES_LIMITED', 'true').lower() == 'true'
    MODEL_DEPLOYMENT = os.environ.get('MODEL_DEPLOYMENT', 'production')

    # RAG configuration
    RAG_CHUNK_SIZE = int(os.environ.get('RAG_CHUNK_SIZE', 500))
    RAG_CHUNK_OVERLAP = int(os.environ.get('RAG_CHUNK_OVERLAP', 50))
    RAG_TOP_K = int(os.environ.get('RAG_TOP_K', 3))
    # Updated to use an open-access model by default
    RAG_MODEL = os.environ.get('RAG_MODEL', 'facebook/opt-350m')
    RAG_EMBEDDINGS_MODEL = os.environ.get('RAG_EMBEDDINGS_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

    # Additional model options in case the default doesn't work
    ALTERNATIVE_MODELS = [
        'facebook/opt-350m',
        'google/flan-t5-base',
        'HuggingFaceH4/zephyr-7b-beta',
        'facebook/opt-125m'
    ]

    # Processing configuration
    MAX_LENGTH = int(os.environ.get('MAX_LENGTH', 512))
    TEMPERATURE = float(os.environ.get('TEMPERATURE', 0.7))
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 8))
    CONCURRENT_REQUESTS = int(os.environ.get('CONCURRENT_REQUESTS', 4))

    # Cache configuration
    CACHE_TYPE = os.environ.get('CACHE_TYPE', 'simple')
    CACHE_DEFAULT_TIMEOUT = int(os.environ.get('CACHE_DEFAULT_TIMEOUT', 300))

    # Rate limiting
    RATELIMIT_ENABLED = os.environ.get('RATELIMIT_ENABLED', 'true').lower() == 'true'
    RATELIMIT_DEFAULT = os.environ.get('RATELIMIT_DEFAULT', '100 per day')

    # Logging configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    ENABLE_FILE_LOGGING = os.environ.get('ENABLE_FILE_LOGGING', 'false').lower() == 'true'
    LOG_FILE = os.environ.get('LOG_FILE', 'app.log')

    @staticmethod
    def init_app(app):
        """Initialize application."""
        # Create required directories
        required_dirs = [
            os.path.join(basedir, 'database'),
            os.path.join(basedir, Config.UPLOAD_FOLDER),
            os.path.join(basedir, 'logs'),
            os.path.join(basedir, 'models'),
            os.path.join(basedir, 'instance')  # Added instance directory
        ]
        for directory in required_dirs:
            os.makedirs(directory, exist_ok=True)

        # Configure logging
        if Config.ENABLE_FILE_LOGGING:
            import logging
            from logging.handlers import RotatingFileHandler

            log_file = os.path.join(basedir, 'logs', Config.LOG_FILE)
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10485760,  # 10MB
                backupCount=10
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
            ))
            file_handler.setLevel(getattr(logging, Config.LOG_LEVEL))
            app.logger.addHandler(file_handler)

            app.logger.setLevel(getattr(logging, Config.LOG_LEVEL))
            app.logger.info('Application startup')

        # Initialize the database directory if using SQLite
        if 'sqlite' in Config.SQLALCHEMY_DATABASE_URI:
            db_path = Config.SQLALCHEMY_DATABASE_URI.replace('sqlite:///', '')
            os.makedirs(os.path.dirname(db_path), exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False
    TEMPLATES_AUTO_RELOAD = True
    EXPLAIN_TEMPLATE_LOADING = True


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False

    # Override these in production
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    SECRET_KEY = os.environ.get('SECRET_KEY')

    # Production-specific settings
    SESSION_COOKIE_SECURE = True
    REMEMBER_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_HTTPONLY = True

    # Stricter rate limiting for production
    RATELIMIT_DEFAULT = '50 per day'


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False
    PRESERVE_CONTEXT_ON_EXCEPTION = False


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
