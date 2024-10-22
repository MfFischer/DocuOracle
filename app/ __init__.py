from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from app.swagger_config import swaggerui_blueprint

# Initialize the Flask app
app = Flask(__name__)

# Configurations (can be moved to config.py if needed)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database/app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Import models (ensure these are imported after db initialization)
from app.models import User, Document

# Register Swagger UI blueprint
app.register_blueprint(swaggerui_blueprint, url_prefix='/swagger')

# Import and register routes
from app.routes import app as routes_blueprint
app.register_blueprint(routes_blueprint)

# Import and register API routes
from app.api import api_blueprint
app.register_blueprint(api_blueprint)

# Create all database tables (for initial run, optional)
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)
