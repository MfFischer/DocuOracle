from flask_swagger_ui import get_swaggerui_blueprint

# Define the Swagger URL where Swagger UI will be accessible
SWAGGER_URL = '/swagger'  # URL for accessing Swagger UI

# Path to the Swagger JSON file (you can also serve it dynamically)
API_URL = '/static/swagger.json'  # Path to your swagger.json file

# Configure the Swagger UI Blueprint
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,  # Swagger UI endpoint
    API_URL,      # Swagger JSON file endpoint
    config={      # Additional Swagger UI config (optional)
        'app_name': "DocuOracle API"
    }
)
