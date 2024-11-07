from flask_swagger_ui import get_swaggerui_blueprint

# Swagger configuration
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'

# Configure Swagger UI Blueprint
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "DocuOracle API",
        'dom_id': '#swagger-ui',
        'deepLinking': True,
        'layout': 'BaseLayout',
        'showExtensions': True,
        'showCommonExtensions': True
    }
)

# Define API documentation
SWAGGER_DOCS = {
    "openapi": "3.0.0",
    "info": {
        "title": "DocuOracle API",
        "description": "API documentation for DocuOracle - Document Analysis and Query System",
        "version": "1.0.0",
        "contact": {
            "name": "Your Name",
            "email": "your.email@example.com"
        }
    },
    "servers": [
        {
            "url": "http://localhost:5000",
            "description": "Development server"
        }
    ],
    "components": {
        "securitySchemes": {
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT"
            }
        },
        "schemas": {
            "Error": {
                "type": "object",
                "properties": {
                    "error": {
                        "type": "string"
                    }
                }
            },
            "Document": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "filename": {"type": "string"},
                    "uploaded_at": {"type": "string", "format": "date-time"},
                    "file_type": {"type": "string"},
                    "processed": {"type": "boolean"}
                }
            }
        }
    },
    "paths": {
        "/api/documents": {
            "get": {
                "tags": ["Documents"],
                "summary": "Get all documents",
                "security": [{"bearerAuth": []}],
                "responses": {
                    "200": {
                        "description": "List of documents",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/Document"}
                                }
                            }
                        }
                    }
                }
            },
            "post": {
                "tags": ["Documents"],
                "summary": "Upload a new document",
                "security": [{"bearerAuth": []}],
                "requestBody": {
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "file": {
                                        "type": "string",
                                        "format": "binary"
                                    }
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "201": {
                        "description": "Document uploaded successfully"
                    }
                }
            }
        },
        "/api/query": {
            "post": {
                "tags": ["Query"],
                "summary": "Query a document",
                "security": [{"bearerAuth": []}],
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "document_id": {"type": "integer"},
                                    "question": {"type": "string"}
                                },
                                "required": ["document_id", "question"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Query response"
                    }
                }
            }
        },
        "/api/graph": {
            "post": {
                "tags": ["Graphs"],
                "summary": "Generate a graph from document data",
                "security": [{"bearerAuth": []}],
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "document_id": {"type": "integer"},
                                    "chart_type": {"type": "string"},
                                    "x_col": {"type": "string"},
                                    "y_col": {"type": "string"}
                                },
                                "required": ["document_id", "chart_type", "x_col", "y_col"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Generated graph data"
                    }
                }
            }
        }
    }
}