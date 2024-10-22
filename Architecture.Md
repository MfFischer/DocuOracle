docuoracle/
│
├── app/
│   ├── __init__.py           # Initializes the Flask app
│   ├── routes.py             # Handles routes for home, login, register, and document upload
│   ├── api.py                # API endpoints for handling document and query requests
│   ├── models.py             # SQLAlchemy models for user and document storage
│   ├── utils.py              # Document parsing utilities (PDF, DOCX, Excel)
│   ├── llama_handler.py      # LLaMA (Ollama) model handling logic
│   ├── graph_handler.py      # Handles data analysis and graph generation
│   ├── templates/
│   │   ├── base.html         # Base layout with header, footer, Bootstrap styling
│   │   ├── home.html         # Home page with welcome message
│   │   ├── login.html        # Login form
│   │   ├── register.html     # Registration form
│   │   ├── upload.html       # Document upload and query input page
│   ├── static/
│   │   ├── styles.css        # Custom styles, if any
│   │   └── swagger.json      # Swagger JSON documentation for API
│
├── database/
│   └── app.db                # SQLite database for users and documents
│
├── llama_model/              # Placeholder directory for LLaMA (Ollama) model if needed
│   └── model_weights/        # Model weights for LLaMA (optional)
│
├── scripts/
│   └── install.sh            # Installation script for dependencies
│
├── tests/
│   ├── test_routes.py        # Unit tests for API and routes functionality
│   └── test_graphs.py        # Unit tests for graph generation
│
├── requirements.txt          # Python dependencies (Flask, SQLAlchemy, etc.)
├── config.py                 # Configuration settings (SQLAlchemy DB URI, secret key, etc.)
├── run.py                    # Flask app entry point
└── README.md                 # Project documentation
