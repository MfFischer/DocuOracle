docuoracle/
│
├── docuoracle_app/
│   ├── __init__.py           # Initializes the Flask app
│   ├── routes.py             # Handles routes for home, login, register, and document upload
│   ├── api.py                # API endpoints for handling document and query requests
│   ├── models.py             # SQLAlchemy models for user and document storage
│   ├── utils.py              # Document parsing utilities (PDF, DOCX, Excel)
│   ├── llama_handler.py      # LLaMA (Ollama) model handling logic
│   ├── graph_handler.py      # Handles data analysis and graph generation
│   ├── swagger_config.py 
│   ├── templates/
│   │   ├── errors/ 
│   │   │    ├── 404.html
│   │   │    ├── 500.html
│   │   ├── base.html         # Base layout with header, footer, Bootstrap styling
│   │   ├── home.html         # Home page with welcome message
│   │   ├── login.html        # Login form
│   │   ├── register.html     # Registration form
│   │   ├── profile.html
│   │   ├── settings.html
│   │   ├── reset_password.html
│   │   ├── upload.html
│   │   ├── view_document.html
│   │   ├── analyze.html
│   ├── static/
│   │   ├── styles.css        # Custom styles, if any
│   │   └── swagger.json      # Swagger JSON documentation for API
│
├── database/
│   └── app.db                # SQLite database for users and documents
│
├── models/              # Placeholder directory for LLaMA (Ollama) model if needed
│   └── mistral-7b-instruct-v0.1.Q4_K_M.gguf        # Model weights for LLaMA (optional)
│
├── scripts/
│   └── install.sh            # Installation script for dependencies
├── uploads/
│   └── 1/ 
│
├── tests/
│   ├── test_routes.py        # Unit tests for API and routes functionality
│   └── test_graphs.py        # Unit tests for graph generation
│
├── requirements.txt          # Python dependencies (Flask, SQLAlchemy, etc.)
├── config.py                 # Configuration settings (SQLAlchemy DB URI, secret key, etc.)
├── init_db.py       
├── docker_compose.yml
├── run.py                    # Flask app entry point
└── README.md                 # Project documentation
