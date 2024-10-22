
 # Project Name: DocuOracle #
Description: An interactive, AI-powered document-based assistant that enables users to upload various document formats (PDF, Word, Excel) and ask questions based on the content. It leverages modern UI design principles with Flask as the backend and incorporates LLaMA (Ollama) for AI-based text analysis.

## Key Features :
Document Upload & Querying:

Users can upload documents in multiple formats: PDF, Word (DOCX), and Excel.
The system extracts relevant information and allows users to ask questions about the document content.
AI-Powered Answer Generation:

Utilizes LLaMA (Ollama), a large language model, to analyze and respond to user queries based on the uploaded documents.

## Data Visualization:

Automatically generates data analysis graphs (line, bar, scatter plots) from uploaded Excel files.
Graphs are displayed interactively using Matplotlib or Plotly for visual data exploration.
Modern, User-Friendly UI:

The frontend design is interactive, elegant, and artistic, with smooth animations and a modern aesthetic.
The UI is built using Bootstrap and custom CSS with a unique color scheme (#32012F, #524C42, #E2DFD0, #F97300).

## API & Swagger Documentation:

Provides a RESTful API for programmatic access to document upload and querying.
Integrated with Swagger UI for interactive API documentation and testing.
Authentication:

Supports user login and registration functionality to secure access and manage user-specific document data.

## Database Management:

Stores user and document information in a SQLite database, with future scalability to other databases like PostgreSQL.
Flask-Migrate is used for database version control and migrations.

## Local Desktop Installation:

The app is designed to be installable and run locally on the desktop, making it easy for users to use on their personal systems.

## Technologies Used:
Flask (Python Web Framework)
LLaMA (Ollama) for AI-powered text analysis
Flask-SQLAlchemy (Database ORM)
Flask-Migrate (Database migrations)
Flask-RESTful (API development)
Flask-Swagger-UI (API documentation)
Pandas, Matplotlib, Plotly (Data analysis and visualization)
Bootstrap (Frontend framework)
