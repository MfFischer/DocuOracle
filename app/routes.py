# app/routes.py

from flask import render_template, request, redirect, url_for, flash
from app import app, db
from app.models import User, Document
from app.utils import parse_pdf, parse_word, parse_excel
from app.llama_handler import generate_response
from app.graph_handler import parse_excel, generate_matplotlib_graph, generate_plotly_graph
import os

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Implement login logic here
    pass

@app.route('/register', methods=['GET', 'POST'])
def register():
    # Implement registration logic here
    pass

@app.route('/logout')
def logout():
    # Implement logout logic here
    pass

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        question = request.form.get('question')
        chart_type = request.form.get('chart_type')  # e.g., 'line', 'bar', 'scatter'
        x_col = request.form.get('x_col')
        y_col = request.form.get('y_col')

        if uploaded_file:
            filename = uploaded_file.filename
            filepath = os.path.join('documents', filename)
            uploaded_file.save(filepath)

            # Save to database
            new_doc = Document(filename=filename, filepath=filepath, user_id=1)  # Replace with actual user ID
            db.session.add(new_doc)
            db.session.commit()

            # Determine file type and parse accordingly
            if filename.endswith('.pdf'):
                document_text = parse_pdf(filepath)
            elif filename.endswith('.docx'):
                document_text = parse_word(filepath)
            elif filename.endswith(('.xls', '.xlsx', '.csv')):
                df = parse_excel(filepath)
                # Optionally, generate a graph if specific parameters are provided
                if chart_type and x_col and y_col:
                    # Choose either Matplotlib or Plotly based on preference
                    graph = generate_matplotlib_graph(df, chart_type, x_col, y_col)
                    # Or use Plotly
                    # graph = generate_plotly_graph(df, chart_type, x_col, y_col)
                else:
                    graph = None
                document_text = df.to_string()

            else:
                flash('Unsupported file type.', 'danger')
                return redirect(request.url)

            # Generate response using LLaMA (Ollama)
            if question:
                answer = generate_response(question, document_text)
            else:
                answer = "No question provided."

            return render_template('upload.html', answer=answer, graph=graph)

    return render_template('upload.html')
