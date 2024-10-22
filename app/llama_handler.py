import subprocess
from flask import current_app


def generate_response(question, document_text):
    input_text = f"Document: {document_text}\nQuestion: {question}"

    # Example subprocess call to interact with Ollama
    # Adjust the command based on how Ollama accepts input and returns output
    try:
        result = subprocess.run(
            [current_app.config['OLLAMA_PATH'], input_text],
            capture_output=True,
            text=True,
            shell=True  # Use shell=True if necessary on Windows
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I couldn't process your request."
