import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app

if __name__ == "__main__":
    app.run(debug=True)
