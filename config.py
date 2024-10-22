# config.py

import os


class Config:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///database/app.db'
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your_secret_key'
    OLLAMA_PATH = r"C:\Users\k\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Ollama.lnk"
