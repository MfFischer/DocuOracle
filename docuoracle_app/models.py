from . import db
from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from enum import Enum
from dataclasses import dataclass
import torch


class ModelProvider(Enum):
    """Enum for model providers"""
    LOCAL = "local"
    HUGGINGFACE_SPACE = "hf_space"
    API = "api"


@dataclass
class LlamaConfig:
    """Configuration for local Llama model"""
    MODEL_PATH: str = "models/llama"
    MODEL_TYPE: str = "llama"
    GPU_LAYERS: int = 0
    CONTEXT_LENGTH: int = 2048
    THREADS: int = 4


@dataclass
class RAGConfig:
    """Configuration for RAG components"""
    llm_model: str = "facebook/opt-350m"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    provider: ModelProvider = ModelProvider.HUGGINGFACE_SPACE
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8
    concurrent_requests: int = 4
    temperature: float = 0.7
    max_length: int = 512
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 3


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Define the relationship only once here
    documents = db.relationship(
        'Document',
        backref=db.backref('user', lazy=True),
        lazy=True,
        cascade='all, delete-orphan'
    )

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120), nullable=False)
    filepath = db.Column(db.String(200), nullable=False)
    file_type = db.Column(db.String(10))
    processed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # Added this line
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f'<Document {self.filename}>'

    @property
    def upload_date(self):
        return self.uploaded_at.strftime('%Y-%m-%d %H:%M:%S')
