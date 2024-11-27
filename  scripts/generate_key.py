import secrets
import os
from pathlib import Path


def generate_secret_key():
    """Generate a secure secret key."""
    return secrets.token_hex(32)


def update_env_file():
    """Update .env file with new secret key and create if doesn't exist."""
    # Get project root directory
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'

    # Default environment variables
    default_env = {
        'FLASK_APP': 'run.py',
        'FLASK_ENV': 'development',
        'SECRET_KEY': generate_secret_key(),
        'DATABASE_URL': 'sqlite:///database/app.db',
        'UPLOAD_FOLDER': 'uploads',
        'MODEL_PROVIDER': 'hf_space',
        'RAG_CHUNK_SIZE': '500',
        'RAG_CHUNK_OVERLAP': '50',
        'RAG_TOP_K': '3'
    }

    # Read existing .env file if it exists
    existing_env = {}
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    existing_env[key] = value

    # Merge existing with defaults, keeping existing values except SECRET_KEY
    final_env = default_env.copy()
    for key, value in existing_env.items():
        if key != 'SECRET_KEY':  # Always generate new SECRET_KEY
            final_env[key] = value

    # Write to .env file
    with open(env_path, 'w') as f:
        for key, value in final_env.items():
            f.write(f'{key}={value}\n')

    print(f'Environment file updated at: {env_path}')
    print('Generated new secret key and updated environment variables.')
    print('\nNOTE: Add your Hugging Face token to .env file:')
    print('HF_TOKEN=your_token_here')


def create_example_env():
    """Create .env.example file with dummy values."""
    project_root = Path(__file__).parent.parent
    example_env_path = project_root / '.env.example'

    example_content = '''# DocuOracle Environment Configuration
FLASK_APP=run.py
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///database/app.db
UPLOAD_FOLDER=uploads
HF_TOKEN=your-hugging-face-token-here
MODEL_PROVIDER=hf_space
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=50
RAG_TOP_K=3
'''

    with open(example_env_path, 'w') as f:
        f.write(example_content)

    print(f'Created example environment file at: {example_env_path}')


if __name__ == '__main__':
    update_env_file()
    create_example_env()