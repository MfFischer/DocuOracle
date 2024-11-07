import os
import subprocess
import requests
import json
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LlamaConfig:
    MODEL: str = 'llama2'
    MAX_TOKENS: int = 2000
    TEMPERATURE: float = 0.7


class LlamaHandler:
    def __init__(self, config: Optional[LlamaConfig] = None):
        self.config = config or LlamaConfig()
        self._initialized = False

    def initialize_llama(self) -> Tuple[bool, str]:
        """Initialize the Llama model."""
        try:
            # Check if Ollama service is running and model is loaded
            response = requests.get('http://localhost:11434/api/tags')

            if response.status_code == 200:
                models = response.json().get('models', [])
                if any(model['name'] == self.config.MODEL for model in models):
                    self._initialized = True
                    return True, "Llama model is ready!"
                else:
                    # Try to pull the model
                    pull_response = requests.post(
                        'http://localhost:11434/api/pull',
                        json={'name': self.config.MODEL}
                    )
                    if pull_response.status_code == 200:
                        self._initialized = True
                        return True, "Llama model has been initialized successfully!"
                    else:
                        return False, "Failed to pull Llama model. Please make sure Ollama is properly installed."
            else:
                return False, "Ollama service is not running. Please start Ollama first."

        except requests.exceptions.ConnectionError:
            return False, "Could not connect to Ollama. Please make sure Ollama is running."
        except Exception as e:
            return False, f"Error initializing Llama: {str(e)}"

    def process_document_with_llama(self, document_text: str, query: str) -> str:
        """Process document with LLaMA model."""
        if not self._initialized:
            return "Please initialize Llama first by clicking the 'Initialize Llama' button."

        try:
            # Construct the prompt
            prompt = f"""Based on the following document, please answer this question: {query}

Document content:
{document_text}

Question: {query}
Answer: """

            # Make request to Ollama API
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': self.config.MODEL,
                    'prompt': prompt,
                    'temperature': self.config.TEMPERATURE,
                    'max_tokens': self.config.MAX_TOKENS
                }
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                raise Exception(f"Ollama API error: {response.status_code}")

        except Exception as e:
            print(f"Error processing document: {e}")
            return f"Sorry, I couldn't process your request: {str(e)}"


# Create singleton instance
llama_handler = LlamaHandler()


def initialize_llama() -> Tuple[bool, str]:
    return llama_handler.initialize_llama()


def process_document_with_llama(document_text: str, query: str) -> str:
    return llama_handler.process_document_with_llama(document_text, query)