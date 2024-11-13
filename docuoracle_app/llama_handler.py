import requests
import pandas as pd
import json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Union
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LlamaConfig:
    """Configuration for Llama API"""
    API_URL: str = "http://localhost:11434/api"  # Updated for Ollama API
    MAX_TOKENS: int = 2000
    TEMPERATURE: float = 0.7
    MODEL_NAME: str = "llama2"  # Default model name
    RETRY_ATTEMPTS: int = 3
    RETRY_DELAY: int = 2  # seconds


class LlamaHandler:
    def __init__(self, config: Optional[LlamaConfig] = None):
        """Initialize LlamaHandler with optional custom configuration"""
        self.config = config or LlamaConfig()
        self._initialized = False
        self._health_check_endpoint = f"{self.config.API_URL}/health"
        self._generate_endpoint = f"{self.config.API_URL}/generate"

    def initialize_llama(self) -> Tuple[bool, str]:
        """
        Initialize connection to Llama server with retry logic
        Returns:
            Tuple[bool, str]: Success status and message
        """
        logger.info("Attempting to initialize Llama connection...")

        for attempt in range(self.config.RETRY_ATTEMPTS):
            try:
                response = requests.post(
                    self._generate_endpoint,
                    json={
                        "model": self.config.MODEL_NAME,
                        "prompt": "Test connection",
                        "stream": False
                    },
                    timeout=10
                )

                if response.status_code == 200:
                    self._initialized = True
                    logger.info("Llama initialization successful!")
                    return True, "Llama model initialized successfully!"

                logger.warning(f"Attempt {attempt + 1} failed with status code: {response.status_code}")
                time.sleep(self.config.RETRY_DELAY)

            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error on attempt {attempt + 1}: {str(e)}")
                if attempt == self.config.RETRY_ATTEMPTS - 1:
                    return False, "Could not connect to Llama server. Please ensure it's running."
                time.sleep(self.config.RETRY_DELAY)

            except Exception as e:
                logger.error(f"Unexpected error during initialization: {str(e)}")
                return False, f"Error initializing Llama: {str(e)}"

        return False, "Could not initialize Llama after multiple attempts"

    def process_document_with_llama(
            self,
            document_text: str,
            query: str,
            is_data_analysis: bool = False
    ) -> Dict[str, Union[bool, str, dict]]:
        """
        Process document or data using Llama API
        Args:
            document_text: Text content to analyze
            query: User's question
            is_data_analysis: Whether this is a data analysis task
        Returns:
            Dict containing success status, answer, and any error messages
        """
        if not self._initialized:
            return {
                'success': False,
                'error': 'Llama is not initialized. Please initialize first.',
                'answer': None
            }

        try:
            # Construct appropriate prompt based on task type
            if is_data_analysis:
                prompt = self._construct_data_analysis_prompt(document_text, query)
            else:
                prompt = self._construct_document_qa_prompt(document_text, query)

            # Make API request
            response = requests.post(
                self._generate_endpoint,
                json={
                    "model": self.config.MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "parameters": {
                        "max_tokens": self.config.MAX_TOKENS,
                        "temperature": self.config.TEMPERATURE,
                    }
                }
            )

            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()

                return {
                    'success': True,
                    'answer': answer,
                    'error': None,
                    'type': 'data_analysis' if is_data_analysis else 'document_qa'
                }
            else:
                error_msg = f"API error: {response.status_code}"
                logger.error(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'answer': None
                }

        except Exception as e:
            error_msg = f"Error processing document: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'answer': None
            }

    def analyze_excel_data(self, df: pd.DataFrame, query: str) -> Dict[str, Union[bool, str, dict]]:
        """
        Analyze Excel data using Llama
        Args:
            df: Pandas DataFrame containing the data
            query: User's analysis question
        Returns:
            Dict containing analysis results or error information
        """
        try:
            # Generate comprehensive data summary
            data_summary = self._generate_data_summary(df)
            return self.process_document_with_llama(data_summary, query, is_data_analysis=True)

        except Exception as e:
            error_msg = f"Error analyzing Excel data: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'answer': None
            }

    def get_status(self) -> Dict[str, Union[bool, str]]:
        """
        Get current Llama server status
        Returns:
            Dict containing initialization status and connection information
        """
        try:
            # Test connection with a simple generation request
            response = requests.post(
                self._generate_endpoint,
                json={
                    "model": self.config.MODEL_NAME,
                    "prompt": "Test connection",
                    "stream": False
                },
                timeout=5
            )

            if response.status_code == 200:
                return {
                    'initialized': True,
                    'status': 'connected',
                    'api_url': self.config.API_URL
                }

            return {
                'initialized': False,
                'status': 'error',
                'api_url': self.config.API_URL
            }

        except Exception as e:
            logger.error(f"Error checking Llama status: {str(e)}")
            return {
                'initialized': False,
                'status': 'disconnected',
                'api_url': self.config.API_URL
            }

    def _construct_data_analysis_prompt(self, data_text: str, query: str) -> str:
        """Construct prompt for data analysis tasks"""
        return f"""You are a data analyst. Analyze the following data and answer the question.
Please provide clear insights, patterns, and relevant statistical information.

Data:
{data_text}

Question: {query}

Please structure your analysis as follows:
1. Key Findings:
2. Statistical Summary:
3. Recommendations:
4. Additional Insights:

Analysis:"""

    def _construct_document_qa_prompt(self, document_text: str, query: str) -> str:
        """Construct prompt for document Q&A tasks"""
        return f"""Please analyze this document and answer the question clearly and concisely.

Document:
{document_text}

Question: {query}

Provide a detailed answer with clear reasoning and evidence from the document.

Answer:"""

    def _generate_data_summary(self, df: pd.DataFrame) -> str:
        """Generate comprehensive data summary for analysis"""
        summary_parts = []

        # Basic information
        summary_parts.append(f"Data Summary:")
        summary_parts.append(f"- Total Rows: {len(df)}")
        summary_parts.append(f"- Total Columns: {len(df.columns)}")
        summary_parts.append(f"- Columns: {', '.join(df.columns.tolist())}")

        # Numerical summary
        if not df.empty:
            summary_parts.append("\nNumerical Summary:")
            summary_parts.append(df.describe().to_string())

            # First few rows
            summary_parts.append("\nFirst few rows:")
            summary_parts.append(df.head().to_string())

            # Column types
            summary_parts.append("\nColumn Types:")
            summary_parts.append(df.dtypes.to_string())

            # Additional analysis availability
            if df.select_dtypes(include=['number']).columns.any():
                summary_parts.append("\nNumerical Analysis Available")
            if df.select_dtypes(include=['datetime']).columns.any():
                summary_parts.append("\nTemporal Analysis Available")
            if df.select_dtypes(include=['object']).columns.any():
                summary_parts.append("\nCategorical Analysis Available")

        return "\n".join(summary_parts)


# Create singleton instance
llama_handler = LlamaHandler()


# Convenience functions for external use
def initialize_llama() -> Tuple[bool, str]:
    """Initialize Llama server connection"""
    return llama_handler.initialize_llama()


def process_document_with_llama(
        document_text: str,
        query: str,
        is_data_analysis: bool = False
) -> Dict[str, Union[bool, str, dict]]:
    """Process document using Llama"""
    return llama_handler.process_document_with_llama(document_text, query, is_data_analysis)


def analyze_excel_data(df: pd.DataFrame, query: str) -> Dict[str, Union[bool, str, dict]]:
    """Analyze Excel data using Llama"""
    return llama_handler.analyze_excel_data(df, query)


def get_llama_status() -> Dict[str, Union[bool, str]]:
    """Get Llama server status"""
    return llama_handler.get_status()