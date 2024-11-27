import pandas as pd
from ctransformers import AutoModelForCausalLM
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Union, List
from enum import Enum
import logging
from huggingface_hub import login
import os
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Enum for model providers"""
    LOCAL = "local"
    HUGGINGFACE_SPACE = "hf_space"
    API = "api"


@dataclass
class RAGConfig:
    """Configuration for RAG components"""
    llm_model: str = "facebook/opt-350m"  # Using OPT model
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


@dataclass
class LlamaConfig:
    """Configuration for local Llama model"""
    MODEL_PATH: str = "models/llama"
    MODEL_TYPE: str = "llama"
    GPU_LAYERS: int = 0
    CONTEXT_LENGTH: int = 2048


class LlamaHandler:
    def __init__(self, config: Optional[LlamaConfig] = None, rag_config: Optional[RAGConfig] = None):
        self.config = config or LlamaConfig()
        self.rag_config = rag_config or RAGConfig()

        # Better device handling
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.rag_config:
            self.rag_config.device = self.device

        self._initialized = False
        self._model = None
        self._embeddings_model = None
        self._tokenizer = None
        self._vector_store = None

    def initialize_llama(self) -> Tuple[bool, str]:
        """Initialize model using Hugging Face"""
        logger.info("Attempting to initialize Hugging Face model...")

        try:
            # Get HF token from environment
            hf_token = os.getenv('HF_TOKEN')
            if not hf_token:
                error_msg = "HF_TOKEN not found in environment variables"
                logger.error(error_msg)
                return False, error_msg

            # Login to Hugging Face
            login(token=hf_token)

            # Initialize tokenizer and model
            model_name = "facebook/opt-350m"  # Using OPT model instead of Mistral
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._model = AutoModel.from_pretrained(
                    model_name,
                    token=hf_token,
                    device_map=self.device,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )

                self._initialized = True
                logger.info("Hugging Face OPT model initialization successful!")
                return True, "Model initialized successfully!"

            except Exception as model_error:
                error_msg = f"Error loading model: {str(model_error)}"
                logger.error(error_msg)
                return False, error_msg

        except Exception as e:
            error_msg = f"Error during initialization: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def select_rag_model(self, requirements: Dict) -> RAGConfig:
        """Select appropriate RAG models based on requirements"""
        if requirements.get('deployment') == 'production':
            if requirements.get('resources') == 'limited':
                return RAGConfig(
                    llm_model='facebook/opt-350m',  # Open-access model for limited resources
                    embedding_model='sentence-transformers/all-MiniLM-L6-v2',
                    provider=ModelProvider.HUGGINGFACE_SPACE,
                    batch_size=8,
                    concurrent_requests=4
                )
            else:
                return RAGConfig(
                    llm_model='facebook/opt-1.3b',  # Larger open-access model
                    embedding_model='sentence-transformers/all-mpnet-base-v2',
                    provider=ModelProvider.HUGGINGFACE_SPACE,
                    batch_size=16,
                    concurrent_requests=8
                )
        else:
            return RAGConfig(
                llm_model='facebook/opt-350m',  # Default to smaller model for development
                embedding_model='sentence-transformers/all-MiniLM-L6-v2',
                provider=ModelProvider.HUGGINGFACE_SPACE,
                batch_size=4,
                concurrent_requests=2
            )

    def initialize_rag(self, requirements: Dict) -> Tuple[bool, str]:
        """Initialize RAG components"""
        try:
            # Login to Hugging Face if token is available
            hf_token = os.getenv('HF_TOKEN')
            if hf_token:
                login(token=hf_token)
                logger.info("Successfully authenticated with Hugging Face")

            # Select and configure RAG models
            self.rag_config = self.select_rag_model(requirements)

            # Initialize embedding model
            self._embeddings_model = SentenceTransformer(
                self.rag_config.embedding_model,
                device=self.rag_config.device
            )
            logger.info(f"Initialized embeddings model: {self.rag_config.embedding_model}")

            # Initialize tokenizer and model based on provider
            if self.rag_config.provider == ModelProvider.HUGGINGFACE_SPACE:
                self._tokenizer = AutoTokenizer.from_pretrained(self.rag_config.llm_model)
                self._model = AutoModel.from_pretrained(
                    self.rag_config.llm_model,
                    device_map=self.rag_config.device,
                    torch_dtype=torch.float16 if self.rag_config.device == "cuda" else torch.float32
                )
                logger.info(f"Initialized LLM model: {self.rag_config.llm_model}")

            self._initialized = True
            return True, "RAG components initialized successfully!"

        except Exception as e:
            error_msg = f"Error initializing RAG components: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def process_document_with_llama(
            self,
            document_text: str,
            query: str,
            is_data_analysis: bool = False
    ) -> Dict[str, Union[bool, str, dict]]:
        """Process document using traditional Llama approach"""
        if not self._initialized or not self._model:
            return {
                'success': False,
                'error': 'Llama is not initialized. Please initialize first.',
                'answer': None
            }

        try:
            # Construct prompt based on task type
            if is_data_analysis:
                prompt = self._construct_data_analysis_prompt(document_text, query)
            else:
                prompt = self._construct_document_qa_prompt(document_text, query)

            # Generate response
            response = self._model(
                prompt,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                temperature=self.config.TEMPERATURE,
                stop=["</s>", "[/INST]"]
            )

            # Clean up the response
            cleaned_response = self._clean_response(response)

            return {
                'success': True,
                'answer': cleaned_response,
                'error': None,
                'type': 'data_analysis' if is_data_analysis else 'document_qa'
            }

        except Exception as e:
            error_msg = f"Error processing document: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'answer': None
            }

    def process_document_with_rag(
            self,
            document_text: str,
            query: str,
            is_data_analysis: bool = False
    ) -> Dict[str, Union[bool, str, dict]]:
        """Process document using RAG approach"""
        if not self._initialized:
            return {
                'success': False,
                'error': 'RAG components not initialized. Please initialize first.',
                'answer': None
            }

        try:
            # Split document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.rag_config.chunk_size,
                chunk_overlap=self.rag_config.chunk_overlap
            )
            chunks = text_splitter.split_text(document_text)

            # Create or update vector store
            if not self._vector_store:
                embeddings = self._embeddings_model.encode(chunks)
                self._vector_store = FAISS.from_embeddings(
                    embeddings=embeddings,
                    texts=chunks,
                    embedding=self._embeddings_model
                )

            # Get relevant chunks for the query
            query_embedding = self._embeddings_model.encode([query])[0]
            relevant_chunks = self._vector_store.similarity_search_by_vector(
                query_embedding,
                k=3
            )

            # Construct prompt with relevant context
            context = "\n".join([chunk.page_content for chunk in relevant_chunks])
            if is_data_analysis:
                prompt = self._construct_data_analysis_prompt(context, query)
            else:
                prompt = self._construct_document_qa_prompt(context, query)

            # Generate response
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True,
                                     max_length=self.rag_config.max_length)
            outputs = self._model.generate(
                inputs["input_ids"].to(self.rag_config.device),
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                temperature=self.config.TEMPERATURE,
                do_sample=True
            )

            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            cleaned_response = self._clean_response(response)

            return {
                'success': True,
                'answer': cleaned_response,
                'sources': [chunk.page_content for chunk in relevant_chunks],
                'error': None,
                'type': 'rag_analysis'
            }

        except Exception as e:
            error_msg = f"Error processing document with RAG: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'answer': None
            }

    def analyze_excel_data(self, df: pd.DataFrame, query: str) -> Dict[str, Union[bool, str, dict]]:
        """Analyze Excel data with either traditional or RAG approach"""
        try:
            data_summary = self._generate_data_summary(df)
            if self.rag_config and self._embeddings_model:
                return self.process_document_with_rag(data_summary, query, is_data_analysis=True)
            else:
                return self.process_document_with_llama(data_summary, query, is_data_analysis=True)

        except Exception as e:
            error_msg = f"Error analyzing Excel data: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'answer': None
            }

    def _clean_response(self, response: str) -> str:
        """Clean up model response"""
        if not response:
            return ""

        # Remove any remaining instruction tokens
        response = response.replace("<s>", "").replace("</s>", "")
        response = response.replace("[INST]", "").replace("[/INST]", "")

        # Clean up whitespace
        response = response.strip()

        return response

    def _construct_data_analysis_prompt(self, data_text: str, query: str) -> str:
        """Construct prompt for data analysis"""
        return f"""<s>[INST] You are a data analyst. Analyze the following data and answer the question.
Please provide clear insights, patterns, and relevant statistical information.

Data:
{data_text}

Question: {query}

Please structure your analysis as follows:
1. Key Findings:
2. Statistical Summary:
3. Recommendations:
4. Additional Insights:

Analysis: [/INST]"""

    def _construct_document_qa_prompt(self, document_text: str, query: str) -> str:
        """Construct prompt for document Q&A"""
        return f"""<s>[INST] Please analyze this document and answer the question clearly and concisely.

Document:
{document_text}

Question: {query}

Provide a detailed answer with clear reasoning and evidence from the document.

Answer: [/INST]"""

    def _generate_data_summary(self, df: pd.DataFrame) -> str:
        """Generate summary of DataFrame"""
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

    def get_status(self) -> Dict[str, Union[bool, str]]:
        """Get current status of the handler"""
        status = {
            'initialized': self._initialized,
            'status': 'connected' if self._initialized else 'disconnected',
            'model_path': self.config.MODEL_PATH,
            'model_type': self.config.MODEL_TYPE,
            'gpu_layers': self.config.GPU_LAYERS
        }

        if self.rag_config:
            status.update({
                'rag_enabled': True,
                'embedding_model': self.rag_config.embedding_model,
                'llm_model': self.rag_config.llm_model,
                'provider': self.rag_config.provider.value
            })

        return status


# Create singleton instance
llama_handler = LlamaHandler()


# Convenience functions for external use
def initialize_llama() -> Tuple[bool, str]:
    return llama_handler.initialize_llama()


def initialize_rag(requirements: Dict) -> Tuple[bool, str]:
    return llama_handler.initialize_rag(requirements)


def process_document_with_llama(
        document_text: str,
        query: str,
        is_data_analysis: bool = False
) -> Dict[str, Union[bool, str, dict]]:
    return llama_handler.process_document_with_llama(document_text, query, is_data_analysis)


def process_document_with_rag(
        document_text: str,
        query: str,
        is_data_analysis: bool = False
) -> Dict[str, Union[bool, str, dict]]:
    return llama_handler.process_document_with_rag(document_text, query, is_data_analysis)


def analyze_excel_data(df: pd.DataFrame, query: str) -> Dict[str, Union[bool, str, dict]]:
    return llama_handler.analyze_excel_data(df, query)


def get_llama_status() -> Dict[str, Union[bool, str]]:
    return llama_handler.get_status()


def select_rag_model(requirements: Dict) -> Dict[str, Union[str, ModelProvider, int]]:
    return llama_handler.select_rag_model(requirements)


# Optional: Add validation functions
def validate_document(document_text: str) -> bool:
    """Validate document text before processing"""
    try:
        if not document_text or not isinstance(document_text, str):
            logger.error("Invalid document text provided")
            return False
        if len(document_text.strip()) == 0:
            logger.error("Empty document text provided")
            return False
        return True
    except Exception as e:
        logger.error(f"Error validating document: {str(e)}")
        return False


def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate DataFrame before processing"""
    try:
        if df is None or not isinstance(df, pd.DataFrame):
            logger.error("Invalid DataFrame provided")
            return False
        if df.empty:
            logger.error("Empty DataFrame provided")
            return False
        return True
    except Exception as e:
        logger.error(f"Error validating DataFrame: {str(e)}")
        return False


# Error handling wrapper (optional)
def safe_process(func):
    """Decorator for safe processing with error handling"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            return {
                'success': False,
                'error': f"Processing error in {func.__name__}: {str(e)}",
                'answer': None
            }

    return wrapper


# Example usage of the safe_process decorator:
@safe_process
def process_document_safely(document_text: str, query: str, use_rag: bool = False) -> Dict[str, Union[bool, str, dict]]:
    """Safely process document with error handling"""
    if not validate_document(document_text):
        return {
            'success': False,
            'error': 'Invalid document text',
            'answer': None
        }

    if use_rag:
        return llama_handler.process_document_with_rag(document_text, query)
    else:
        return llama_handler.process_document_with_llama(document_text, query)


# Configuration helper functions
def update_rag_config(new_config: Dict) -> Tuple[bool, str]:
    """Update RAG configuration"""
    try:
        requirements = {
            'deployment': new_config.get('deployment', 'production'),
            'resources': new_config.get('resources', 'limited')
        }
        new_rag_config = llama_handler.select_rag_model(requirements)
        success, message = llama_handler.initialize_rag(requirements)
        return success, message
    except Exception as e:
        return False, f"Error updating RAG config: {str(e)}"


def get_available_models() -> Dict[str, List[str]]:
    """Get list of available models"""
    return {
        'llm_models': [
            'mistralai/Mistral-7B-Instruct-v0.1',
            'mistralai/Mixtral-8x7B-Instruct-v0.1',
            'meta-llama/Llama-2-13b-chat-hf'
        ],
        'embedding_models': [
            'sentence-transformers/all-MiniLM-L6-v2',
            'sentence-transformers/all-mpnet-base-v2'
        ]
    }


# If running as main module
if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Initialize with RAG
    requirements = {
        'deployment': 'production',
        'resources': 'limited'
    }

    success, message = initialize_rag(requirements)
    if success:
        logger.info("RAG initialization successful")

        # Example document processing
        test_doc = "This is a test document for RAG processing."
        test_query = "What is this document about?"

        result = process_document_safely(test_doc, test_query, use_rag=True)
        if result['success']:
            logger.info(f"Processing result: {result['answer']}")
        else:
            logger.error(f"Processing error: {result['error']}")
    else:
        logger.error(f"RAG initialization failed: {message}")