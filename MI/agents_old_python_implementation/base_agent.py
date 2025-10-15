"""
Base Agent Class - MI Knowledge Hub
Foundation class for all specialized agents
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class BaseAgent(ABC):
    """
    Abstract base class for all MI Knowledge Hub agents.

    Provides common functionality for:
    - Claude API integration
    - Logging
    - File I/O
    - Error handling
    - Rate limiting
    """

    def __init__(
        self,
        name: str,
        description: str,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        """
        Initialize base agent.

        Args:
            name: Agent name
            description: Agent description
            model: Claude model to use
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
        """
        self.name = name
        self.description = description
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize Claude client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        self.client = anthropic.Anthropic(api_key=api_key)

        # Setup logging
        self.logger = self._setup_logger()

        # Project paths
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.content_dir = self.project_root / "content"
        self.logs_dir = self.project_root / "logs"
        self.reviews_dir = self.project_root / "reviews"

        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.content_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.reviews_dir.mkdir(exist_ok=True)

        self.logger.info(f"Initialized {self.name}")

    def _setup_logger(self) -> logging.Logger:
        """
        Setup logger for this agent.

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # File handler
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / f"{self.name.lower().replace(' ', '_')}.log"
        )
        file_handler.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger

    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate response using Claude API.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            Generated response text
        """
        try:
            self.logger.debug(f"Generating response with prompt length: {len(prompt)}")

            params = {
                "model": self.model,
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature or self.temperature,
                "messages": [{"role": "user", "content": prompt}],
            }

            if system_prompt:
                params["system"] = system_prompt

            response = self.client.messages.create(**params)

            # Extract text from response
            text_content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text_content += block.text

            self.logger.debug(f"Generated response length: {len(text_content)}")
            return text_content

        except anthropic.APIError as e:
            self.logger.error(f"Claude API error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in generate_response: {e}")
            raise

    def load_json(self, file_path: Path) -> Any:
        """
        Load JSON data from file.

        Args:
            file_path: Path to JSON file

        Returns:
            Parsed JSON data
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.logger.debug(f"Loaded JSON from {file_path}")
            return data
        except FileNotFoundError:
            self.logger.warning(f"File not found: {file_path}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in {file_path}: {e}")
            raise

    def save_json(self, data: Any, file_path: Path, indent: int = 2) -> bool:
        """
        Save data to JSON file.

        Args:
            data: Data to save
            file_path: Path to JSON file
            indent: JSON indentation level

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
            self.logger.info(f"Saved JSON to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving JSON to {file_path}: {e}")
            return False

    def load_text(self, file_path: Path) -> Optional[str]:
        """
        Load text content from file.

        Args:
            file_path: Path to text file

        Returns:
            File content as string
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.logger.debug(f"Loaded text from {file_path}")
            return content
        except FileNotFoundError:
            self.logger.warning(f"File not found: {file_path}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading text from {file_path}: {e}")
            return None

    def save_text(self, content: str, file_path: Path) -> bool:
        """
        Save text content to file.

        Args:
            content: Text content to save
            file_path: Path to text file

        Returns:
            True if successful, False otherwise
        """
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            self.logger.info(f"Saved text to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving text to {file_path}: {e}")
            return False

    def timestamp(self) -> str:
        """
        Get current timestamp in ISO format.

        Returns:
            ISO formatted timestamp string
        """
        return datetime.now().isoformat()

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute agent's primary function.
        Must be implemented by subclasses.

        Args:
            **kwargs: Agent-specific arguments

        Returns:
            Dictionary with execution results
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"
