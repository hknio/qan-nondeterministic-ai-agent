from typing import List, Dict, Any, Optional
from litellm import completion
from nondeterministic_agent.utils.log_config import configure_logger

# Get the logger instance
logger = configure_logger(__name__)


class LLMService:
    def __init__(
        self,
        default_model: str = "deepseek/deepseek-chat",
    ):
        self.default_model = default_model

    def run_single_completion(
        self,
        prompt: str,
        response_format: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """
        Run single completion request and log usage statistics

        Args:
            prompt: Prompt string to send to the model
            response_format: Optional response format configuration
            model: Optional model override, uses default_model if not specified

        Returns:
            Generated content from the model

        Raises:
            LiteLLMError: If the completion request fails
        """
        used_model = model or self.default_model
        logger.info(f"Processing request to language model: {used_model}")

        max_tokens = max_tokens or 8192

        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]

        try:
            if response_format:
                response = completion(
                    model=used_model,
                    messages=messages,
                    response_format=response_format,
                    max_tokens=max_tokens
                )
            else:
                response = completion(
                    model=used_model,
                    messages=messages,
                    max_tokens=max_tokens
                )

            # Log token usage only if not streaming (as streaming responses may not have usage attribute)
            logger.info(
                f"Token usage - Prompt: {response.usage.prompt_tokens}, "
                f"Completion: {response.usage.completion_tokens}, "
                f"Total: {response.usage.total_tokens}"
            )

            # For non-streaming, return the content
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error during LLM completion: {str(e)}")
            raise

    def run_completion(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """
        Run completion request and log usage statistics

        Args:
            messages: List of message dictionaries to send to the model
            response_format: Optional response format configuration
            model: Optional model override, uses default_model if not specified

        Returns:
            Generated content from the model

        Raises:
            LiteLLMError: If the completion request fails
        """
        used_model = model or self.default_model
        logger.info(f"Processing request to language model: {used_model}")

        max_tokens = max_tokens or 8192

        try:
            if response_format:
                response = completion(
                    model=used_model,
                    messages=messages,
                    response_format=response_format,
                    max_tokens=max_tokens
                )
            else:
                response = completion(
                    model=used_model,
                    messages=messages,
                    max_tokens=max_tokens
                )

            # Log token usage with colorful output
            logger.info(
                f"Token usage - Prompt: {response.usage.prompt_tokens}, "
                f"Completion: {response.usage.completion_tokens}, "
                f"Total: {response.usage.total_tokens}"
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error during LLM completion: {str(e)}")
            raise
