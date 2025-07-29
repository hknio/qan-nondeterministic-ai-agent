import logging
from typing import Dict, Any, Optional

from qan import (
    QanMultiClient,
    ServerError,
    PrecompileError,
    CompileError,
    ExecuteError,
    ExecuteTimeoutError,
    NonDeterministicError,
    SystemError,
    ExecutionResult,
)

# Assuming settings are correctly configured and importable
from nondeterministic_agent.config.settings import settings

logger = logging.getLogger(__name__)


class QANService:
    """
    Encapsulates interactions with the QAN multi-client service.

    Manages the QanMultiClient lifecycle and translates QAN exceptions
    into standardized dictionary results.
    """

    def __init__(self, service_settings: Optional[Any] = None):
        """
        Initializes the QANService with configuration.

        Args:
            service_settings: An object with QAN configuration attributes
                             (QAN_SERVERS, QAN_TIMEOUT, QAN_MAX_RETRIES, QAN_RETRY_DELAY).
                             Defaults to the global settings instance.
        """
        s = service_settings or settings  # Use provided settings or global default
        self.servers = s.QAN_SERVERS
        self.timeout = s.QAN_TIMEOUT
        self.max_retries = s.QAN_MAX_RETRIES
        self.retry_delay = s.QAN_RETRY_DELAY
        self.client: Optional[QanMultiClient] = None
        logger.info(
            f"QANService configured with servers: {self.servers}, timeout: {self.timeout}"
        )

    async def __aenter__(self):
        """Creates and returns the QanMultiClient instance."""
        if self.client is None:
            self.client = QanMultiClient(
                servers=self.servers,
                timeout=self.timeout,
                max_retries=self.max_retries,
                retry_delay=self.retry_delay,
            )
            logger.debug("QanMultiClient instance created.")
        # No need to explicitly connect, happens on first call or __aenter__
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Closes the QanMultiClient instance."""
        if self.client:
            # QanMultiClient doesn't have an explicit close method in the example usage.
            # Assuming underlying httpx clients are managed by its context manager.
            logger.debug("QanMultiClient context exited.")
            self.client = None
        # Propagate exceptions if any occurred within the context
        return False  # Returning False re-raises the exception if one occurred

    def _get_client(self) -> QanMultiClient:
        """Ensures the client is initialized."""
        if self.client is None:
            # This should ideally not happen if used within async with context.
            logger.error("QANService client accessed outside of async context.")
            raise RuntimeError("QANService must be used within an async with block.")
        return self.client

    async def health(self) -> bool:
        """Checks the health of the majority of configured QAN servers."""
        client = self._get_client()
        try:
            is_healthy = await client.health()
            logger.info(f"QAN health check: Healthy Majority = {is_healthy}")
            return is_healthy
        except Exception as e:
            logger.error(f"QAN health check failed: {e}", exc_info=True)
            return False

    async def precompile(self, language: str, code_bytes: bytes) -> Dict[str, Any]:
        """
        Sends code for precompilation to QAN servers.

        Args:
            language: The programming language of the code.
            code_bytes: The source code as bytes.

        Returns:
            A dictionary with keys: 'success', 'result' (bytes | None),
            'error', 'stderr', 'exit_code', 'non_deterministic', 'stage', 'details'.
        """
        client = self._get_client()
        stage = "precompile"
        result_dict = {
            "success": False,
            "result": None,
            "error": None,
            "stderr": None,
            "exit_code": None,
            "non_deterministic": False,
            "stage": stage,
            "details": None,
        }
        try:
            logger.debug(f"Sending precompile request for {language}...")
            decompiled_bytes = await client.precompile(language, code_bytes)
            result_dict.update(
                {
                    "success": True,
                    "result": decompiled_bytes,
                }
            )
            logger.debug(f"Precompile successful for {language}.")
            return result_dict

        except NonDeterministicError as e:
            logger.warning(f"NonDeterministicError during {stage}: {e}")
            result_dict.update(
                {
                    "error": f"NonDeterministicError: {e}",
                    "non_deterministic": True,
                    "detection_method": "qan_precompile",
                    "details": str(e),
                }
            )
            return result_dict
        except PrecompileError as e:
            logger.warning(f"PrecompileError: {e}")
            result_dict.update(
                {
                    "error": f"PrecompileError: {e}",
                    "stderr": e.stderr,
                    "exit_code": e.exit_code,
                    "details": str(e),
                }
            )
            return result_dict
        except ServerError as e:
            logger.error(f"ServerError during {stage}: {e}")
            result_dict.update(
                {
                    "error": f"ServerError: {e}",
                    "stderr": getattr(e, "details", str(e)),
                    "exit_code": -1,  # Indicate server-side issue
                    "details": str(e),
                }
            )
            return result_dict
        except Exception as e:
            logger.error(f"Unexpected error during {stage}: {e}", exc_info=True)
            result_dict.update(
                {
                    "error": f"Unexpected Error: {e}",
                    "exit_code": -1,
                    "details": str(e),
                }
            )
            return result_dict

    async def compile(self, language: str, decompiled_bytes: bytes) -> Dict[str, Any]:
        """
        Sends precompiled code for compilation to QAN servers.

        Args:
            language: The programming language.
            decompiled_bytes: The result from a successful precompile step.

        Returns:
            A dictionary similar to precompile's result.
        """
        client = self._get_client()
        stage = "compile"
        result_dict = {
            "success": False,
            "result": None,
            "error": None,
            "stderr": None,
            "exit_code": None,
            "non_deterministic": False,
            "stage": stage,
            "details": None,
        }
        try:
            logger.debug(f"Sending compile request for {language}...")
            binary_bytes = await client.compile(language, decompiled_bytes)
            result_dict.update(
                {
                    "success": True,
                    "result": binary_bytes,
                }
            )
            logger.debug(f"Compile successful for {language}.")
            return result_dict

        except NonDeterministicError as e:
            logger.warning(f"NonDeterministicError during {stage}: {e}")
            result_dict.update(
                {
                    "error": f"NonDeterministicError: {e}",
                    "non_deterministic": True,
                    "detection_method": "qan_compile",
                    "details": str(e),
                }
            )
            return result_dict
        except CompileError as e:
            logger.warning(f"CompileError: {e}")
            result_dict.update(
                {
                    "error": f"CompileError: {e}",
                    "stderr": e.stderr,
                    "exit_code": e.exit_code,
                    "details": str(e),
                }
            )
            return result_dict
        except ServerError as e:
            logger.error(f"ServerError during {stage}: {e}")
            result_dict.update(
                {
                    "error": f"ServerError: {e}",
                    "stderr": getattr(e, "details", str(e)),
                    "exit_code": -1,
                    "details": str(e),
                }
            )
            return result_dict
        except Exception as e:
            logger.error(f"Unexpected error during {stage}: {e}", exc_info=True)
            result_dict.update(
                {
                    "error": f"Unexpected Error: {e}",
                    "exit_code": -1,
                    "details": str(e),
                }
            )
            return result_dict

    async def execute(self, binary_bytes: bytes) -> Dict[str, Any]:
        """
        Sends compiled binary for execution to QAN servers.

        Args:
            binary_bytes: The result from a successful compile step.

        Returns:
            A dictionary similar to precompile's result, adding 'stdout',
            'system_error', and 'timeout_error' flags/fields.
            The 'result' key holds the `ExecutionResult` object on success.
        """
        client = self._get_client()
        stage = "execute"
        result_dict = {
            "success": False,
            "result": None,
            "error": None,
            "stderr": None,
            "stdout": None,
            "exit_code": None,
            "non_deterministic": False,
            "stage": stage,
            "details": None,
            "system_error": False,
            "timeout_error": False,
        }
        try:
            logger.debug(f"Sending execute request...")
            exec_result: ExecutionResult = await client.execute(binary_bytes)
            result_dict.update(
                {
                    "success": True,
                    "result": exec_result,  # Store the ExecutionResult object
                    "stdout": exec_result.stdout,
                    "stderr": exec_result.stderr,
                    "exit_code": exec_result.exit_code,
                    "non_deterministic": False,  # Explicitly deterministic if no error
                }
            )
            logger.debug(
                f"Execute successful (deterministic). Exit code: {exec_result.exit_code}"
            )
            return result_dict

        except NonDeterministicError as e:
            logger.warning(f"NonDeterministicError during {stage}: {e}")
            result_dict.update(
                {
                    "error": f"NonDeterministicError: {e}",
                    "non_deterministic": True,
                    "detection_method": "qan_execute",
                    "details": str(e),
                    # Attempt to get results from embedded ExecutionResult if available
                    "stdout": getattr(e, "stdout", None),
                    "stderr": getattr(e, "stderr", None),
                    "exit_code": getattr(e, "exit_code", None),
                }
            )
            return result_dict
        except SystemError as e:
            logger.error(f"SystemError during {stage}: {e}")
            result_dict.update(
                {
                    "error": f"SystemError: {e}",
                    "system_error": True,
                    "stderr": e.stderr,
                    "exit_code": e.exit_code,
                    "details": str(e),
                }
            )
            return result_dict
        except ExecuteTimeoutError as e:
            logger.warning(f"ExecuteTimeoutError: {e}")
            stdout = getattr(e.response, "stdout", "") if hasattr(e, "response") else ""
            stderr = (
                getattr(e.response, "stderr", e.stderr)
                if hasattr(e, "response")
                else getattr(e, "stderr", "")
            )
            exit_code = (
                getattr(e.response, "exit_code", e.exit_code)
                if hasattr(e, "response")
                else getattr(e, "exit_code", -1)
            )
            result_dict.update(
                {
                    "error": f"ExecuteTimeoutError: {e}",
                    "timeout_error": True,
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": exit_code,
                    "details": str(e),
                }
            )
            return result_dict
        except ExecuteError as e:
            logger.warning(f"ExecuteError: {e}")
            stdout = getattr(e.response, "stdout", "") if hasattr(e, "response") else ""
            stderr = (
                getattr(e.response, "stderr", e.stderr)
                if hasattr(e, "response")
                else getattr(e, "stderr", "")
            )
            exit_code = (
                getattr(e.response, "exit_code", e.exit_code)
                if hasattr(e, "response")
                else getattr(e, "exit_code", -1)
            )
            result_dict.update(
                {
                    "error": f"ExecuteError: {e}",
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": exit_code,
                    "details": str(e),
                }
            )
            return result_dict
        except ServerError as e:
            logger.error(f"ServerError during {stage}: {e}")
            result_dict.update(
                {
                    "error": f"ServerError: {e}",
                    "stderr": getattr(e, "details", str(e)),
                    "exit_code": -1,
                    "details": str(e),
                }
            )
            return result_dict
        except Exception as e:
            logger.error(f"Unexpected error during {stage}: {e}", exc_info=True)
            result_dict.update(
                {
                    "error": f"Unexpected Error: {e}",
                    "exit_code": -1,
                    "details": str(e),
                }
            )
            return result_dict
