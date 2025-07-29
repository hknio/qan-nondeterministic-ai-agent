import asyncio
import base64
import hashlib
import httpx
import logging
import math
from pydantic import BaseModel
from typing import List, Union, Optional, Dict, Any, Set, Callable, TypeVar, Awaitable


logger = logging.getLogger(__name__)

T = TypeVar("T")


###############################################################################
#                               Pydantic Models                               #
###############################################################################
class ProcessResult(BaseModel):
    """
    Represents the result of a process execution.

    :param stdout: Standard output of the process.
    :type stdout: str
    :param stderr: Standard error of the process.
    :type stderr: str
    :param exit_code: Exit code of the process.
    :type exit_code: int
    """

    stdout: str
    stderr: str
    exit_code: int


class PrecompileResponse(BaseModel):
    """
    Represents the server response for a precompile request.

    :param filtered_pem: The filtered PEM if applicable.
    :type filtered_pem: Optional[str]
    :param decompiled_code: The base64-encoded decompiled code.
    :type decompiled_code: Optional[str]
    :param error: The error string if any issue occurred.
    :type error: Optional[str]
    :param process_result: Detailed process execution result.
    :type process_result: ProcessResult
    """

    filtered_pem: Optional[str] = None
    decompiled_code: Optional[str] = None
    error: Optional[str] = None
    process_result: ProcessResult


class CompileResponse(BaseModel):
    """
    Represents the server response for a compile request.

    :param pem_file: Optional PEM file content.
    :type pem_file: Optional[str]
    :param binary: The base64-encoded compiled binary.
    :type binary: Optional[str]
    :param error: The error string if any issue occurred.
    :type error: Optional[str]
    :param process_result: Detailed process execution result.
    :type process_result: ProcessResult
    """

    pem_file: Optional[str] = None
    binary: Optional[str] = None
    error: Optional[str] = None
    process_result: ProcessResult


class ExecuteResponse(BaseModel):
    """
    Represents the server response for an execute request.

    :param stdout: Execution standard output.
    :type stdout: str
    :param stderr: Execution standard error.
    :type stderr: str
    :param exit_code: Exit code of the execution.
    :type exit_code: int
    :param error: The error string if any issue occurred.
    :type error: Optional[str]
    :param debug: Optional debug string for additional info.
    :type debug: Optional[str]
    :param process_result: Detailed process execution result.
    :type process_result: ProcessResult
    """

    stdout: str
    stderr: str
    exit_code: int
    error: Optional[str] = None
    debug: Optional[str] = None
    process_result: ProcessResult


###############################################################################
#                         Request Body Models (client)                       #
###############################################################################
class ContractRequest(BaseModel):
    """
    Represents the body of a request to compile or precompile a contract.

    :param language: The programming language of the contract (e.g., "solidity").
    :type language: str
    :param code: The base64-encoded source code.
    :type code: str
    """

    language: str
    code: str


class ExecuteRequest(BaseModel):
    """
    Represents the body of a request to execute a compiled binary.

    :param binary: The base64-encoded compiled binary.
    :type binary: str
    :param block_chainid: The chain ID for the block context.
    :type block_chainid: int
    :param block_timestamp: The block timestamp for execution context.
    :type block_timestamp: int
    :param block_hash: The block hash for execution context.
    :type block_hash: str
    :param block_coinbase: The block coinbase address.
    :type block_coinbase: str
    :param msg_value: The msg.value context in the execution environment.
    :type msg_value: int
    :param msg_sender: The msg.sender address in the execution environment.
    :type msg_sender: str
    :param tx_origin: The tx.origin address in the execution environment.
    :type tx_origin: str
    :param vm_timeout: The execution timeout in milliseconds.
    :type vm_timeout: int
    """

    binary: str
    block_chainid: int = 1121
    block_timestamp: int = 123456789
    block_hash: str = "0xcafebabe"
    block_coinbase: str = "0xdeadbeef"
    msg_value: int = 1337
    msg_sender: str = "0xfaceb00c"
    tx_origin: str = "0x0000dead"
    vm_timeout: int = 10_000


###############################################################################
#                             QanApiClient Class                              #
###############################################################################
class QanApiClient:
    """
    Asynchronous client for communicating with the QVM Contract Service.

    This client supports the following primary operations:
      - Precompile: Send code for initial processing.
      - Compile: Compile code into a binary.
      - Execute: Run a compiled binary.
      - Retrieve Supported Languages.
      - Health check.

    Retries requests on HTTP 408 (Request Timeout) up to `max_retries`.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        max_retries: int = 2,
        timeout: float = 60.0,
        retry_delay: float = 3.0,
    ):
        """
        Initialize the QanApiClient.

        :param base_url: Base URL of the server's FastAPI instance.
        :type base_url: str
        :param max_retries: Maximum number of retries on HTTP 408 errors.
        :type max_retries: int
        :param timeout: Timeout in seconds for each request.
        :type timeout: float
        :param retry_delay: Delay in seconds before retrying a failed request.
        :type retry_delay: float
        """
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.timeout = timeout
        self.retry_delay = retry_delay

        # Create a shared AsyncClient
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    async def __aenter__(self):
        """
        Enter the asynchronous context manager.

        :return: The QanApiClient instance.
        :rtype: QanApiClient
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the asynchronous context manager, closing resources.

        :param exc_type: Exception type if raised in context.
        :type exc_type: Any
        :param exc_val: Exception value if raised in context.
        :type exc_val: Any
        :param exc_tb: Traceback if exception raised in context.
        :type exc_tb: Any
        """
        await self.close()

    async def close(self):
        """
        Close the underlying HTTP client to free resources.
        """
        await self._client.aclose()

    async def _request(
        self,
        method: str,
        url: str,
        json_data=None,
        retry_count: int = 0,
    ) -> httpx.Response:
        """
        Internal helper method to perform an HTTP request with automatic retries on errors.

        :param method: The HTTP method (GET, POST, etc.).
        :type method: str
        :param url: The endpoint path to request.
        :type url: str
        :param json_data: The JSON-serializable data to send in the request body.
        :type json_data: Any
        :param retry_count: Current retry attempt count.
        :type retry_count: int
        :return: The httpx.Response object.
        :rtype: httpx.Response
        :raises ServerError: If a network, connection issue, or non-200 status code is encountered
            after retries are exhausted.
        """
        response = None
        retry = retry_count < self.max_retries
        try:
            response = await self._client.request(method, url, json=json_data)
            if response.status_code == 200:
                return response
            if not retry:
                raise ServerError(
                    f"HTTP error with status code {response.status_code}",
                    details=response.text,
                )
        except httpx.RequestError as exc:
            if not retry:
                raise ServerError(f"Network error calling {url}: {exc}") from exc
        except httpx.HTTPError as exc:
            if not retry:
                raise ServerError(
                    f"HTTP error calling {url}: {exc}",
                ) from exc

        if (
            response and response.status_code != 408
        ):  # For non-408 errors, wait before retrying
            await asyncio.sleep(self.retry_delay)
        return await self._request(
            method,
            url,
            json_data=json_data,
            retry_count=retry_count + 1,
        )

    ###############################################################################
    #                             Public Endpoints                                #
    ###############################################################################
    async def precompile(self, language: str, code_bytes: bytes) -> PrecompileResponse:
        """
        Call the /precompile endpoint with base64-encoded code.

        :param language: Programming language of the contract (e.g., "solidity").
        :type language: str
        :param code_bytes: The raw source code in bytes.
        :type code_bytes: bytes
        :return: The parsed PrecompileResponse object from the server.
        :rtype: PrecompileResponse
        :raises ServerError: If the server returns a non-2xx status or times out.
        """
        encoded_code = base64.b64encode(code_bytes).decode("utf-8")
        req_body = ContractRequest(language=language, code=encoded_code)
        resp = await self._request("POST", "/precompile", json_data=req_body.dict())
        return PrecompileResponse(**resp.json())

    async def compile(self, language: str, code_bytes: bytes) -> CompileResponse:
        """
        Call the /compile endpoint with base64-encoded code.

        :param language: Programming language of the contract (e.g., "solidity").
        :type language: str
        :param code_bytes: The raw source code in bytes.
        :type code_bytes: bytes
        :return: The parsed CompileResponse object from the server.
        :rtype: CompileResponse
        :raises ServerError: If the server returns a non-2xx status or times out.
        """
        encoded_code = base64.b64encode(code_bytes).decode("utf-8")
        req_body = ContractRequest(language=language, code=encoded_code)
        resp = await self._request("POST", "/compile", json_data=req_body.dict())
        return CompileResponse(**resp.json())

    async def execute(
        self,
        binary_bytes: bytes,
        block_chainid: int = 1121,
        block_timestamp: int = 123456789,
        block_hash: str = "0xcafebabe",
        block_coinbase: str = "0xdeadbeef",
        msg_value: int = 1337,
        msg_sender: str = "0xfaceb00c",
        tx_origin: str = "0x0000dead",
        vm_timeout: int = 10_000,
    ) -> ExecuteResponse:
        """
        Call the /execute endpoint with base64-encoded compiled binary.

        :param binary_bytes: The compiled binary code in bytes.
        :type binary_bytes: bytes
        :param block_chainid: The chain ID for the block context.
        :type block_chainid: int
        :param block_timestamp: The block timestamp for execution context.
        :type block_timestamp: int
        :param block_hash: The block hash for execution context.
        :type block_hash: str
        :param block_coinbase: The block coinbase address.
        :type block_coinbase: str
        :param msg_value: The msg.value in the execution context.
        :type msg_value: int
        :param msg_sender: The msg.sender address in the execution context.
        :type msg_sender: str
        :param tx_origin: The tx.origin address in the execution environment.
        :type tx_origin: str
        :param vm_timeout: The execution timeout in milliseconds.
        :type vm_timeout: int
        :return: The parsed ExecuteResponse object from the server.
        :rtype: ExecuteResponse
        :raises ServerError: If the server returns a non-2xx status or times out.
        """
        encoded_binary = base64.b64encode(binary_bytes).decode("utf-8")
        req_body = ExecuteRequest(
            binary=encoded_binary,
            block_chainid=block_chainid,
            block_timestamp=block_timestamp,
            block_hash=block_hash,
            block_coinbase=block_coinbase,
            msg_value=msg_value,
            msg_sender=msg_sender,
            tx_origin=tx_origin,
            vm_timeout=vm_timeout,
        )
        resp = await self._request("POST", "/execute", json_data=req_body.dict())
        return ExecuteResponse(**resp.json())

    async def supported_languages(self) -> List[str]:
        """
        Retrieve a list of supported languages from the /supported_languages endpoint.

        :return: A list of supported language strings.
        :rtype: List[str]
        :raises ServerError: If the server returns a non-2xx status or times out.
        """
        resp = await self._request("GET", "/supported_languages")
        return resp.json().get("supported_languages", [])

    async def health(self) -> dict:
        """
        Call the /health endpoint to check server status.

        :return: The JSON response from the server (e.g., {"status": "ok"}).
        :rtype: dict
        :raises ServerError: If the server returns a non-2xx status or times out.
        """
        resp = await self._request("GET", "/health")
        return resp.json()


###############################################################################
#                               Custom Exceptions                             #
###############################################################################
class ServerError(Exception):
    """
    Represents a server-side or network-related error.

    :param message: Error message.
    :type message: str
    :param details: Optional additional detail about the error.
    :type details: Optional[str]
    """

    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message)
        self.details = details

    def __str__(self) -> str:
        return f"ServerError(msg={self.args[0]}, details={self.details})"


class PrecompileError(Exception):
    """
    Error specifically related to the precompile step.

    :param error_message: The 'error' field from the precompile response.
    :type error_message: str
    :param exit_code: The process exit code.
    :type exit_code: int
    :param stderr: The process stderr output.
    :type stderr: str
    """

    def __init__(self, error_message: str, exit_code: int, stderr: str):
        super().__init__(error_message)
        self.exit_code = exit_code
        self.stderr = stderr

    def __str__(self) -> str:
        return (
            f"PrecompileError(msg={self.args[0]}, "
            f"exit_code={self.exit_code}, stderr={self.stderr})"
        )


class CompileError(Exception):
    """
    Error specifically related to the compile step.

    :param error_message: The 'error' field from the compile response.
    :type error_message: str
    :param exit_code: The process exit code.
    :type exit_code: int
    :param stderr: The process stderr output.
    :type stderr: str
    """

    def __init__(self, error_message: str, exit_code: int, stderr: str):
        super().__init__(error_message)
        self.exit_code = exit_code
        self.stderr = stderr

    def __str__(self) -> str:
        return (
            f"CompileError(msg={self.args[0]}, "
            f"exit_code={self.exit_code}, stderr={self.stderr})"
        )


class ExecuteError(Exception):
    """
    Error specifically related to the execute step.

    :param error_message: The 'error' field from the execute response.
    :type error_message: str
    :param exit_code: The process exit code.
    :type exit_code: int
    :param stderr: The process stderr output.
    :type stderr: str
    """

    def __init__(
        self, error_message: str, exit_code: int, stderr: str, response: ExecuteResponse
    ):
        super().__init__(error_message)
        self.exit_code = exit_code
        self.stderr = stderr
        self.response = response

    def __str__(self) -> str:
        return (
            f"ExecuteError(msg={self.args[0]}, "
            f"exit_code={self.exit_code}, stderr={self.stderr})"
        )


class SystemError(Exception):
    """
    Error specifically related to a system crash during execution.

    :param error_message: The 'error' field from the execute response.
    :type error_message: str
    :param exit_code: The process exit code.
    :type exit_code: int
    :param stderr: The process stderr output.
    :type stderr: str
    """

    def __init__(
        self, error_message: str, exit_code: int, stderr: str, response: ExecuteResponse
    ):
        super().__init__(error_message)
        self.exit_code = exit_code
        self.stderr = stderr
        self.response = response

    def __str__(self) -> str:
        return (
            f"SystemError(msg={self.args[0]}, "
            f"exit_code={self.exit_code}, stderr={self.stderr})"
        )


class ExecuteTimeoutError(Exception):
    """
    Error specifically related to a timeout during execution.

    :param error_message: The 'error' field from the execute response.
    :type error_message: str
    :param exit_code: The process exit code.
    :type exit_code: int
    :param stderr: The process stderr output.
    :type stderr: str
    """

    def __init__(
        self, error_message: str, exit_code: int, stderr: str, response: ExecuteResponse
    ):
        super().__init__(error_message)
        self.exit_code = exit_code
        self.stderr = stderr
        self.response = response

    def __str__(self) -> str:
        return (
            f"ExecuteTimeoutError(msg={self.args[0]}, "
            f"exit_code={self.exit_code}, stderr={self.stderr})"
        )


###############################################################################
#                              Execution Result                               #
###############################################################################
class ExecutionResult:
    """
    Container for execution results from the server.

    :param stdout: Standard output produced by execution.
    :type stdout: str
    :param stderr: Standard error produced by execution.
    :type stderr: str
    :param exit_code: Exit code of the execution.
    :type exit_code: int
    """

    def __init__(self, stdout: str, stderr: str, exit_code: int):
        self.stdout: str = stdout
        self.stderr: str = stderr
        self.exit_code: int = exit_code

    def __str__(self) -> str:
        return (
            f"ExecutionResult(stdout={self.stdout}, "
            f"stderr={self.stderr}, exit_code={self.exit_code})"
        )


###############################################################################
#                               QanClient Class                               #
###############################################################################
class QanClient:
    """
    A higher-level client that uses QanApiClient under the hood.

    It translates low-level HTTP responses into:
      - Return values (e.g., bytes, ExecutionResult).
      - Custom exceptions (e.g., PrecompileError, CompileError, ExecuteError, ServerError).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        max_retries: int = 2,
        timeout: float = 60.0,
        retry_delay: float = 3.0,
    ):
        """
        Initialize the QanClient with an internal QanApiClient.

        :param base_url: Base URL of the server's FastAPI instance.
        :type base_url: str
        :param max_retries: Maximum number of retries on HTTP 408 errors.
        :type max_retries: int
        :param timeout: Timeout in seconds for each request.
        :type timeout: float
        :param retry_delay: Delay in seconds before retrying a failed request.
        :type retry_delay: float
        """
        self._api = QanApiClient(
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
            retry_delay=retry_delay,
        )

    async def __aenter__(self):
        """
        Enter the asynchronous context manager.

        :return: The QanClient instance.
        :rtype: QanClient
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the asynchronous context manager, closing resources.

        :param exc_type: Exception type if raised in context.
        :type exc_type: Any
        :param exc_val: Exception value if raised in context.
        :type exc_val: Any
        :param exc_tb: Traceback if exception raised in context.
        :type exc_tb: Any
        """
        await self.close()

    async def close(self):
        """
        Close the underlying API client.
        """
        await self._api.close()

    async def precompile(self, language: str, code_bytes: bytes) -> bytes:
        """
        Perform a precompile step on the given code.

        :param language: Programming language of the code (e.g., "solidity").
        :type language: str
        :param code_bytes: Raw source code in bytes.
        :type code_bytes: bytes
        :return: The base64-decoded decompiled code.
        :rtype: bytes
        :raises PrecompileError: If the response contains an error field.
        :raises ServerError: If a non-2xx status or timeout occurs.
        """
        resp = await self._api.precompile(language, code_bytes)
        if resp.error:
            raise PrecompileError(
                error_message=resp.error,
                exit_code=resp.process_result.exit_code,
                stderr=resp.process_result.stderr,
            )

        if not resp.decompiled_code:
            raise PrecompileError(
                error_message="No decompiled code returned",
                exit_code=resp.process_result.exit_code,
                stderr=resp.process_result.stderr,
            )

        return base64.b64decode(resp.decompiled_code)

    async def compile(self, language: str, code_bytes: bytes) -> bytes:
        """
        Compile the given code into a binary.

        :param language: Programming language of the code (e.g., "solidity").
        :type language: str
        :param code_bytes: Raw source code in bytes.
        :type code_bytes: bytes
        :return: The base64-decoded compiled binary.
        :rtype: bytes
        :raises CompileError: If the response contains an error field.
        :raises ServerError: If a non-2xx status or timeout occurs.
        """
        resp = await self._api.compile(language, code_bytes)
        if resp.error:
            raise CompileError(
                error_message=resp.error,
                exit_code=resp.process_result.exit_code,
                stderr=resp.process_result.stderr,
            )

        if not resp.binary:
            raise CompileError(
                error_message="No binary returned",
                exit_code=resp.process_result.exit_code,
                stderr=resp.process_result.stderr,
            )

        return base64.b64decode(resp.binary)

    async def execute(
        self,
        binary_bytes: bytes,
        block_chainid: int = 1121,
        block_timestamp: int = 123456789,
        block_hash: str = "0xcafebabe",
        block_coinbase: str = "0xdeadbeef",
        msg_value: int = 1337,
        msg_sender: str = "0xfaceb00c",
        tx_origin: str = "0x0000dead",
        vm_timeout: int = 10_000,
    ) -> ExecutionResult:
        """
        Execute the given compiled binary.

        :param binary_bytes: The compiled binary code in bytes.
        :type binary_bytes: bytes
        :param block_chainid: The chain ID for the block context.
        :type block_chainid: int
        :param block_timestamp: The block timestamp for execution context.
        :type block_timestamp: int
        :param block_hash: The block hash for execution context.
        :type block_hash: str
        :param block_coinbase: The block coinbase address.
        :type block_coinbase: str
        :param msg_value: The msg.value in the execution context.
        :type msg_value: int
        :param msg_sender: The msg.sender address in the execution context.
        :type msg_sender: str
        :param tx_origin: The tx.origin address in the execution environment.
        :type tx_origin: str
        :param vm_timeout: The execution timeout in milliseconds.
        :type vm_timeout: int
        :return: The execution result (stdout, stderr, exit_code).
        :rtype: ExecutionResult
        :raises ExecuteError: If the response contains an error field.
        :raises SystemError: If the response contains the error "HERMIT_CRASHED".
        :raises ExecuteTimeoutError: If the response contains the error "TIMEOUT".
        :raises ServerError: If a non-2xx status or timeout occurs.
        """
        resp = await self._api.execute(
            binary_bytes,
            block_chainid=block_chainid,
            block_timestamp=block_timestamp,
            block_hash=block_hash,
            block_coinbase=block_coinbase,
            msg_value=msg_value,
            msg_sender=msg_sender,
            tx_origin=tx_origin,
            vm_timeout=vm_timeout,
        )

        if resp.error:
            if resp.error == "HERMIT_CRASHED":
                raise SystemError(
                    error_message=resp.error,
                    exit_code=resp.exit_code,
                    stderr=resp.stderr,
                    response=resp,
                )
            elif resp.error == "TIMEOUT":
                raise ExecuteTimeoutError(
                    error_message=resp.error,
                    exit_code=resp.exit_code,
                    stderr=resp.stderr,
                    response=resp,
                )
            else:
                raise ExecuteError(
                    error_message=resp.error,
                    exit_code=resp.exit_code,
                    stderr=resp.stderr,
                    response=resp,
                )

        return ExecutionResult(
            stdout=resp.stdout,
            stderr=resp.stderr,
            exit_code=resp.exit_code,
        )

    async def run_code(self, language: str, code_bytes: bytes) -> ExecutionResult:
        """
        Run code (precompile -> compile -> execute) on all servers and verify determinism.

        :param language: Programming language (e.g., "solidity").
        :type language: str
        :param code_bytes: The raw source code in bytes.
        :type code_bytes: bytes
        :return: The final execution result if majority match in each step.
        :rtype: ExecutionResult
        :raises NonDeterministicError: If results differ among servers.
        :raises PrecompileError: If the majority returns this error in precompile step.
        :raises CompileError: If the majority returns this error in compile step.
        :raises ExecuteError: If the majority returns this error in execute step.
        :raises SystemError: If the response contains the error "HERMIT_CRASHED".
        :raises ExecuteTimeoutError: If the response contains the error "TIMEOUT".
        :raises ServerError: If insufficient valid responses or other server issues occur.
        """
        decompiled_code = await self.precompile(language, code_bytes)
        binary = await self.compile(language, decompiled_code)
        execution_result = await self.execute(binary)
        return execution_result

    async def supported_languages(self) -> Set[str]:
        """
        Retrieve the set of supported languages from the server.

        :return: A set of supported language strings.
        :rtype: Set[str]
        :raises ServerError: If a non-2xx status or timeout occurs.
        """
        languages = await self._api.supported_languages()
        return set(languages)

    async def health(self) -> bool:
        """
        Check the server health.

        :return: True if the server status is "ok", otherwise False.
        :rtype: bool
        :raises ServerError: If a non-2xx status or timeout occurs.
        """
        resp = await self._api.health()
        return resp.get("status") == "ok"


###############################################################################
#                          NonDeterministicError                              #
###############################################################################
class NonDeterministicError(Exception):
    """
    Raised when multiple servers produce differing (non-deterministic) results.

    :param phase: One of "precompile", "compile", "execute".
    :type phase: str
    :param differences: Dictionary describing how servers differ. Key is a
        string representation of the result/error, value is list of servers.
    :type differences: Dict[str, List[str]]
    :param message: Optional user-friendly message.
    :type message: Optional[str]
    """

    def __init__(
        self,
        phase: str,
        differences: Dict[str, List[str]],
        message: Optional[str] = None,
    ):
        self.phase = phase
        self.differences = differences

        if not message:
            lines = [f"Non-deterministic results encountered during '{phase}':"]
            for result_repr, servers in differences.items():
                lines.append(f"  Result: {result_repr} => from servers: {servers}")
            message = "\n".join(lines)

        super().__init__(message)

    def __str__(self) -> str:
        return (
            f"NonDeterministicError(phase={self.phase}, "
            f"differences={self.differences}, message={self.args[0]})"
        )


###############################################################################
#                              QanMultiClient                                 #
###############################################################################
class QanMultiClient:
    """
    A client that wraps multiple QanClient instances and enforces deterministic responses.

    This client runs requests against multiple servers concurrently and verifies
    that their results match (majority vote). If the majority results differ from
    each other, a NonDeterministicError is raised.
    """

    def __init__(
        self,
        servers: List[str],
        max_retries: int = 2,
        timeout: float = 60.0,
        retry_delay: float = 3.0,
    ):
        """
        :param servers: List of base URLs for each server.
        :type servers: List[str]
        :param max_retries: Maximum number of retries on HTTP 408 errors.
        :type max_retries: int
        :param timeout: Timeout in seconds for each request.
        :type timeout: float
        :param retry_delay: Delay in seconds before retrying a failed request.
        :type retry_delay: float
        """
        self._servers = servers
        self._clients = [
            QanClient(
                base_url=base_url,
                max_retries=max_retries,
                timeout=timeout,
                retry_delay=retry_delay,
            )
            for base_url in servers
        ]
        self._majority_threshold = math.ceil(len(self._clients) / 2)

    async def __aenter__(self):
        """
        Enter the asynchronous context manager.

        :return: The QanMultiClient instance.
        :rtype: QanMultiClient
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the asynchronous context manager, closing resources.

        :param exc_type: Exception type if raised in context.
        :type exc_type: Any
        :param exc_val: Exception value if raised in context.
        :type exc_val: Any
        :param exc_tb: Traceback if exception raised in context.
        :type exc_tb: Any
        """
        await self.close()

    async def close(self):
        """
        Close all underlying QanClient instances.
        """
        close_tasks = [client.close() for client in self._clients]
        await asyncio.gather(*close_tasks)

    ###########################################################################
    #                              Public Methods                             #
    ###########################################################################

    async def precompile(self, language: str, code_bytes: bytes) -> bytes:
        """
        Call precompile on all servers concurrently and verify determinism.

        :param language: Programming language (e.g., "solidity").
        :type language: str
        :param code_bytes: The raw source code in bytes.
        :type code_bytes: bytes
        :return: A consistent precompile result (decompiled code) if majority match.
        :rtype: bytes
        :raises NonDeterministicError: If results differ among servers.
        :raises PrecompileError: If the majority returns this error.
        :raises ServerError: If insufficient valid responses or other server issues occur.
        """
        phase = "precompile"
        calls = [
            self._call_with(client.precompile, language, code_bytes)
            for client in self._clients
        ]
        results = await self._gather_calls(phase=phase, calls=calls)
        if isinstance(results, Exception):
            raise results
        return results

    async def compile(self, language: str, code_bytes: bytes) -> bytes:
        """
        Call compile on all servers concurrently and verify determinism.

        :param language: Programming language (e.g., "solidity").
        :type language: str
        :param code_bytes: The raw source code in bytes.
        :type code_bytes: bytes
        :return: A consistent compile result (compiled binary) if majority match.
        :rtype: bytes
        :raises NonDeterministicError: If results differ among servers.
        :raises CompileError: If the majority returns this error.
        :raises ServerError: If insufficient valid responses or other server issues occur.
        """
        phase = "compile"
        calls = [
            self._call_with(client.compile, language, code_bytes)
            for client in self._clients
        ]
        results = await self._gather_calls(phase=phase, calls=calls)
        if isinstance(results, Exception):
            raise results
        return results

    async def execute(
        self,
        binary_bytes: bytes,
        block_chainid: int = 1121,
        block_timestamp: int = 123456789,
        block_hash: str = "0xcafebabe",
        block_coinbase: str = "0xdeadbeef",
        msg_value: int = 1337,
        msg_sender: str = "0xfaceb00c",
        tx_origin: str = "0x0000dead",
        vm_timeout: int = 10_000,
    ) -> ExecutionResult:
        """
        Call execute on all servers concurrently and verify determinism.

        :param binary_bytes: The compiled binary code in bytes.
        :type binary_bytes: bytes
        :param block_chainid: The chain ID for the block context.
        :type block_chainid: int
        :param block_timestamp: The block timestamp for execution context.
        :type block_timestamp: int
        :param block_hash: The block hash for execution context.
        :type block_hash: str
        :param block_coinbase: The block coinbase address.
        :type block_coinbase: str
        :param msg_value: The msg.value in the execution context.
        :type msg_value: int
        :param msg_sender: The msg.sender address in the execution context.
        :type msg_sender: str
        :param tx_origin: The tx.origin address in the execution environment.
        :type tx_origin: str
        :param vm_timeout: The execution timeout in milliseconds.
        :type vm_timeout: int
        :return: A consistent execution result if majority match.
        :rtype: ExecutionResult
        :raises NonDeterministicError: If results differ among servers.
        :raises ExecuteError: If the majority returns this error.
        :raises SystemError: If the response contains the error "HERMIT_CRASHED".
        :raises ExecuteTimeoutError: If the response contains the error "TIMEOUT".
        :raises ServerError: If insufficient valid responses or other server issues occur.
        """
        phase = "execute"
        calls = [
            self._call_with(
                client.execute,
                binary_bytes,
                block_chainid=block_chainid,
                block_timestamp=block_timestamp,
                block_hash=block_hash,
                block_coinbase=block_coinbase,
                msg_value=msg_value,
                msg_sender=msg_sender,
                tx_origin=tx_origin,
                vm_timeout=vm_timeout,
            )
            for client in self._clients
        ]
        results = await self._gather_calls(phase=phase, calls=calls)
        if isinstance(results, Exception):
            raise results
        return results

    async def run_code(self, language: str, code_bytes: bytes) -> ExecutionResult:
        """
        Run code (precompile -> compile -> execute) on all servers and verify determinism.

        :param language: Programming language (e.g., "solidity").
        :type language: str
        :param code_bytes: The raw source code in bytes.
        :type code_bytes: bytes
        :return: The final execution result if majority match in each step.
        :rtype: ExecutionResult
        :raises NonDeterministicError: If results differ among servers.
        :raises PrecompileError: If the majority returns this error in precompile step.
        :raises CompileError: If the majority returns this error in compile step.
        :raises ExecuteError: If the majority returns this error in execute step.
        :raises SystemError: If the response contains the error "HERMIT_CRASHED".
        :raises ExecuteTimeoutError: If the response contains the error "TIMEOUT".
        :raises ServerError: If insufficient valid responses or other server issues occur.
        """
        decompiled_code = await self.precompile(language, code_bytes)
        binary = await self.compile(language, decompiled_code)
        execution_result = await self.execute(binary)
        return execution_result

    async def supported_languages(self) -> Set[str]:
        """
        Retrieve supported languages from all servers, ignoring errors if enough valid responses exist.

        :return: Intersection set of supported languages among the majority of servers.
        :rtype: Set[str]
        :raises ServerError: If insufficient valid responses are available.
        """
        calls = [
            self._call_with(client.supported_languages) for client in self._clients
        ]
        results = await asyncio.gather(*calls, return_exceptions=True)

        ok_results = []
        for idx, r in enumerate(results):
            if isinstance(r, ServerError):
                logger.warning(
                    f"[supported_languages] ServerError from {self._servers[idx]}: {r}"
                )
            elif isinstance(r, Exception):
                logger.warning(
                    f"[supported_languages] Unexpected error from {self._servers[idx]}: {r}"
                )
            else:
                ok_results.append(r)

        if len(ok_results) == 0 and len(results) > 0:
            raise results[0]

        if len(ok_results) < self._majority_threshold:
            raise ServerError("Not enough valid responses for supported_languages")

        # Return the intersection of all successful sets
        intersection = set.intersection(*ok_results) if ok_results else set()
        return intersection

    async def health(self) -> bool:
        """
        Check the health of all servers. Returns True if the majority are healthy.

        :return: True if a majority of servers report healthy status, otherwise False.
        :rtype: bool
        """
        calls = [self._call_with(client.health) for client in self._clients]
        results = await asyncio.gather(*calls, return_exceptions=True)

        ok_count = 0
        for idx, r in enumerate(results):
            if isinstance(r, ServerError):
                logger.warning(f"[health] ServerError from {self._servers[idx]}: {r}")
            elif isinstance(r, Exception):
                logger.warning(
                    f"[health] Unexpected error from {self._servers[idx]}: {r}"
                )
            else:
                if r is True:
                    ok_count += 1

        if ok_count == 0 and len(results) > 0:
            raise results[0]

        return ok_count >= self._majority_threshold

    ###########################################################################
    #                            Internal Helpers                             #
    ###########################################################################
    async def _call_with(
        self, func: Callable[..., Awaitable[T]], *args, **kwargs
    ) -> Union[T, Exception]:
        """
        Execute a function call, capturing any exceptions and returning them instead of raising.

        :param func: The async function to call.
        :type func: Callable[..., Awaitable[T]]
        :param args: Positional arguments for the function.
        :type args: Any
        :param kwargs: Keyword arguments for the function.
        :type kwargs: Any
        :return: The function result or the exception that was raised.
        :rtype: Union[T, Exception]
        """
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            return e

    async def _gather_calls(
        self,
        phase: str,
        calls: List[Awaitable[Union[bytes, ExecutionResult, Exception]]],
    ) -> Union[bytes, ExecutionResult, Exception]:
        """
        Gather calls concurrently and apply majority logic to detect non-deterministic outcomes.

        :param phase: One of "precompile", "compile", or "execute".
        :type phase: str
        :param calls: List of async call coroutines.
        :type calls: List[Awaitable[Union[bytes, ExecutionResult, Exception]]]
        :return: A single consistent success result or a single consistent error (Exception).
        :rtype: Union[bytes, ExecutionResult, Exception]
        :raises ServerError: If insufficient valid (non-ServerError) responses.
        :raises NonDeterministicError: If more than one distinct successful/error outcome is found.
        """
        results = await asyncio.gather(*calls, return_exceptions=True)

        # Separate server errors from others
        valid_results = []
        exceptions = []
        for i, r in enumerate(results):
            if isinstance(r, ServerError):
                logger.warning(f"[{phase}] ServerError from {self._servers[i]}: {r}")
            elif isinstance(r, Exception):
                exceptions.append(r)
            else:
                valid_results.append((i, r))

        if len(valid_results) == 0:
            if len(exceptions) > 0:
                raise exceptions[0]
            elif len(results) > 0:
                raise results[0]
            else:
                raise ServerError(f"No valid responses for phase '{phase}'")

        if len(valid_results) < self._majority_threshold:
            if len(exceptions) > 0:
                raise exceptions[0]
            else:
                raise ServerError(f"Not enough valid responses for phase '{phase}'")

        # Group by canonical representation
        representation_to_servers: Dict[str, List[str]] = {}
        representation_to_object: Dict[
            str, Union[bytes, ExecutionResult, Exception]
        ] = {}

        for idx, r in valid_results:
            rep = self._canonical_representation(phase, r)
            server_name = self._servers[idx]
            representation_to_servers.setdefault(rep, []).append(server_name)
            if rep not in representation_to_object:
                representation_to_object[rep] = r

        if len(representation_to_servers) > 1:
            raise NonDeterministicError(phase, representation_to_servers)

        # Exactly one representation
        only_rep_key = next(iter(representation_to_servers.keys()))
        obj = representation_to_object[only_rep_key]
        return obj

    def _canonical_representation(self, phase: str, result: Any) -> str:
        """
        Create a hashable representation of a result for comparison among servers.

        :param phase: The phase of the operation ("precompile", "compile", or "execute").
        :type phase: str
        :param result: The result object (bytes, ExecutionResult, or an Exception).
        :type result: Any
        :return: A string representing the object's outcome.
        :rtype: str
        """
        if isinstance(result, bytes):
            return f"bytes({hashlib.sha1(result).hexdigest()})"
        return str(result)
