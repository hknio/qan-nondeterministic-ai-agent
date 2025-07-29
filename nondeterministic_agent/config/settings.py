import os
from typing import List, Dict
from pydantic_settings import BaseSettings, SettingsConfigDict

# Determine project root dynamically assuming settings.py is in nondeterministic_agent/config
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_ENV_PATH = os.path.join(PROJECT_ROOT, ".env")


class Settings(BaseSettings):
    # --- Environment File ---
    model_config = SettingsConfigDict(
        env_file=DEFAULT_ENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra fields from .env not defined here
    )

    # --- Project Structure ---
    PROJECT_ROOT: str = PROJECT_ROOT
    RESULTS_DIR: str = os.path.join(PROJECT_ROOT, "results")
    PLAN_DIR: str = os.path.join(RESULTS_DIR, "plan")
    TESTS_DIR: str = os.path.join(RESULTS_DIR, "tests")
    API_RESULTS_DIR: str = os.path.join(RESULTS_DIR, "api_results")
    METRICS_DIR: str = os.path.join(RESULTS_DIR, "metrics")
    STATE_DIR: str = os.path.join(RESULTS_DIR, "state")
    HISTORY_DIR: str = os.path.join(RESULTS_DIR, "history")
    SCOPE_DIR: str = os.path.join(RESULTS_DIR, "scope")  # From planning.py

    # --- QAN Server Configuration ---
    QAN_SERVERS_STR: str = "http://s1.server.com:8000 http://s2.server.com:8000 http://s3.server.com:8000"  # String from env
    QAN_TIMEOUT: float = 60.0
    QAN_MAX_RETRIES: int = 2
    QAN_RETRY_DELAY: float = 3.0

    # Derived QAN Server List
    @property
    def QAN_SERVERS(self) -> List[str]:
        return self.QAN_SERVERS_STR.split()

    # --- Language Configuration ---
    SUPPORTED_LANGUAGES: List[str] = [
        "typescript",
        "rust",
        "ruby",
        "php",
        "perl",
        "kotlin",
        "javascript",
        "golang",
        "cxx",
        "c",
        "python",
        # "scala", "java", "csharp" # Add back when ready
    ]
    LANGUAGE_DETAILS: Dict[str, Dict[str, str]] = {
        "typescript": {"extension": ".ts", "placeholder_module": "ts"},
        "scala": {"extension": ".scala", "placeholder_module": "scala"},
        "rust": {"extension": ".rs", "placeholder_module": "rust"},
        "ruby": {"extension": ".rb", "placeholder_module": "ruby"},
        "php": {"extension": ".php", "placeholder_module": "php"},
        "perl": {"extension": ".pl", "placeholder_module": "perl"},
        "kotlin": {"extension": ".kt", "placeholder_module": "kotlin"},
        "javascript": {"extension": ".js", "placeholder_module": "js"},
        "java": {"extension": ".java", "placeholder_module": "java"},
        "golang": {"extension": ".go", "placeholder_module": "go"},
        "cxx": {"extension": ".cpp", "placeholder_module": "cpp"},
        "csharp": {"extension": ".cs", "placeholder_module": "csharp"},
        "c": {"extension": ".c", "placeholder_module": "c"},
        "python": {"extension": ".py", "placeholder_module": "py"},
    }

    # --- Attempt Limits ---
    MAX_PRECOMPILE_ATTEMPTS: int = 3  # Max *fix* attempts per stage
    MAX_COMPILE_ATTEMPTS: int = 3  # Max *fix* attempts per stage
    MAX_EXECUTE_ATTEMPTS: int = 3  # Max *fix* attempts per stage
    MAX_ACTUAL_PIPELINE_ATTEMPTS: int = (
        30  # Max *total* pipeline loop iterations (saves from infinite loops)
    )
    NON_DET_ATTEMPTS: int = 1  # For forcing non-determinism after initial success

    # --- AI Model Configuration (Defaults if not in .env) ---
    # Execution Models
    CODE_WRITER_MODEL: str = "gemini/gemini-2.0-flash-exp"
    CODE_FIXER_MODEL: str = "gemini/gemini-2.0-flash-exp"
    FORCE_NON_DETERMINISM_MODEL: str = "gemini/gemini-2.0-flash-exp"
    # Planning Model
    PLANNING_MODEL: str = "deepseek/deepseek-reasoner"

    # --- Other ---
    # Add any other constants you want to manage centrally


# Create a single instance to be imported
settings = Settings()

# Ensure history directory exists (can be done here or on first use)
os.makedirs(settings.HISTORY_DIR, exist_ok=True)
