import os
import time
import logging
from typing import Dict, Any, Optional, List, Set

# Assume settings and utils are correctly configured and importable
from nondeterministic_agent.utils.file_utils import save_to_json, load_from_json
from nondeterministic_agent.utils.string_utils import sanitize_filename

logger = logging.getLogger(__name__)


class StateManager:
    """
    Manages loading, accessing, and saving the execution state (completed tests).
    Handles state persistence for a specific scope/language combination.
    """

    # Define the structure of a completion entry
    COMPLETION_ENTRY_KEYS = {
        "key",
        "test_id",
        "subject",
        "scope",
        "language",
        "completed_at",
        "status",
    }

    def __init__(self, service_settings: Any, scope_name: str, language: str):
        """
        Initializes the StateManager for a specific scope and language.

        Args:
            service_settings: The application settings instance.
            scope_name: The name of the execution scope.
            language: The programming language being processed.
        """
        self.settings = service_settings
        self.scope_name = scope_name  # Store original scope name
        self.language = language
        # Use sanitized scope name for the state directory path, but language-specific file
        sanitized_scope_dir_name = sanitize_filename(scope_name)
        self.state_dir_for_scope = os.path.join(
            self.settings.STATE_DIR, sanitized_scope_dir_name
        )
        self.state_file = os.path.join(
            self.state_dir_for_scope, f"execution_state_{self.language}.json"
        )
        # Stores list of completion entry dictionaries relevant to this scope/language
        self.completed_tests: List[Dict[str, Any]] = []
        logger.info(
            f"StateManager initialized for Scope='{scope_name}', Language='{language}'."
        )
        logger.info(f"State file path: {self.state_file}")

    def load_state(self) -> None:
        """
        Loads the execution state from the JSON file for the current scope/language.
        Populates the in-memory `self.completed_tests` list.
        """
        self.completed_tests = []  # Clear existing state
        loaded_data = load_from_json(self.state_file)

        if loaded_data is None:
            logger.info(
                f"No existing state file found or failed to load: {self.state_file}"
            )
            return

        # Older format might be just a list, newer format is a dict containing the list
        entries_list = None
        if isinstance(loaded_data, dict):
            # Check if scope/language in metadata matches current instance
            # This helps prevent loading state from a different context if file path was reused
            meta_scope = loaded_data.get("scope")
            meta_lang = loaded_data.get("language")
            if meta_scope != self.scope_name or meta_lang != self.language:
                logger.warning(
                    f"State file {self.state_file} metadata mismatch! "
                    f"Expected {self.scope_name}/{self.language}, found {meta_scope}/{meta_lang}. "
                    f"Ignoring loaded state."
                )
                return
            entries_list = loaded_data.get("completed_tests")
        elif isinstance(loaded_data, list):
            # Support older format directly
            logger.debug("Loaded state file seems to be older list format.")
            entries_list = loaded_data

        if not isinstance(entries_list, list):
            logger.warning(
                f"Invalid format in state file {self.state_file}: 'completed_tests' is not a list or file is not dict/list."
            )
            return

        loaded_count = 0
        valid_count = 0
        for entry in entries_list:
            loaded_count += 1
            if not isinstance(entry, dict):
                logger.warning("Skipping non-dictionary entry in loaded state.")
                continue
            # Validate structure and filter for current scope/language (redundant if metadata matched, but safer)
            if (
                entry.get("scope") == self.scope_name
                and entry.get("language") == self.language
                and self.COMPLETION_ENTRY_KEYS.issubset(entry.keys())
            ):
                self.completed_tests.append(entry)
                valid_count += 1
            else:
                logger.debug(
                    f"Filtering out state entry: {entry.get('key', 'N/A')} due to mismatch or missing keys."
                )

        logger.info(
            f"Loaded {valid_count} valid state entries for {self.scope_name}/{self.language} from {self.state_file} (out of {loaded_count} total entries in file)."
        )

    def get_completed_test_keys(self) -> Set[str]:
        """
        Returns a set of unique keys for completed tests currently in memory.
        The key format is typically '{scope}_{subject}_{test_id}'.
        """
        return set(
            entry.get("key", "") for entry in self.completed_tests if entry.get("key")
        )

    def get_completed_status(self, test_id: str, subject: str) -> Optional[str]:
        """
        Finds the completion entry for a specific test and returns its status.

        Args:
            test_id: The ID of the test.
            subject: The subject of the test.

        Returns:
            The status string if found, otherwise None.
        """
        # Construct the expected key
        expected_key = f"{self.scope_name}_{subject}_{test_id}"
        for entry in self.completed_tests:
            if entry.get("key") == expected_key:
                return entry.get("status")
        return None  # Not found

    def mark_completed(self, test_id: str, subject: str, status: str) -> None:
        """
        Updates the in-memory state list to mark a test as completed or
        updates its status if already present. Does not save immediately.

        Args:
            test_id: The ID of the test.
            subject: The subject of the test.
            status: The final status string for the test.
        """
        if not test_id or not subject or not status:
            logger.error("mark_completed called with missing arguments.")
            return

        completion_key = f"{self.scope_name}_{subject}_{test_id}"
        now_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

        new_entry = {
            "key": completion_key,
            "test_id": test_id,
            "subject": subject,
            "scope": self.scope_name,
            "language": self.language,
            "completed_at": now_utc,
            "status": status,
        }

        # Find if entry already exists and update it
        found_index = -1
        for i, entry in enumerate(self.completed_tests):
            if entry.get("key") == completion_key:
                found_index = i
                break

        if found_index != -1:
            logger.debug(
                f"Updating existing completion status for {test_id} ({subject}) to {status}"
            )
            self.completed_tests[found_index] = new_entry
        else:
            logger.debug(
                f"Marking test {test_id} ({subject}) as completed with status {status}"
            )
            self.completed_tests.append(new_entry)

    def save_state(self) -> bool:
        """
        Saves the current in-memory execution state (`self.completed_tests`)
        to the JSON file.

        Returns:
            True if save was successful, False otherwise.
        """
        logger.info(
            f"Saving execution state for {self.scope_name}/{self.language} ({len(self.completed_tests)} entries) to {self.state_file}"
        )

        # Prepare data in the new format with metadata
        state_to_save = {
            "scope": self.scope_name,
            "language": self.language,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "completed_tests": self.completed_tests,  # Already filtered during load/mark
        }

        if save_to_json(state_to_save, self.state_file):
            return True
        else:
            # Error logged by save_to_json
            return False
