import importlib
import os
import logging
import yaml
import json
from typing import Dict, Any, Optional

from nondeterministic_agent.config.settings import settings

logger = logging.getLogger(__name__)


def save_to_file(content: str, file_path: str) -> bool:
    """
    Saves string content to a file, creating directories if necessary.

    Args:
        content (str): The string content to save.
        file_path (str): The full path to the file.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        dir_name = os.path.dirname(file_path)
        if dir_name:  # Ensure dirname is not empty (for relative paths in current dir)
            os.makedirs(dir_name, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
        logger.debug(f"Content successfully saved to {file_path}")
        return True
    except OSError as e:
        logger.error(f"OS Error saving file {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving file {file_path}: {e}", exc_info=True)
        return False


def save_to_yaml(data: Dict[str, Any], file_path: str) -> bool:
    """
    Saves a dictionary to a YAML file, creating directories if necessary.

    Args:
        data (Dict[str, Any]): The dictionary data to save.
        file_path (str): The full path to the YAML file.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(
                data, f, default_flow_style=False, allow_unicode=True, sort_keys=False
            )
        logger.info(f"Data successfully saved to YAML file: {file_path}")
        return True
    except yaml.YAMLError as e:
        logger.error(f"YAML Error saving data to {file_path}: {e}")
        return False
    except OSError as e:
        logger.error(f"OS Error saving YAML file {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(
            f"Unexpected error saving YAML file {file_path}: {e}", exc_info=True
        )
        return False


def load_from_yaml(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Loads data from a YAML file.

    Args:
        file_path (str): The full path to the YAML file.

    Returns:
        Optional[Dict[str, Any]]: The loaded data as a dictionary, or None if error occurs.
    """
    if not os.path.exists(file_path):
        logger.debug(f"YAML file not found: {file_path}")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        # Basic check if loaded data is a dictionary (often expected)
        if not isinstance(data, dict):
            logger.warning(
                f"YAML file {file_path} did not contain a dictionary (loaded type: {type(data)})."
            )
            # Depending on use case, might return data or None. Let's return data for flexibility.
        logger.debug(f"Data successfully loaded from YAML file: {file_path}")
        return data
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {file_path}: {e}")
        return None
    except OSError as e:
        logger.error(f"OS Error loading YAML file {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error loading YAML file {file_path}: {e}", exc_info=True
        )
        return None


# Placeholder Loading Function (kept internal to the agent for now)
def load_placeholder(language: str) -> str:
    """Dynamically loads placeholder code for a given language."""
    if language not in settings.LANGUAGE_DETAILS:
        raise ValueError(
            f"Language '{language}' not configured in settings.LANGUAGE_DETAILS."
        )
    module_name = settings.LANGUAGE_DETAILS[language]["placeholder_module"]
    placeholder_module_path = f"nondeterministic_agent.placeholders.{module_name}"
    try:
        module = importlib.import_module(placeholder_module_path)
        if hasattr(module, "placeholder_code"):
            logger.info(f"Successfully loaded placeholder for language: {language}")
            return module.placeholder_code
        else:
            raise AttributeError(
                f"Placeholder module '{placeholder_module_path}.py' must define 'placeholder_code' variable."
            )
    except ImportError:
        logger.error(
            f"Could not import placeholder module for '{language}' at '{placeholder_module_path}'"
        )
        raise
    except Exception as e:
        logger.error(f"Error loading placeholder for language '{language}': {str(e)}")
        raise


# --- JSON Handling Helpers (similar to YAML, useful for state/metrics) ---


def save_to_json(data: Any, file_path: str, indent: int = 2) -> bool:
    """
    Saves data to a JSON file, creating directories if necessary.

    Args:
        data (Any): The data to save (should be JSON serializable).
        file_path (str): The full path to the JSON file.
        indent (int): Indentation level for pretty printing.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.info(f"Data successfully saved to JSON file: {file_path}")
        return True
    except TypeError as e:
        logger.error(
            f"Type Error saving data to JSON {file_path} (check serializability): {e}"
        )
        return False
    except OSError as e:
        logger.error(f"OS Error saving JSON file {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(
            f"Unexpected error saving JSON file {file_path}: {e}", exc_info=True
        )
        return False


def load_from_json(file_path: str) -> Optional[Any]:
    """
    Loads data from a JSON file.

    Args:
        file_path (str): The full path to the JSON file.

    Returns:
        Optional[Any]: The loaded data, or None if an error occurs.
    """
    if not os.path.exists(file_path):
        logger.debug(f"JSON file not found: {file_path}")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(f"Data successfully loaded from JSON file: {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file {file_path}: {e}")
        return None
    except OSError as e:
        logger.error(f"OS Error loading JSON file {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error loading JSON file {file_path}: {e}", exc_info=True
        )
        return None
