import re
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def extract_result_content(response_content: str) -> Optional[str]:
    """
    Extracts content between <result> tags (which may include a JSON code block),
    or from JSON code blocks in a response string.

    Args:
        response_content (str): The string containing possibly tagged content

    Returns:
        Optional[str]: The extracted content if found, None otherwise
    """
    if not isinstance(response_content, str):
        logger.warning(
            f"Non-string input received in extract_result_content: {type(response_content)}"
        )
        return None

    # Try to extract content from <result> tags
    result_tag_match = re.search(r"<result>(.*?)</result>", response_content, re.DOTALL)
    if result_tag_match:
        content = result_tag_match.group(1).strip()
        # Check if content is wrapped in a JSON code block inside the result tag
        json_block_within_result = re.search(
            r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE
        )
        if json_block_within_result:
            return json_block_within_result.group(1).strip()
        # Check if content is wrapped in YAML block
        yaml_block_within_result = re.search(
            r"```(?:yaml)?\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE
        )
        if yaml_block_within_result:
            return yaml_block_within_result.group(1).strip()

        # If not JSON/YAML block, return the raw content within <result>
        return content

    # Try to extract content from ```json blocks outside <result>
    json_block_match = re.search(
        r"```(?:json)?\s*(.*?)\s*```", response_content, re.DOTALL | re.IGNORECASE
    )
    if json_block_match:
        return json_block_match.group(1).strip()

    # Try to extract content from ```yaml blocks outside <result>
    yaml_block_match = re.search(
        r"```(?:yaml)?\s*(.*?)\s*```", response_content, re.DOTALL | re.IGNORECASE
    )
    if yaml_block_match:
        return yaml_block_match.group(1).strip()

    logger.debug(
        "No <result> tags or ```json/yaml``` blocks found in the response content."
    )
    # If no tags or blocks found, return None or maybe the original content if that's desired?
    # Returning None aligns with the function description.
    return None


def sanitize_filename(name: str) -> str:
    """
    Sanitizes a string to be used as a filename or directory name.
    """
    if not isinstance(name, str):
        logger.warning(
            f"Received non-string type '{type(name)}' in sanitize_filename. Converting to string."
        )
        name = str(name)

    # If there is a multiline then use only first non-empty line
    lines = name.splitlines()
    first_line = ""
    for line in lines:
        stripped_line = line.strip()
        if stripped_line:
            first_line = stripped_line
            break
    name = first_line

    if not name:
        logger.warning(
            "Sanitizing an empty or whitespace-only name. Returning 'invalid_name'."
        )
        return "invalid_name"

    # Remove potentially problematic path elements like '..'
    name = name.replace("..", "_")

    # Remove invalid characters often disallowed in filenames
    name = re.sub(r'[<>:"/\\|?*\'`]', "_", name)

    # Replace whitespace characters with underscores
    name = re.sub(r"\s+", "_", name)

    # Keep only alphanumeric, underscore, hyphen, and period (allowing extensions)
    # Removed the stricter filter to allow periods
    # name = re.sub(r"[^a-zA-Z0-9_\-.]", "", name)

    # Avoid names starting/ending with separators or periods
    name = name.strip("_-.")

    # Limit length (common filesystem limit is 255, be conservative)
    max_len = 100
    if len(name) > max_len:
        logger.debug(
            f"Sanitized name '{name}' exceeds max length {max_len}, truncating."
        )
        # Simple truncation for now, could add smarter boundary finding if needed
        name = name[:max_len]
        # Ensure it doesn't end with a separator after truncation
        name = name.strip("_-.")

    # Ensure the name is not empty after sanitization
    if not name:
        logger.warning(
            "Sanitization resulted in an empty name. Returning 'invalid_name'."
        )
        return "invalid_name"

    # Avoid reserved names in Windows (case-insensitive)
    reserved_names = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }
    name_base = name.split(".")[0]
    if name_base.upper() in reserved_names:
        logger.warning(
            f"Sanitized name '{name}' is a reserved Windows name. Appending underscore."
        )
        name += "_"

    return name
