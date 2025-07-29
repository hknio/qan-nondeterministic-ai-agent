import os
import logging
from typing import List, Optional
import html

# Imports for Interactivity
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.validation import Validator, ValidationError
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import PromptSession

# Configuration
from nondeterministic_agent.config.settings import settings

logger = logging.getLogger(__name__)


# --- Validators ---
class NumberValidator(Validator):
    """Validator for prompt_toolkit ensuring numeric input within a specified range."""

    def __init__(self, min_val: int, max_val: int):
        self.min_val = min_val
        self.max_val = max_val

    def validate(self, document):
        text = document.text
        if not text:
            # Allow empty input, which might trigger default value handling.
            return
        try:
            value = int(text)
            if not (self.min_val <= value <= self.max_val):
                raise ValidationError(
                    message=f"Please enter a number between {self.min_val} and {self.max_val}",
                    cursor_position=len(text),
                )
        except ValueError:
            raise ValidationError(
                message="Please enter a valid number", cursor_position=len(text)
            )


# --- Input Helpers ---


async def get_single_choice(
    prompt_text: str, choices: List[str], default: Optional[int] = None
) -> Optional[str]:
    """
    (Async) Presents a list of choices and gets a single selection from the user
    using prompt_toolkit, integrated with an existing asyncio loop.
    """
    if not choices:
        print("\nError: No choices provided for selection.")
        logger.error("get_single_choice called with no choices.")
        return None

    try:
        os.makedirs(settings.HISTORY_DIR, exist_ok=True)
        history_file = os.path.join(settings.HISTORY_DIR, "single_choice_history")
    except OSError as e:
        logger.warning(
            f"Could not create/access history directory {settings.HISTORY_DIR}: {e}. History disabled."
        )
        history_file = None

    choice_map = {str(i + 1): choice for i, choice in enumerate(choices)}
    choice_completer = WordCompleter(list(choice_map.keys()))
    validator = NumberValidator(1, len(choices))

    # --- FIX: Escape dynamic text ---
    escaped_prompt_text = html.escape(prompt_text)
    options_text = ""
    for i, choice in enumerate(choices, 1):
        options_text += (
            f"<option>{i}. {html.escape(choice)}</option>\n"  # Escape each choice
        )

    style = Style.from_dict(
        {
            "question": "ansicyan bold",
            "option": "ansigreen",
            "highlight": "#ansiyellow",
            "default": "ansigray italic",
        }
    )

    default_str = (
        str(default + 1) if default is not None and 0 <= default < len(choices) else ""
    )
    default_display = ""
    if default_str:
        # --- FIX: Escape the default choice text ---
        escaped_default_choice = html.escape(choices[default])
        default_display = (
            f" <default>(default is {default_str}. {escaped_default_choice})</default>"
        )

    # --- FIX: Build the full HTML string before calling HTML() ---
    full_html_string = (
        f"<question>{escaped_prompt_text}</question>\n"
        f"{options_text}"
        f"Enter your choice (1-{len(choices)}){default_display}: "
    )
    formatted_prompt = HTML(full_html_string)

    session = PromptSession(
        history=FileHistory(history_file) if history_file else None,
        auto_suggest=AutoSuggestFromHistory(),
        style=style,
        completer=choice_completer,
        validator=validator,
        validate_while_typing=False,
    )

    try:
        result = await session.prompt_async(formatted_prompt, default=default_str)

        if result is None:
            print("\nSelection cancelled.")
            return choices[default] if default is not None else None

        if not result.strip() and default_str:
            return choices[default]

        return choice_map.get(result)

    except (KeyboardInterrupt, EOFError):
        print("\nSelection cancelled.")
        return choices[default] if default is not None else None
    except Exception as e:
        logger.error(
            f"Error during async prompt_toolkit interaction: {e}", exc_info=True
        )
        print(f"\nAn error occurred during input: {e}")
        return None


async def get_yes_no_input(
    prompt_text: str, default: Optional[bool] = None
) -> Optional[bool]:
    """(Async) Get a yes/no response from the user using prompt_toolkit."""
    try:
        os.makedirs(settings.HISTORY_DIR, exist_ok=True)
        history_file = os.path.join(settings.HISTORY_DIR, "yes_no_history")
    except OSError as e:
        logger.warning(
            f"Could not create/access history directory {settings.HISTORY_DIR}: {e}. History disabled."
        )
        history_file = None

    yes_no_completer = WordCompleter(["y", "yes", "n", "no"], ignore_case=True)
    default_text = ""
    default_value = ""
    if default is not None:
        if default:
            default_text, default_value = " <default>(Y/n)</default>", "y"
        else:
            default_text, default_value = " <default>(y/N)</default>", "n"

    style = Style.from_dict({"question": "ansicyan bold", "default": "ansigray italic"})

    # --- FIX: Escape prompt_text and build full string ---
    escaped_prompt_text = html.escape(prompt_text)
    full_html_string = f"<question>{escaped_prompt_text}</question>{default_text}: "
    formatted_prompt = HTML(full_html_string)

    session = PromptSession(
        history=FileHistory(history_file) if history_file else None,
        auto_suggest=AutoSuggestFromHistory(),
        style=style,
        completer=yes_no_completer,
        validator=Validator.from_callable(
            lambda text: text.strip().lower() in ("y", "yes", "n", "no", ""),
            error_message="Please enter y/yes or n/no",
            move_cursor_to_end=True,
        ),
        validate_while_typing=False,
    )

    try:
        result = await session.prompt_async(formatted_prompt, default=default_value)

        if result is None:
            print("\nInput cancelled.")
            return default

        response = result.strip().lower()

        if not response and default is not None:
            return default
        if response in ("y", "yes"):
            return True
        if response in ("n", "no"):
            return False

        return default

    except (KeyboardInterrupt, EOFError):
        print("\nInput cancelled")
        return default
    except Exception as e:
        logger.error(
            f"Error during async prompt_toolkit interaction: {e}", exc_info=True
        )
        print(f"\nAn error occurred during input: {e}")
        return None


async def get_multiline_input(
    prompt_text: str, default_value: Optional[str] = None
) -> Optional[str]:
    """(Async) Get multiline input from the user using prompt_toolkit."""
    try:
        os.makedirs(settings.HISTORY_DIR, exist_ok=True)
        history_file = os.path.join(settings.HISTORY_DIR, "multiline_input_history")
    except OSError as e:
        logger.warning(
            f"Could not create/access history directory {settings.HISTORY_DIR}: {e}. History disabled."
        )
        history_file = None

    kb = KeyBindings()

    @kb.add("escape", "enter")
    @kb.add("c-j")
    def _(event):
        event.app.exit(result=event.app.current_buffer.text)

    @kb.add("c-c")
    @kb.add("c-d")
    def _(event):
        event.app.exit(result=None)

    style = Style.from_dict(
        {
            "prompt": "ansicyan bold",
            "default": "ansigray italic",
            "instruction": "ansigreen",
            "key": "#ansiyellow",
        }
    )

    # --- FIX: Escape dynamic text and build full string ---
    escaped_prompt_text = html.escape(prompt_text)
    instruction_part = "<instruction>(Press <key>ESC</key>+<key>Enter</key> or <key>Ctrl+J</key> to submit, <key>Ctrl+C</key>/<key>Ctrl+D</key> to cancel)</instruction>"
    final_html_string = f"<prompt>{escaped_prompt_text}</prompt> {instruction_part}"

    if default_value:
        final_html_string += f"\n<default>(Default provided. Edit or submit empty to use default.)</default>\n"
    else:
        final_html_string += "\n"

    formatted_prompt = HTML(final_html_string)
    # --- End Fix ---

    session = PromptSession(
        history=FileHistory(history_file) if history_file else None,
        auto_suggest=AutoSuggestFromHistory(),
        style=style,
        key_bindings=kb,
        multiline=True,
        wrap_lines=True,
    )

    try:
        result = await session.prompt_async(
            formatted_prompt, default=default_value or ""
        )

        if result is None:
            print("\nInput cancelled.")
            return None

        if result == "" and default_value:
            return default_value
        return result

    except (KeyboardInterrupt, EOFError):
        print("\nInput cancelled")
        return None
    except Exception as e:
        logger.error(
            f"Error during async prompt_toolkit interaction: {e}", exc_info=True
        )
        print(f"\nAn error occurred during input: {e}")
        return None


# --- Scope Listing (remains synchronous) ---
def list_available_scopes() -> List[str]:
    """
    Lists available scopes by looking for directories containing .yaml files
    within the PLAN_DIR specified in settings. (This is synchronous I/O).
    """
    scopes = []
    plan_dir = settings.PLAN_DIR
    if not os.path.exists(plan_dir) or not os.path.isdir(plan_dir):
        logger.warning(f"Plan directory not found or is not a directory: {plan_dir}")
        # Avoid printing here, let the caller handle user feedback
        return scopes
    try:
        for item_name in os.listdir(plan_dir):
            item_path = os.path.join(plan_dir, item_name)
            if os.path.isdir(item_path):
                try:
                    if any(f.endswith(".yaml") for f in os.listdir(item_path)):
                        scopes.append(item_name)
                    else:
                        logger.debug(
                            f"Skipping directory '{item_name}' in {plan_dir} as it contains no YAML files."
                        )
                except OSError as inner_e:
                    logger.warning(
                        f"Could not read contents of directory '{item_path}': {inner_e}"
                    )
    except OSError as e:
        logger.error(f"Error listing scopes in {plan_dir}: {e}")
        # Avoid printing here, let the caller handle user feedback
    return sorted(scopes)
