import os
import logging
import yaml
from typing import Dict, Any, List, Optional

# Configuration and Utilities
from nondeterministic_agent.config.settings import settings  # Use central instance
from nondeterministic_agent.utils.string_utils import (
    sanitize_filename,
    extract_result_content,
)
from nondeterministic_agent.utils.file_utils import save_to_yaml, load_from_yaml
from nondeterministic_agent.utils.cli_helpers import (
    get_yes_no_input,
)  # For interactive part if needed

# Services
from nondeterministic_agent.services.llm_service import (
    LLMService,
)

# Prompts
from nondeterministic_agent.prompts.scope_planner_prompt import (
    non_deterministic_prompt as scope_planner_prompt,
)
from nondeterministic_agent.prompts.tasks_planner_prompt import (
    non_deterministic_prompt as tasks_planner_prompt,
)

logger = logging.getLogger(__name__)


class PlanningAgent:
    """
    Handles scope analysis and test plan generation based on scope definitions.
    Relies on an injected LLMService.
    """

    def __init__(self, llm_service: LLMService, service_settings: Optional[Any] = None):
        """
        Initializes the PlanningAgent.

        Args:
            llm_service: An instance of LLMService for interacting with the planning model.
            service_settings: The application settings instance. Defaults to global settings.
        """
        self.settings = service_settings or settings
        self.llm_service = llm_service
        # Use model name configured in the injected llm_service instance
        logger.info(
            f"Initialized PlanningAgent with LLMService using model: {self.llm_service.default_model}"
        )

        # Ensure base directories exist (using settings)
        os.makedirs(self.settings.SCOPE_DIR, exist_ok=True)
        os.makedirs(self.settings.PLAN_DIR, exist_ok=True)

    def _get_plan_test_for_subject(
        self,
        subject: str,
        task: str,
        scope_name: str,
        existing_test_cases: str = "",
    ) -> Optional[str]:
        """Generates test plan content string for a subject using the LLM."""
        logger.debug(
            f"Generating test plan for subject: '{subject}' in scope '{scope_name}'"
        )
        formatted_prompt = tasks_planner_prompt.format(
            TASK=task,
            SUBJECT=subject,
            EXISTING_TEST_CASES=existing_test_cases,
        )
        try:
            response = self.llm_service.run_single_completion(prompt=formatted_prompt)
            logger.debug(f"Raw tasks response for '{subject}':\n{response}")
            # Extract content expecting YAML/JSON structure
            extracted_content = extract_result_content(response)
            return extracted_content
        except Exception as e:
            logger.error(
                f"LLM service call failed during plan generation for subject '{subject}': {e}",
                exc_info=True,
            )
            return None

    def create_scope_analysis(
        self, scope_name: str, subject_prompt: str
    ) -> Optional[Dict[str, Any]]:
        """
        Creates and saves the scope analysis YAML file.

        Args:
            scope_name: The original name for the scope.
            subject_prompt: The detailed prompt describing the area to analyze.

        Returns:
            The saved scope data dictionary, or None on failure.
        """
        logger.info(f"Generating scope analysis for scope: '{scope_name}'")
        sanitized_scope_filename = sanitize_filename(scope_name)
        scope_file_path = os.path.join(
            self.settings.SCOPE_DIR, f"{sanitized_scope_filename}.yaml"
        )

        if os.path.exists(scope_file_path):
            logger.warning(
                f"Scope file already exists: {scope_file_path}. Overwriting."
            )

        formatted_prompt = scope_planner_prompt.format(SUBJECT=subject_prompt)
        scope_response = None
        try:
            scope_response = self.llm_service.run_single_completion(
                prompt=formatted_prompt
            )
            logger.debug(f"Raw scope response:\n{scope_response}")
        except Exception as e:
            logger.error(
                f"LLM service call failed during scope analysis for '{scope_name}': {e}",
                exc_info=True,
            )
            return None

        scope_content_str = extract_result_content(scope_response)
        if scope_content_str is None:
            logger.error(
                f"No content block found in scope response for '{scope_name}'. Cannot extract subjects. Response:\n{scope_response}"
            )
            return None

        try:
            # Assume extracted content is YAML
            extracted_data = yaml.safe_load(scope_content_str)
            if not isinstance(extracted_data, dict) or "subjects" not in extracted_data:
                raise ValueError(
                    "Extracted content is not a dictionary or missing 'subjects' key."
                )
            subjects = extracted_data.get("subjects", [])
            if not isinstance(subjects, list):
                raise ValueError("'subjects' key does not contain a list.")

            # Prepare data with original scope name
            scope_data_to_save = {
                "scope_name": scope_name,  # Use original name in data
                "prompt": subject_prompt,
                "subjects": subjects,
            }

            # Save using file utils
            if save_to_yaml(scope_data_to_save, scope_file_path):
                logger.info(
                    f"Scope analysis for '{scope_name}' saved to {scope_file_path}"
                )
                return scope_data_to_save
            else:
                # Error logged by save_to_yaml
                return None

        except (yaml.YAMLError, ValueError, TypeError) as e:
            logger.error(
                f"Failed to parse/validate extracted scope content for '{scope_name}': {e}. Content was:\n{scope_content_str}"
            )
            return None
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during scope analysis saving for '{scope_name}': {e}",
                exc_info=True,
            )
            return None

    def create_test_plans_for_scope(
        self, scope_name: str, scope_data: Dict[str, Any]
    ) -> int:
        """
        Creates test plan YAML files for each subject in the scope data.
        Handles resuming based on existing files.

        Args:
            scope_name: The original scope name.
            scope_data: The loaded scope analysis data dictionary.

        Returns:
            The number of test plans successfully generated or found existing.
        """
        subjects = scope_data.get("subjects", [])
        if not subjects:
            logger.warning(
                f"No subjects found in scope data for '{scope_name}'. Cannot generate plans."
            )
            return 0

        sanitized_scope_dirname = sanitize_filename(scope_name)
        scope_plan_dir = os.path.join(self.settings.PLAN_DIR, sanitized_scope_dirname)
        os.makedirs(scope_plan_dir, exist_ok=True)

        # --- Resume logic ---
        subject_to_sanitized = {s: sanitize_filename(s) for s in subjects}
        existing_files = {}
        try:
            if os.path.exists(scope_plan_dir):
                for filename in os.listdir(scope_plan_dir):
                    if filename.endswith(".yaml"):
                        basename = os.path.splitext(filename)[0]
                        existing_files[basename] = os.path.join(
                            scope_plan_dir, filename
                        )
        except OSError as e:
            logger.error(f"Error listing existing plan files in {scope_plan_dir}: {e}")
            # Continue assuming no existing files

        existing_plans = []
        pending_subjects = []
        for subject, sanitized_name in subject_to_sanitized.items():
            if sanitized_name in existing_files:
                existing_plans.append(subject)
            else:
                pending_subjects.append(subject)

        total_subjects = len(subjects)
        completed_subjects = len(existing_plans)
        remaining_subjects = len(pending_subjects)

        # --- Interactive Resume Prompt (Consider moving this to run_planning.py script) ---
        if completed_subjects > 0:
            print(
                f"\nFound {completed_subjects} existing test plans out of {total_subjects}."
            )
            if remaining_subjects > 0:
                print(f"Remaining subjects to process: {remaining_subjects}")
                if not get_yes_no_input(
                    f"Continue generating the remaining {remaining_subjects} test plans?",
                    default=True,
                ):
                    print("Exiting test plan generation.")
                    return completed_subjects
            else:
                print("All plans already exist.")
                return completed_subjects
        # --- End Interactive Resume Prompt ---

        logger.info(
            f"Creating test plans for {remaining_subjects} remaining subjects in scope '{scope_name}'."
        )
        successful_plans = completed_subjects
        total_to_process = remaining_subjects

        try:
            for i, subject in enumerate(pending_subjects):
                current_progress = completed_subjects + i + 1
                percent_done = (
                    (i + 1) / total_to_process * 100 if total_to_process > 0 else 100
                )
                logger.info(
                    f"[{percent_done:.1f}%] Generating plan {i + 1}/{total_to_process} for Subject: '{subject}' (Scope: '{scope_name}')"
                )
                print(
                    f"Processing Subject {i + 1}/{total_to_process}: '{subject}'..."
                )  # User feedback

                sanitized_subject_name = subject_to_sanitized[subject]
                subject_plan_file = os.path.join(
                    scope_plan_dir, f"{sanitized_subject_name}.yaml"
                )

                try:
                    # Use a generic task prompt for plan generation
                    task_description_prompt = f"Generate specific, actionable test cases focused on verifying potential non-deterministic behavior related to '{subject}' within the Linux Kernel {scope_name} context."
                    # Pass existing test cases might be useful if iteratively refining, but omitted for now
                    test_plan_content_str = self._get_plan_test_for_subject(
                        subject=subject,
                        task=task_description_prompt,
                        scope_name=scope_name,
                    )

                    if test_plan_content_str:
                        try:
                            test_plan_extracted_data = yaml.safe_load(
                                test_plan_content_str
                            )
                            # Validate structure
                            if (
                                not isinstance(test_plan_extracted_data, dict)
                                or "tests" not in test_plan_extracted_data
                            ):
                                raise ValueError(
                                    "Extracted content is not a dictionary or missing 'tests' key."
                                )
                            tests_list = test_plan_extracted_data.get("tests", [])
                            if not isinstance(tests_list, list):
                                raise ValueError("'tests' key does not contain a list.")
                            # Ensure basic structure within tests list if possible
                            valid_tests = []
                            for idx, test_item in enumerate(tests_list):
                                if (
                                    isinstance(test_item, dict)
                                    and "id" in test_item
                                    and "description" in test_item
                                ):
                                    valid_tests.append(test_item)
                                else:
                                    logger.warning(
                                        f"Invalid test item structure in plan for '{subject}', item {idx + 1}: {test_item}. Skipping item."
                                    )

                            # Prepare data with original scope/subject
                            subject_plan_data = {
                                "scope": scope_name,
                                "subject": subject,
                                "tests": valid_tests,
                            }
                            if save_to_yaml(subject_plan_data, subject_plan_file):
                                successful_plans += 1
                                logger.debug(
                                    f"Plan for '{subject}' saved to {subject_plan_file}"
                                )
                            else:
                                logger.error(
                                    f"Failed to save plan file for subject '{subject}'."
                                )

                        except (yaml.YAMLError, ValueError, TypeError) as e:
                            logger.warning(
                                f"Failed to parse/validate extracted plan YAML for '{subject}': {e}. Content:\n{test_plan_content_str}"
                            )
                    else:
                        logger.warning(
                            f"Failed to extract or generate plan content for subject: '{subject}'."
                        )

                except KeyboardInterrupt:
                    raise  # Re-raise to be caught by outer handler
                except Exception as e:
                    logger.error(
                        f"Error processing subject '{subject}': {str(e)}", exc_info=True
                    )

        except KeyboardInterrupt:
            logger.info(
                f"Plan generation interrupted by user. {successful_plans}/{total_subjects} plans saved."
            )
            print(
                f"\nTest plan generation interrupted. Progress: {successful_plans}/{total_subjects} completed."
            )

        if successful_plans == total_subjects:
            logger.info(f"Finished generating all plans for scope '{scope_name}'.")
        else:
            logger.warning(
                f"Partial completion: {successful_plans}/{total_subjects} plans generated for '{scope_name}'."
            )

        return successful_plans

    def load_scope_analysis(self, scope_name: str) -> Optional[Dict[str, Any]]:
        """Loads an existing scope analysis YAML file."""
        sanitized_scope = sanitize_filename(scope_name)
        scope_file = os.path.join(self.settings.SCOPE_DIR, f"{sanitized_scope}.yaml")
        scope_data = load_from_yaml(scope_file)
        if isinstance(scope_data, dict) and scope_data.get("scope_name") == scope_name:
            logger.info(f"Successfully loaded scope analysis for '{scope_name}'")
            return scope_data
        elif scope_data is not None:
            logger.warning(
                f"Loaded scope file {scope_file}, but scope_name mismatch or invalid format."
            )
            return None
        else:
            # File not found or load error (logged by load_from_yaml)
            return None

    def list_existing_scopes(self) -> List[str]:
        """Returns a list of existing scope names based on files in SCOPE_DIR."""
        scopes = []
        scope_dir = self.settings.SCOPE_DIR
        if not os.path.exists(scope_dir) or not os.path.isdir(scope_dir):
            return scopes  # Return empty list if dir doesn't exist

        try:
            for filename in os.listdir(scope_dir):
                if filename.endswith(".yaml"):
                    file_path = os.path.join(scope_dir, filename)
                    scope_data = load_from_yaml(file_path)
                    # Prefer original name from file content if possible
                    if isinstance(scope_data, dict) and "scope_name" in scope_data:
                        scopes.append(scope_data["scope_name"])
                    else:
                        # Fallback to filename without extension if load fails or format is wrong
                        scope_name_fallback = os.path.splitext(filename)[0]
                        # Try to "unsanitize" - this is imperfect, maybe just use filename?
                        # Or rely on user knowing the sanitized name matches? Let's use original if possible.
                        # If original name isn't found, maybe skip or log warning?
                        logger.warning(
                            f"Could not load original scope name from {filename}, file might be corrupt or old format."
                        )
                        # Option: Add sanitized name as fallback? scopes.append(scope_name_fallback)
        except OSError as e:
            logger.error(f"Error listing scope files in {scope_dir}: {e}")

        return sorted(list(set(scopes)))  # Ensure uniqueness and sort
