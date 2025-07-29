import os
import logging
from typing import Dict, Any, Optional, List, Tuple

from nondeterministic_agent.utils.file_utils import save_to_json, load_from_json
from nondeterministic_agent.utils.string_utils import sanitize_filename

logger = logging.getLogger(__name__)


class MetricsManager:
    """
    Manages loading, accessing, updating, and saving execution metrics.
    Handles metrics persistence per subject within a scope/language.
    Provides specific methods for recording different test execution events.
    """

    def __init__(self, service_settings: Any, scope_name: str, language: str):
        """
        Initializes the MetricsManager for a specific scope and language.

        Args:
            service_settings: The application settings instance.
            scope_name: The name of the execution scope.
            language: The programming language being processed.
        """
        self.settings = service_settings
        self.scope_name = scope_name
        self.language = language
        self.metrics_dir_for_scope = os.path.join(
            self.settings.METRICS_DIR,
            sanitize_filename(scope_name),  # Use sanitized name for directory path
            self.language,
        )
        self.metrics: List[Dict[str, Any]] = []  # In-memory list of metric dicts
        logger.info(
            f"MetricsManager initialized for Scope='{scope_name}', Language='{language}'."
        )
        logger.info(f"Metrics directory: {self.metrics_dir_for_scope}")

    def _get_metrics_file_for_subject(self, subject: str) -> str:
        """Generates the full path for a subject's metrics JSON file."""
        if not subject:
            subject = "unknown_subject"  # Fallback subject name
        sanitized_subject = sanitize_filename(subject)
        return os.path.join(
            self.metrics_dir_for_scope, f"metrics_{sanitized_subject}.json"
        )

    def _ensure_default_structure(self, metric: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures a metric dictionary has all standard keys with default values."""
        metric.setdefault("test_id", "unknown")
        metric.setdefault("subject", "unknown")
        metric.setdefault("scope", self.scope_name)
        metric.setdefault("language", self.language)
        metric.setdefault("description", "N/A")
        metric.setdefault("overall_success", False)
        metric.setdefault("overall_status", "PENDING")
        metric.setdefault(
            "stage_attempts", {"precompile": 0, "compile": 0, "execute": 0}
        )
        # Added stage_actual_attempts
        metric.setdefault(
            "stage_actual_attempts", {"precompile": 0, "compile": 0, "execute": 0}
        )
        # 'api' field is less used now but kept for potential historical data
        metric.setdefault("api", {"precompile": {}, "compile": {}, "execute": {}})
        metric.setdefault(
            "generation", {"success": False, "duration": 0, "error": None}
        )
        metric.setdefault("non_deterministic", False)
        metric.setdefault("non_deterministic_detection_method", None)
        metric.setdefault("non_deterministic_detected_at_stage", None)
        metric.setdefault("system_error", False)
        metric.setdefault("timeout_error", False)
        metric.setdefault("failure_reason", None)
        # Added fix_error
        metric.setdefault("fix_error", None)
        metric.setdefault("total_duration", 0.0)
        # Added non_determinism_forcing_attempt
        metric.setdefault("non_determinism_forcing_attempt", False)
        # Added non_determinism_attempted
        metric.setdefault("non_determinism_attempted", 0)
        return metric

    def load_all_metrics(self) -> None:
        """
        Loads all existing metric files for the current scope and language
        into the in-memory list `self.metrics`. Ensures loaded metrics conform
        to the default structure.
        """
        self.metrics = []  # Clear existing in-memory metrics
        loaded_subjects_count = 0
        metrics_loaded_count = 0
        if not os.path.exists(self.metrics_dir_for_scope) or not os.path.isdir(
            self.metrics_dir_for_scope
        ):
            logger.info(
                f"Metrics directory not found, no metrics loaded: {self.metrics_dir_for_scope}"
            )
            return

        logger.info(f"Loading existing metrics from: {self.metrics_dir_for_scope}")
        unique_keys = set()
        duplicates_found = 0
        invalid_keys = 0

        for filename in os.listdir(self.metrics_dir_for_scope):
            if filename.startswith("metrics_") and filename.endswith(".json"):
                subject_file = os.path.join(self.metrics_dir_for_scope, filename)
                subject_metrics_data = load_from_json(subject_file)  # Use JSON loader

                if isinstance(subject_metrics_data, list):
                    loaded_subjects_count += 1
                    valid_in_file = 0
                    for metric in subject_metrics_data:
                        if not isinstance(metric, dict):
                            logger.warning(f"Skipping non-dict item in {filename}")
                            continue

                        # Filter strictly by original scope name and language
                        if (
                            metric.get("scope") != self.scope_name
                            or metric.get("language") != self.language
                        ):
                            continue

                        subject = metric.get("subject")
                        test_id = metric.get("test_id")

                        if not subject or not test_id:
                            logger.warning(
                                f"Skipping metric in {filename} missing 'subject' or 'test_id'."
                            )
                            invalid_keys += 1
                            continue

                        # Deduplication Check
                        metric_key = (self.scope_name, self.language, subject, test_id)
                        if metric_key in unique_keys:
                            duplicates_found += 1
                            logger.debug(
                                f"Duplicate metric found and skipped for {metric_key} in {filename}"
                            )
                            continue  # Skip adding duplicate
                        unique_keys.add(metric_key)

                        # --- Ensure default structure for loaded metrics ---
                        structured_metric = self._ensure_default_structure(metric)
                        self.metrics.append(structured_metric)
                        valid_in_file += 1
                        metrics_loaded_count += 1
                    if valid_in_file == 0:
                        logger.debug(
                            f"No valid metrics for {self.scope_name}/{self.language} found in {filename}."
                        )

                elif (
                    subject_metrics_data is not None
                ):  # Loaded something, but not a list
                    logger.warning(
                        f"Metrics file {filename} did not contain a list. Skipping."
                    )

        log_level = (
            logger.warning
            if (duplicates_found > 0 or invalid_keys > 0)
            else logger.info
        )
        log_level(
            f"Metrics loading complete for {self.scope_name}/{self.language}: "
            f"Loaded {metrics_loaded_count} metrics from {loaded_subjects_count} files. "
            f"Duplicates skipped: {duplicates_found}. Invalid keys skipped: {invalid_keys}."
        )

    def find_metric(
        self, test_id: str, subject: str
    ) -> Optional[Tuple[Dict[str, Any], int]]:
        """
        Finds a specific metric dictionary within the managed `self.metrics` list.
        Ensures the returned metric conforms to the default structure.

        Args:
            test_id: The ID of the test.
            subject: The subject of the test.

        Returns:
            A tuple containing (metric_dictionary, index) if found, otherwise None.
            The dictionary returned is a reference to the one in the list.
        """
        if not test_id or not subject:
            logger.warning("find_metric called with missing test_id or subject.")
            return None

        for index, metric in enumerate(self.metrics):
            # Match criteria: MUST match scope, language, test_id, subject.
            if (
                metric.get("scope") == self.scope_name
                and metric.get("language") == self.language
                and metric.get("test_id") == test_id
                and metric.get("subject") == subject
            ):
                # Ensure structure just in case before returning
                # Modifying in place if structure changed is essential here
                structured_metric = self._ensure_default_structure(metric)
                if (
                    structured_metric is not metric
                ):  # Check if the dictionary object itself changed (e.g. if a copy was made)
                    self.metrics[index] = (
                        structured_metric  # Ensure the list holds the structured version
                    )

                return self.metrics[
                    index
                ], index  # Return the (potentially updated) metric from the list
        return None  # Not found

    def update_or_append_metric(self, metric_to_update: Dict[str, Any]) -> None:
        """
        Updates an existing metric in the `self.metrics` list or appends it if new.
        Ensures the metric conforms to the current scope, language, and default structure.
        This is the primary method for adding/modifying metrics in the list.

        Args:
            metric_to_update: The metric dictionary with potentially new data.
        """
        if not isinstance(metric_to_update, dict):
            logger.error(
                f"Attempted metric update with non-dict: {type(metric_to_update)}"
            )
            return

        test_id = metric_to_update.get("test_id")
        subject = metric_to_update.get("subject")

        if not test_id or not subject:
            logger.error(
                f"Metric update failed: Missing test_id/subject. Data: {metric_to_update}"
            )
            return

        # --- Enforce Scope and Language Consistency ---
        metric_to_update["scope"] = self.scope_name
        metric_to_update["language"] = self.language

        # Ensure the incoming data has the default structure before searching/appending
        structured_metric_update = self._ensure_default_structure(
            metric_to_update.copy()
        )  # Work on a copy to avoid side effects

        existing_metric_info = self.find_metric(test_id, subject)

        if existing_metric_info:
            # Found existing metric, update it in place
            # Note: find_metric already returned a reference to the dict in the list
            existing_metric, index = existing_metric_info
            existing_metric.update(
                structured_metric_update
            )  # Update the dict in the list
            logger.debug(f"Updated metric for {test_id} ({subject}) at index {index}")
        else:
            # Append the structured copy
            self.metrics.append(structured_metric_update)
            logger.debug(f"Appended new metric for {test_id} ({subject})")

    def create_default_metric(
        self, test_id: str, subject: str, description: str = "N/A"
    ) -> Dict[str, Any]:
        """
        Creates a new metric dictionary with default values for this instance's
        scope and language, ensuring it conforms to the structure.

        Args:
            test_id: The ID of the test.
            subject: The subject of the test.
            description: The description of the test.

        Returns:
            A dictionary representing the default metric structure.
        """
        default_metric = {
            "test_id": test_id,
            "subject": subject,
            "scope": self.scope_name,
            "language": self.language,
            "description": description,
            # Initialize other fields via _ensure_default_structure
        }
        return self._ensure_default_structure(default_metric)

    # --- Specific Event Recording Methods ---

    def record_generation_result(
        self,
        test_id: str,
        subject: str,
        success: bool,
        duration: float,
        error: Optional[str],
    ):
        """Records the outcome of the code generation step."""
        metric_info = self.find_metric(test_id, subject)
        if not metric_info:
            logger.error(
                f"Cannot record generation result: Metric not found for {test_id} ({subject})."
            )
            return
        metric, _ = metric_info
        # Ensure 'generation' key exists (should be handled by find_metric -> _ensure_default_structure)
        metric.setdefault(
            "generation", {"success": False, "duration": 0, "error": None}
        )
        metric["generation"]["success"] = success
        metric["generation"]["duration"] = duration
        metric["generation"]["error"] = error
        # No need to call update_or_append_metric, modification is in place
        logger.debug(
            f"Recorded generation result for {test_id} ({subject}): Success={success}"
        )

    def record_fix_attempt(self, test_id: str, subject: str, stage: str):
        """Increments the fix attempt counter for a specific stage."""
        metric_info = self.find_metric(test_id, subject)
        if not metric_info:
            logger.error(
                f"Cannot record fix attempt: Metric not found for {test_id} ({subject})."
            )
            return
        metric, _ = metric_info
        # Ensure stage key exists
        if stage not in metric.get("stage_attempts", {}):
            metric.setdefault("stage_attempts", {})[stage] = 0  # Initialize if missing
            logger.warning(
                f"Initialized missing stage '{stage}' in stage_attempts for {test_id}"
            )
        metric["stage_attempts"][stage] = metric["stage_attempts"].get(stage, 0) + 1
        logger.debug(
            f"Recorded fix attempt #{metric['stage_attempts'][stage]} for stage '{stage}' on {test_id} ({subject})"
        )

    def record_actual_stage_attempt(self, test_id: str, subject: str, stage: str):
        """Increments the *actual* run attempt counter for a specific stage."""
        metric_info = self.find_metric(test_id, subject)
        if not metric_info:
            logger.error(
                f"Cannot record actual stage attempt: Metric not found for {test_id} ({subject})."
            )
            return
        metric, _ = metric_info
        # Ensure stage key exists
        if stage not in metric.get("stage_actual_attempts", {}):
            metric.setdefault("stage_actual_attempts", {})[stage] = (
                0  # Initialize if missing
            )
            logger.warning(
                f"Initialized missing stage '{stage}' in stage_actual_attempts for {test_id}"
            )
        metric["stage_actual_attempts"][stage] = (
            metric["stage_actual_attempts"].get(stage, 0) + 1
        )
        logger.debug(
            f"Recorded actual run attempt #{metric['stage_actual_attempts'][stage]} for stage '{stage}' on {test_id} ({subject})"
        )

    def record_stage_success(self, test_id: str, subject: str, stage: str):
        """Resets the fix attempt counter for a stage upon successful completion."""
        metric_info = self.find_metric(test_id, subject)
        if not metric_info:
            logger.error(
                f"Cannot record stage success: Metric not found for {test_id} ({subject})."
            )
            return
        metric, _ = metric_info
        if stage in metric.get("stage_attempts", {}):
            metric["stage_attempts"][stage] = 0
            logger.debug(
                f"Reset fix attempts for successful stage '{stage}' on {test_id} ({subject})"
            )
        else:
            logger.warning(
                f"Attempted to reset attempts for non-existent stage '{stage}' in metric for {test_id}"
            )

    def record_fix_result(
        self, test_id: str, subject: str, fix_successful: bool, error_msg: Optional[str]
    ):
        """Records the outcome of an AI fix attempt."""
        metric_info = self.find_metric(test_id, subject)
        if not metric_info:
            logger.error(
                f"Cannot record fix result: Metric not found for {test_id} ({subject})."
            )
            return
        metric, _ = metric_info
        metric["fix_error"] = None if fix_successful else error_msg
        logger.debug(
            f"Recorded fix result for {test_id} ({subject}): Success={fix_successful}"
        )

    def record_non_det_force_attempt(self, test_id: str, subject: str):
        """Records that an attempt was made to force non-determinism."""
        metric_info = self.find_metric(test_id, subject)
        if not metric_info:
            logger.error(
                f"Cannot record non-det force attempt: Metric not found for {test_id} ({subject})."
            )
            return
        metric, _ = metric_info
        metric["non_determinism_forcing_attempt"] = True
        # Ensure key exists before incrementing
        metric["non_determinism_attempted"] = (
            metric.get("non_determinism_attempted", 0) + 1
        )
        logger.debug(
            f"Recorded non-determinism forcing attempt #{metric['non_determinism_attempted']} for {test_id} ({subject})"
        )

    def record_test_completion(
        self,
        test_id: str,
        subject: str,
        status: str,
        success: bool,
        reason: Optional[str],
        non_det: bool,
        non_det_method: Optional[str],
        non_det_stage: Optional[str],
        sys_error: bool,
        timeout: bool,
        duration: float,
    ):
        """Records the final completion status and details of a test."""
        metric_info = self.find_metric(test_id, subject)
        if not metric_info:
            logger.error(
                f"Cannot record test completion: Metric not found for {test_id} ({subject}). Creating default."
            )
            # If completion MUST be recorded, create a default metric
            metric = self.create_default_metric(
                test_id, subject, description="Completion Recorded - Original Not Found"
            )
            self.update_or_append_metric(metric)  # Add it to the list
            # Now try finding it again to proceed
            metric_info = self.find_metric(test_id, subject)
            if not metric_info:
                logger.critical(
                    f"Failed to create/find metric for completion recording: {test_id} ({subject}). Data lost."
                )
                return  # Abort if still not found

        metric, _ = metric_info

        metric["overall_status"] = status
        metric["overall_success"] = success
        metric["failure_reason"] = reason
        metric["non_deterministic"] = non_det
        metric["non_deterministic_detection_method"] = non_det_method
        metric["non_deterministic_detected_at_stage"] = non_det_stage
        metric["system_error"] = sys_error
        metric["timeout_error"] = timeout
        metric["total_duration"] = duration

        logger.debug(
            f"Recorded test completion for {test_id} ({subject}): Status={status}, Success={success}"
        )

    # --- Saving and Loading ---

    def save_metrics_for_subject(self, subject: str) -> bool:
        """
        Saves all in-memory metrics belonging to a specific subject for
        the current scope/language to its JSON file.

        Args:
            subject: The subject whose metrics should be saved.

        Returns:
            True if save was attempted (even if no metrics found), False on error.
        """
        if not subject:
            logger.warning("Attempted to save metrics for empty subject.")
            return False

        subject_metrics_to_save = [
            m
            for m in self.metrics
            if m.get("subject") == subject
            and m.get("scope")
            == self.scope_name  # Ensure scope/lang match, though should be inherent
            and m.get("language") == self.language
        ]

        if not subject_metrics_to_save:
            logger.debug(
                f"No in-memory metrics found for subject '{subject}' ({self.scope_name}/{self.language}). Skipping save."
            )
            return True  # Nothing to save is not an error

        subject_file = self._get_metrics_file_for_subject(subject)
        logger.debug(
            f"Saving {len(subject_metrics_to_save)} metrics for subject '{subject}' to {subject_file}"
        )

        # Use file_utils saver
        if save_to_json(subject_metrics_to_save, subject_file):
            return True
        else:
            # Error logged by save_to_json
            return False

    def save_all_metrics(self) -> None:
        """
        Saves metrics for ALL subjects currently held in memory for this
        scope/language to their respective files.
        """
        subjects_in_memory = set(
            m.get("subject")
            for m in self.metrics
            if m.get("subject")
            and m.get("scope") == self.scope_name  # Ensure scope/lang match
            and m.get("language") == self.language
        )

        if not subjects_in_memory:
            logger.info(
                f"No subjects found with metrics in memory for {self.scope_name}/{self.language}. Nothing to save."
            )
            return

        logger.info(
            f"Saving all metrics for {len(subjects_in_memory)} subjects under {self.scope_name}/{self.language}..."
        )
        saved_count = 0
        failed_count = 0
        # Ensure directory exists before saving individuals
        os.makedirs(self.metrics_dir_for_scope, exist_ok=True)
        for subject in subjects_in_memory:
            if self.save_metrics_for_subject(subject):
                saved_count += 1
            else:
                failed_count += 1

        log_func = logger.warning if failed_count > 0 else logger.info
        log_func(
            f"Finished saving metrics. Subjects processed: {len(subjects_in_memory)}. Files attempted/updated: {saved_count}. Failures: {failed_count}."
        )

    def reset_metrics_for_run(self) -> int:
        """
        Resets runtime status fields and attempt counters for all metrics currently
        loaded in memory for the instance's scope and language.
        This is typically used for a 'force_restart'. It modifies metrics IN PLACE.

        Returns:
            The number of metrics that were reset.
        """
        reset_count = 0
        logger.warning(
            f"Resetting runtime status and attempts for all loaded metrics ({len(self.metrics)}) for {self.scope_name}/{self.language}..."
        )
        for metric in self.metrics:
            # Scope/Language check is implicit as self.metrics is already filtered by __init__ and load_all_metrics
            # Ensure structure before resetting
            structured_metric = self._ensure_default_structure(metric)
            if (
                structured_metric is not metric
            ):  # Check if the dictionary object itself changed
                # Find index and update list reference if needed (shouldn't happen if find_metric works)
                try:
                    idx = self.metrics.index(metric)
                    self.metrics[idx] = structured_metric
                    metric = structured_metric  # Use the structured version
                except ValueError:
                    logger.error(
                        f"Metric vanished during reset structure check: {metric.get('test_id')}. Skipping reset."
                    )
                    continue

            metric["stage_attempts"] = {"precompile": 0, "compile": 0, "execute": 0}
            metric["stage_actual_attempts"] = {
                "precompile": 0,
                "compile": 0,
                "execute": 0,
            }
            metric["overall_status"] = "PENDING"
            metric["overall_success"] = False
            metric["failure_reason"] = None
            metric["fix_error"] = None
            metric["non_deterministic"] = False
            metric["non_deterministic_detection_method"] = None
            metric["non_deterministic_detected_at_stage"] = None
            metric["system_error"] = False
            metric["timeout_error"] = False
            metric["total_duration"] = 0.0
            metric["non_determinism_forcing_attempt"] = False
            # Do NOT reset non_determinism_attempted - this counts historical attempts across runs
            # Do NOT reset generation status here - reset happens in process_test
            reset_count += 1

        logger.info(
            f"Reset {reset_count} metrics for {self.scope_name}/{self.language}."
        )
        return reset_count
