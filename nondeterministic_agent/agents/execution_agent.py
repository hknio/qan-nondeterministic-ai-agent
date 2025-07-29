import asyncio
import os
import time
import logging
import sys
from typing import Dict, Any, Optional, List, Tuple

# Configuration and Utilities
from nondeterministic_agent.config.settings import settings
from nondeterministic_agent.utils.file_utils import load_placeholder, save_to_json
from nondeterministic_agent.utils.string_utils import sanitize_filename

# Services and Managers (Dependencies)
from nondeterministic_agent.services.qan_service import QANService
from nondeterministic_agent.services.aider_service import AiderService
from nondeterministic_agent.managers.metrics_manager import MetricsManager
from nondeterministic_agent.managers.state_manager import StateManager

# Aider specific imports needed for type hinting and managing coders
from aider.coders import Coder
from qan import ExecutionResult  # For type hinting

logger = logging.getLogger(__name__)


class ExecutionAgent:
    """
    Orchestrates the process of generating code for test plans, executing it via QAN,
    attempting fixes on errors, attempting to force non-determinism, and tracking results.
    Relies on injected services (QAN, Aider) and managers (Metrics, State).
    """

    def __init__(
        self,
        scope_name: str,
        language: str,
        qan_service: QANService,
        aider_service: AiderService,
        metrics_manager: MetricsManager,
        state_manager: StateManager,
        force_restart: bool = False,
        specific_subject: Optional[str] = None,
    ):
        # --- Input Validation ---
        if not scope_name:
            raise ValueError("Scope name cannot be empty.")
        if not language:
            raise ValueError("Language cannot be empty.")

        self.scope_name = scope_name
        self.language = language
        self.force_restart = force_restart
        self.specific_subject = specific_subject
        self.sanitized_scope_name = sanitize_filename(scope_name)
        self.sanitized_specific_subject = (
            sanitize_filename(specific_subject) if specific_subject else None
        )

        # --- Dependency Injection ---
        self.settings = settings
        self.qan_service = qan_service
        self.aider_service = aider_service
        self.metrics_manager = metrics_manager
        self.state_manager = state_manager

        # --- Language Validation and Setup ---
        if language not in self.settings.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: '{language}'. Supported: {self.settings.SUPPORTED_LANGUAGES}"
            )
        if language not in self.settings.LANGUAGE_DETAILS:
            raise ValueError(
                f"Language '{language}' missing config in settings.LANGUAGE_DETAILS."
            )
        self.language_details = self.settings.LANGUAGE_DETAILS[language]
        self.file_extension = self.language_details["extension"]

        logger.info(
            f"Initializing ExecutionAgent for Scope='{self.scope_name}', Language='{self.language}'"
        )
        if specific_subject:
            logger.info(
                f"Targeting specific subject: '{self.specific_subject}' (Sanitized: '{self.sanitized_specific_subject}')"
            )

        # --- Directory Setup ---
        self.plan_dir_for_scope = os.path.join(
            self.settings.PLAN_DIR, self.sanitized_scope_name
        )
        self.tests_dir_for_scope = os.path.join(
            self.settings.TESTS_DIR, self.sanitized_scope_name
        )
        self.api_results_dir_for_scope = os.path.join(
            self.settings.API_RESULTS_DIR, self.sanitized_scope_name, self.language
        )
        # Metrics and State directories are managed by their respective managers

        os.makedirs(self.tests_dir_for_scope, exist_ok=True)
        os.makedirs(self.api_results_dir_for_scope, exist_ok=True)

        # --- Load Placeholder Code ---
        try:
            self.placeholder_code = load_placeholder(self.language)
        except Exception as e:
            logger.critical(
                f"Failed to load placeholder code for {self.language}. Cannot proceed. Error: {e}",
                exc_info=True,
            )
            sys.exit(1)  # Critical failure

        # --- State and Metrics Initialization ---
        self.coders: Dict[str, Coder] = {}  # Manages Aider Coder instances per test_id

        if not self.force_restart:
            self.state_manager.load_state()
        # Always load metrics, reset happens based on force_restart inside manager
        self.metrics_manager.load_all_metrics()
        if self.force_restart:
            logger.info(
                f"Force restart enabled: Resetting runtime metrics for {self.scope_name}/{self.language}."
            )
            self.metrics_manager.reset_metrics_for_run()

        # Log attempt limits from settings
        logger.info(
            f"Max Fix Attempts per stage: Pre={self.settings.MAX_PRECOMPILE_ATTEMPTS}, Comp={self.settings.MAX_COMPILE_ATTEMPTS}, Exec={self.settings.MAX_EXECUTE_ATTEMPTS}"
        )
        logger.info(
            f"Max Actual Pipeline Attempts: {self.settings.MAX_ACTUAL_PIPELINE_ATTEMPTS}"  # Log new limit
        )
        logger.info(
            f"Non-Determinism Forcing Attempts: {self.settings.NON_DET_ATTEMPTS}"
        )

    # --- Helper Methods ---

    def _get_test_id_from_path(self, file_path: str) -> str:
        """Helper to extract the test ID (filename without extension)."""
        return os.path.splitext(os.path.basename(file_path))[0]

    # --- Core Action Methods ---

    async def _generate_code(
        self, subject: str, test: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Generates code for a test, updates metrics, and manages the coder instance.
        Relies on MetricsManager for persistence. Now calls AiderService asynchronously.

        Returns:
            A tuple: (output_file_path or None, metric_object_reference or None)
        """
        test_id = test.get("id", "unknown_test_id")
        sanitized_subject = sanitize_filename(subject)
        description = test.get("description", "No description")

        # Get or create metric using the manager
        metric_info = self.metrics_manager.find_metric(test_id, subject)
        if metric_info:
            metric = metric_info[0]
            # Reset generation status if not force_restart
            if not self.force_restart:
                metric["generation"] = {"success": False, "duration": 0, "error": None}
            metric["description"] = description  # Update description
        else:
            metric = self.metrics_manager.create_default_metric(
                test_id, subject, description
            )
            # Important: Add the newly created metric to the manager's list
            self.metrics_manager.update_or_append_metric(metric)
            # Re-find it to ensure we have the managed reference
            metric_info = self.metrics_manager.find_metric(test_id, subject)
            if not metric_info:
                logger.critical(
                    f"Metric lost immediately after creation for {test_id}. Aborting generation."
                )
                # Cannot proceed without a managed metric
                return None, None
            metric = metric_info[0]

        start_time = time.time()
        generation_success = False
        generation_error = None
        output_file = None
        coder_instance = None

        try:
            # Output Path
            test_output_dir = os.path.join(
                self.tests_dir_for_scope, sanitized_subject, self.language
            )
            os.makedirs(test_output_dir, exist_ok=True)
            output_file = os.path.join(
                test_output_dir, f"{test_id}{self.file_extension}"
            )

            coder_instance = await self.aider_service.generate_initial_code(
                language=self.language,
                placeholder_code=self.placeholder_code,
                output_file=output_file,
                test_id=test_id,
                subject=subject,
                description=description,
            )

            if coder_instance:
                generation_success = True
                self.coders[test_id] = coder_instance  # Store the coder instance
                logger.info(f"Code generated successfully: {output_file}")
            else:
                generation_error = (
                    f"Code generation failed (AiderService returned None)"
                )
                logger.error(f"Error generating code for {test_id}: {generation_error}")
                output_file = None  # Indicate failure
                if test_id in self.coders:
                    del self.coders[test_id]  # Clean up coder ref if failed

        except Exception as e:
            generation_error = f"Code generation failed: {str(e)}"
            logger.error(
                f"Error generating code for {test_id}: {generation_error}",
                exc_info=True,
            )
            output_file = None
            if test_id in self.coders:
                del self.coders[test_id]

        finally:
            # Update metrics via manager regardless of success/failure
            duration = time.time() - start_time
            self.metrics_manager.record_generation_result(
                test_id, subject, generation_success, duration, generation_error
            )
            # The metric object itself is updated in place by the manager method

        return output_file, metric  # Return the managed metric object reference

    async def _fix_code(
        self,
        code_file_path: str,
        metric: Dict[str, Any],  # Pass metric ref for info
        stage: str,
        error_details: str,
        stdout: Optional[str],
        stderr: Optional[str],
    ) -> bool:
        """
        Attempts to fix code using AiderService. Updates coder instance and metric via manager.
        Now directly awaits the async AiderService method.

        Args:
            code_file_path: Path to the code file.
            metric: The metric object reference for this test.
            stage: The stage where the error occurred (e.g., 'precompile', 'execute_timeout').
            error_details: The error message or details from the failed stage.
            stdout: Optional stdout from the failed execution.
            stderr: Optional stderr from the failed execution or compilation.

        Returns:
            bool: True if the Aider fix attempt ran successfully, False otherwise.
                  Note: This indicates the *fixer ran*, not that the code is *actually fixed*.
        """
        test_id = metric["test_id"]
        subject = metric["subject"]
        scope = metric["scope"]
        description = metric["description"]
        language = metric["language"]

        if test_id not in self.coders:
            err_msg = "Fix attempt failed: Coder instance missing"
            logger.error(
                f"Cannot attempt fix for {test_id}: No existing coder instance found."
            )
            self.metrics_manager.record_fix_result(test_id, subject, False, err_msg)
            return False

        existing_coder = self.coders[test_id]

        logger.info(
            f"Attempting AI fix for {stage} error in {test_id} (Sub: {subject}, Scope: {scope})..."
        )

        updated_coder = await self.aider_service.fix_code(
            existing_coder=existing_coder,
            code_file_path=code_file_path,
            test_id=test_id,
            language=language,
            stage=stage,
            error_details=error_details,
            description=description,
            subject=subject,
            scope=scope,
            stdout=stdout,
            stderr=stderr,
        )

        if updated_coder:
            self.coders[test_id] = updated_coder  # Update the stored coder
            self.metrics_manager.record_fix_result(
                test_id, subject, True, None
            )  # Record success
            logger.info(
                f"AI fix attempt for {test_id} completed (Aider run successful)."
            )
            return True
        else:
            err_msg = f"AI fix attempt failed during Aider run for stage {stage}"
            self.metrics_manager.record_fix_result(test_id, subject, False, err_msg)
            logger.error(
                f"AI fix attempt failed for {test_id} (AiderService returned None)."
            )
            # Keep the old coder instance for potential retry context or analysis
            return False

    async def _make_code_non_deterministic(
        self,
        code_file_path: str,
        metric: Dict[str, Any],  # Pass metric ref for info
        test_description: str,
    ) -> bool:
        """
        Attempts to modify code for non-determinism using AiderService. Updates coder instance.
        Now directly awaits the async AiderService method.

        Args:
            code_file_path: Path to the code file.
            metric: The metric object reference for this test.
            test_description: Context including original description and potentially last output.

        Returns:
            bool: True if the Aider modification ran successfully, False otherwise.
        """
        test_id = metric["test_id"]
        subject = metric["subject"]
        scope = metric["scope"]
        language = metric["language"]

        if test_id not in self.coders:
            err_msg = "Non-det modification failed: Coder instance missing"
            logger.error(
                f"Cannot attempt non-det modification for {test_id}: No coder instance."
            )
            self.metrics_manager.record_fix_result(
                test_id, subject, False, err_msg
            )  # Use fix_error field?
            return False

        existing_coder = self.coders[test_id]
        logger.info(
            f"Attempting AI modification to potentially introduce non-determinism: {test_id}..."
        )

        updated_coder = await self.aider_service.make_code_non_deterministic(
            existing_coder=existing_coder,
            code_file_path=code_file_path,
            test_id=test_id,
            language=language,
            test_description=test_description,
            subject=subject,
            scope=scope,
        )

        if updated_coder:
            self.coders[test_id] = updated_coder  # Update the stored coder
            # Clear any previous fix error if modification was successful
            self.metrics_manager.record_fix_result(test_id, subject, True, None)
            logger.info(
                f"AI non-determinism modification attempt completed for {test_id}."
            )
            return True
        else:
            err_msg = "AI non-determinism modification failed during Aider run"
            self.metrics_manager.record_fix_result(test_id, subject, False, err_msg)
            logger.error(f"AI non-determinism modification failed for {test_id}.")
            return False

    # --- Result Handling Methods (Simplified: Format API result dict) ---

    def _handle_success(
        self,
        test_id: str,
        subject: str,
        exec_result: ExecutionResult,
        forced_attempt_flag: bool,
    ) -> Dict[str, Any]:
        """Formats the result dictionary for successful, deterministic execution."""
        exit_code = exec_result.exit_code
        status = "SUCCESS" if exit_code == 0 else "EXECUTION_ERROR"
        log_level = logger.info if status == "SUCCESS" else logger.warning
        emoji = "✅" if status == "SUCCESS" else "⚠️"
        log_level(
            f"{emoji} Test {test_id} ({subject}) completed deterministically with status: {status} (Exit Code: {exit_code})"
        )

        failure_reason = (
            f"Execution finished with non-zero exit code: {exit_code}"
            if exit_code != 0
            else None
        )

        # Prepare Result Data
        result_data = {
            "test_id": test_id,
            "language": self.language,
            "scope": self.scope_name,
            "subject": subject,
            "status": status,
            "stdout": exec_result.stdout,
            "stderr": exec_result.stderr,
            "exit_code": exit_code,
            "non_deterministic": False,
            "non_deterministic_detection_method": None,
            "non_deterministic_detected_at_stage": None,
            "system_error": False,
            "timeout_error": False,
            "stage": "execute",  # Stage where success was determined
            "forced_attempt": forced_attempt_flag,
            "error": failure_reason,  # Include non-zero exit code reason
        }
        return result_data

    def _handle_failure(
        self,
        test_id: str,
        subject: str,
        stage: str,
        error_msg: str,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
        exit_code: Optional[int] = None,
        system_error: bool = False,
        timeout_error: bool = False,
        non_det_info: Optional[
            Tuple[bool, Optional[str], Optional[str]]
        ] = None,  # (is_non_det, method, stage)
        forced_attempt_flag: bool = False,
    ) -> Dict[str, Any]:
        """Formats the result dictionary for generic failures."""
        if system_error:
            log_prefix, overall_status, failure_reason = (
                "SYSTEM_ERROR",
                "SYSTEM_ERROR",
                f"SystemError at {stage}: {error_msg}",
            )
        elif timeout_error:
            log_prefix, overall_status, failure_reason = (
                "TIMEOUT_ERROR",
                "TIMEOUT",
                f"Execution Timeout at {stage}: {error_msg}",
            )
        else:
            log_prefix, overall_status, failure_reason = (
                f"FAILED ({stage.upper()})",
                "FAILED",
                f"{stage.capitalize()} failed: {error_msg}",
            )

        logger.error(f"{log_prefix} for {test_id} ({subject}): {error_msg}")

        is_non_det, non_det_method, non_det_stage = non_det_info or (False, None, None)

        # Prepare Result Data
        result_data = {
            "test_id": test_id,
            "language": self.language,
            "scope": self.scope_name,
            "subject": subject,
            "status": overall_status,
            "stage": stage,  # Stage where failure occurred
            "error": failure_reason,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "non_deterministic": is_non_det,
            "non_deterministic_detection_method": non_det_method,
            "non_deterministic_detected_at_stage": non_det_stage,
            "system_error": system_error,
            "timeout_error": timeout_error,
            "forced_attempt": forced_attempt_flag,
        }
        return result_data

    def _handle_non_deterministic(
        self,
        test_id: str,
        subject: str,
        qan_outcome: Dict[str, Any],  # Result from QANService call
        forced_attempt_flag: bool,
    ) -> Dict[str, Any]:
        """Formats the result dictionary for non-deterministic detection by QAN."""
        stage = qan_outcome.get("stage", "unknown")
        detection_method = qan_outcome.get("detection_method", "unknown")
        details = qan_outcome.get("details", "No details provided")
        log_forced = "(Forced Attempt)" if forced_attempt_flag else ""
        logger.info(
            f"🎯 Non-deterministic behavior DETECTED at '{stage.upper()}' stage for {test_id} ({subject}) {log_forced}"
        )

        # Prepare Result Data
        result_data = {
            "test_id": test_id,
            "language": self.language,
            "scope": self.scope_name,
            "subject": subject,
            "status": "NON_DETERMINISTIC",
            "stdout": qan_outcome.get(
                "stdout", "Non-deterministic results detected."
            ),  # Include if available
            "stderr": details,
            "exit_code": qan_outcome.get(
                "exit_code", -1
            ),  # Include if available, else -1
            "non_deterministic": True,
            "non_deterministic_detection_method": detection_method,
            "non_deterministic_detected_at_stage": stage,
            "system_error": False,
            "timeout_error": False,
            "stage": stage,  # Stage where non-determinism was detected
            "forced_attempt": forced_attempt_flag,
            "error": f"Non-deterministic at {stage}: {details}",
        }
        return result_data

    def _handle_max_attempts_reached(
        self,
        test_id: str,
        subject: str,
        max_attempts: int,
        forced_attempt_flag: bool,
    ) -> Dict[str, Any]:
        """Formats the result dictionary when the maximum pipeline attempts are reached."""
        failure_reason = f"Pipeline exceeded maximum allowed actual attempts ({max_attempts}). Possible infinite loop or persistent unfixable error."
        logger.error(
            f"MAX_ATTEMPTS_REACHED for {test_id} ({subject}): {failure_reason}"
        )

        result_data = {
            "test_id": test_id,
            "language": self.language,
            "scope": self.scope_name,
            "subject": subject,
            "status": "FAILED",  # Keep status as FAILED for consistency
            "stage": "pipeline_max_attempts",  # Indicate where the failure occurred
            "error": failure_reason,
            "stdout": None,  # No specific stage output applies
            "stderr": None,
            "exit_code": -1,  # Use a specific code? -1 usually indicates error
            "non_deterministic": False,
            "non_deterministic_detection_method": None,
            "non_deterministic_detected_at_stage": None,
            "system_error": False,  # Not strictly a system error, but a process failure
            "timeout_error": False,
            "forced_attempt": forced_attempt_flag,
        }
        return result_data

    # --- Pipeline Orchestration ---

    # --- Stage Execution Helpers ---
    async def _execute_precompile_stage(self, code_bytes: bytes) -> Dict[str, Any]:
        """Executes the precompile stage via QAN service."""
        return await self.qan_service.precompile(self.language, code_bytes)

    async def _execute_compile_stage(self, decompiled_bytes: bytes) -> Dict[str, Any]:
        """Executes the compile stage via QAN service."""
        return await self.qan_service.compile(self.language, decompiled_bytes)

    async def _execute_execute_stage(self, binary_bytes: bytes) -> Dict[str, Any]:
        """Executes the execute stage via QAN service."""
        return await self.qan_service.execute(binary_bytes)

    # --- Fix Attempt Helper ---
    async def _attempt_stage_fix(
        self,
        stage: str,
        qan_outcome: Dict[str, Any],
        code_file_path: str,
        metric: Dict[str, Any],  # Pass metric ref
    ) -> bool:
        """
        Checks if a fix should be attempted for a failed stage, performs the fix,
        updates metrics via manager, and returns if a retry is warranted.

        Returns:
            bool: True if a fix was successfully attempted (caller should retry stage),
                  False otherwise (caller should handle as final failure).
        """
        test_id = metric["test_id"]
        subject = metric["subject"]

        # Check if error is potentially fixable (not system error)
        if qan_outcome.get("system_error"):
            logger.info(
                f"SystemError detected for {test_id} at stage {stage}. Cannot fix."
            )
            return False

        # Determine max attempts for the stage
        max_attempts_map = {
            "precompile": self.settings.MAX_PRECOMPILE_ATTEMPTS,
            "compile": self.settings.MAX_COMPILE_ATTEMPTS,
            "execute": self.settings.MAX_EXECUTE_ATTEMPTS,
        }
        stage_max_fixes = max_attempts_map.get(stage, 0)

        # Get current fix attempt count for this stage from metric
        fix_attempt_count = metric.get("stage_attempts", {}).get(stage, 0)

        if fix_attempt_count < stage_max_fixes:
            # Record the intent to attempt a fix (increments counter)
            self.metrics_manager.record_fix_attempt(test_id, subject, stage)
            current_attempt_num = fix_attempt_count + 1  # For logging
            logger.info(
                f"Attempting {stage} AI fix #{current_attempt_num}/{stage_max_fixes} for {test_id}..."
            )

            # Extract details for the fix prompt
            error_details = qan_outcome.get("error", "Unknown QAN error")
            stdout = qan_outcome.get("stdout")
            stderr = qan_outcome.get("stderr")
            is_timeout = qan_outcome.get("timeout_error", False)
            fix_stage_name = f"{stage}{'_timeout' if is_timeout else ''}"

            # Call the fix function
            fix_run_success = await self._fix_code(
                code_file_path=code_file_path,
                metric=metric,  # Pass metric ref
                stage=fix_stage_name,
                error_details=error_details,
                stdout=stdout,
                stderr=stderr,
            )
            # Note: _fix_code already called record_fix_result in manager

            if fix_run_success:
                logger.info(
                    f"AI fix successful for {test_id}, restarting stage '{stage}'."
                )
                return True  # Indicate that the caller should retry the stage
            else:
                logger.error(
                    f"AI fix run failed for {test_id} at stage {stage}. Aborting retries for this stage."
                )
                return False  # Indicate fix attempt failed, do not retry
        else:
            # Max attempts reached
            logger.error(
                f"{stage.capitalize()} failed after {fix_attempt_count}/{stage_max_fixes} fix attempts for {test_id}."
            )
            return False  # Indicate max attempts reached, do not retry

    # --- Non-Determinism Forcing Helpers ---

    def _backup_code_file(self, code_file_path: str) -> Optional[str]:
        """Creates a backup of the code file."""
        backup_file = f"{code_file_path}.backup_nondet_attempt"
        try:
            with open(code_file_path, "rb") as src, open(backup_file, "wb") as dst:
                dst.write(src.read())
            logger.debug(f"Created backup for non-determinism attempt: {backup_file}")
            return backup_file
        except Exception as e:
            logger.error(
                f"Failed to backup {code_file_path} for non-determinism attempt: {e}"
            )
            return None

    def _restore_code_file(
        self, backup_file_path: str, original_file_path: str
    ) -> bool:
        """Restores the code file from backup."""
        if not os.path.exists(backup_file_path):
            logger.error(f"Backup file {backup_file_path} not found for restoration.")
            return False
        try:
            os.replace(backup_file_path, original_file_path)
            logger.info(f"Restored original code from {backup_file_path}.")
            return True
        except Exception as e:
            logger.error(
                f"FATAL: Failed to restore {original_file_path} from backup {backup_file_path}: {e}"
            )
            return False

    def _cleanup_backup_file(self, backup_file_path: Optional[str]) -> None:
        """Removes the backup file if it exists."""
        if backup_file_path and os.path.exists(backup_file_path):
            try:
                os.remove(backup_file_path)
                logger.debug(f"Removed backup file {backup_file_path}.")
            except Exception as e:
                logger.warning(f"Could not remove backup file {backup_file_path}: {e}")

    async def _attempt_force_non_determinism(
        self,
        code_file_path: str,
        metric: Dict[str, Any],  # Pass metric ref
        last_stdout: Optional[str],
    ) -> Dict[str, Any]:
        """
        Attempts to make code non-deterministic, re-runs the pipeline, handles backup/restore.

        Returns:
            The result dictionary from the pipeline run *after* the modification attempt.
        """
        test_id = metric["test_id"]
        subject = metric["subject"]
        original_description = metric.get("description", "No description available.")
        force_prompt_description = f"""Original Test Description:\n{original_description}\n\n---\nLast Successful Deterministic Run Output (stdout):\n{last_stdout or "(No previous stdout)"}\n---"""

        backup_file = self._backup_code_file(code_file_path)
        if backup_file is None:
            # Backup failed, return failure result immediately
            failure_reason = "Backup failed before non-determinism attempt"
            result = self._handle_failure(
                test_id,
                subject,
                "force_nondet_backup",
                failure_reason,
                system_error=True,  # Treat as system error? Or specific stage?
                forced_attempt_flag=True,  # Mark attempt even if backup failed
            )
            # Also record in metrics
            self.metrics_manager.record_test_completion(
                test_id,
                subject,
                result["status"],
                False,
                failure_reason,
                False,
                None,
                None,
                True,
                False,
                0.0,  # Assume 0 duration for backup fail
            )
            return result

        modification_successful = await self._make_code_non_deterministic(
            code_file_path, metric, force_prompt_description
        )
        # Note: _make_code_non_deterministic calls record_fix_result in manager

        pipeline_result = None
        if modification_successful:
            logger.info(
                f"AI modification for non-determinism applied to {test_id}. Re-running validation pipeline..."
            )
            # Record the attempt in metrics
            self.metrics_manager.record_non_det_force_attempt(test_id, subject)

            # Re-run Pipeline - prevent recursive forcing attempts
            # The metric object reference `metric` should still be valid
            pipeline_result = await self._run_qan_pipeline(
                code_file_path, metric, allow_non_det_attempt=False
            )
            # _run_qan_pipeline now handles final metric recording

            # Ensure the forced_attempt flag is set in the final result dict
            if pipeline_result:
                pipeline_result["forced_attempt"] = True

        else:
            # Modification failed (Aider error)
            logger.warning(
                f"Failed to apply AI non-deterministic modification for {test_id}. Restoring original code."
            )
            # Error already recorded by _make_code_non_deterministic via record_fix_result
            if not self._restore_code_file(backup_file, code_file_path):
                # Restore failed - this is critical
                failure_reason = "FATAL: AI modification failed AND restore failed"
                result = self._handle_failure(
                    test_id,
                    subject,
                    "force_nondet_restore_fail",
                    failure_reason,
                    system_error=True,
                    forced_attempt_flag=True,
                )
                self.metrics_manager.record_test_completion(
                    test_id,
                    subject,
                    result["status"],
                    False,
                    failure_reason,
                    False,
                    None,
                    None,
                    True,
                    False,
                    0.0,
                )
                self._cleanup_backup_file(backup_file)  # Attempt cleanup anyway
                return result

            # Restore succeeded, now return failure indicating modification failed
            failure_reason = "AI non-deterministic modification failed"
            pipeline_result = self._handle_failure(
                test_id,
                subject,
                "force_nondet_modify_fail",
                failure_reason,
                forced_attempt_flag=True,
            )
            # Record the final state in metrics
            self.metrics_manager.record_test_completion(
                test_id,
                subject,
                pipeline_result["status"],
                False,
                failure_reason,
                False,
                None,
                None,
                False,
                False,
                0.0,  # Duration handled elsewhere
            )

        # Cleanup backup file regardless of outcome (after potential restore)
        self._cleanup_backup_file(backup_file)

        return pipeline_result

    async def _check_and_run_force_non_determinism(
        self,
        code_file_path: str,
        metric: Dict[str, Any],
        initial_success_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Checks if non-determinism forcing should be run, manages the attempt loop,
        and returns the final result after attempts.
        """
        test_id = metric["test_id"]
        subject = metric["subject"]
        last_pipeline_result = initial_success_result  # Start with the original success

        if self.settings.NON_DET_ATTEMPTS <= 0:
            return initial_success_result  # No attempts configured

        logger.info(
            f"Checking if non-determinism forcing is needed for {test_id} ({subject})."
        )

        for attempt in range(self.settings.NON_DET_ATTEMPTS):
            logger.info(
                f"Attempting to force non-determinism for {test_id} (Attempt {attempt + 1}/{self.settings.NON_DET_ATTEMPTS})..."
            )

            # Get the most recent metric state before the attempt
            metric_info = self.metrics_manager.find_metric(test_id, subject)
            if not metric_info:
                logger.error(
                    f"Metrics lost before non-det force attempt {attempt + 1} for {test_id}. Stopping force attempts."
                )
                # Return the result from the *previous* iteration or initial success
                return last_pipeline_result

            current_metric = metric_info[0]
            last_stdout = last_pipeline_result.get(
                "stdout"
            )  # Use stdout from previous run

            # Run the forcing logic (modifies code, reruns pipeline)
            force_pipeline_result = await self._attempt_force_non_determinism(
                code_file_path,
                current_metric,
                last_stdout,
            )

            last_pipeline_result = (
                force_pipeline_result  # Store the result of this attempt
            )

            # Check outcome of the forcing attempt's pipeline run
            if force_pipeline_result.get("non_deterministic"):
                logger.info(
                    f"Non-determinism successfully forced and detected on attempt {attempt + 1} for {test_id}."
                )
                # The final result is non-deterministic, metrics updated in _attempt_force_non_determinism -> _run_qan_pipeline
                return force_pipeline_result  # Exit loop and return ND result
            elif force_pipeline_result.get("status", "FAILED") not in [
                "SUCCESS",
                "EXECUTION_ERROR",
            ]:
                logger.warning(
                    f"Pipeline failed during non-det forcing attempt {attempt + 1} for {test_id} (Status: {force_pipeline_result.get('status')}). Stopping forcing attempts."
                )
                # Return the failure result from this attempt
                return force_pipeline_result  # Exit loop

            if attempt < self.settings.NON_DET_ATTEMPTS - 1:
                logger.info(
                    f"Non-determinism forcing attempt {attempt + 1} did not result in detected non-determinism. Continuing to next attempt..."
                )
                await asyncio.sleep(0.5)  # Optional small delay
            else:
                logger.info(
                    f"All non-determinism forcing attempts completed for {test_id}. Final state was deterministic (Status: {force_pipeline_result.get('status')})."
                )

        # If loop finishes without returning early, return the result of the last attempt
        return last_pipeline_result

    # --- Main Pipeline Runner ---
    async def _run_qan_pipeline(
        self,
        code_file_path: str,
        metric: Dict[str, Any],  # Pass metric ref
        allow_non_det_attempt: bool = True,
    ) -> Dict[str, Any]:
        """
        Orchestrates the main QAN pipeline: precompile -> compile -> execute.
        Handles retries with AI fixes for each stage using helper methods.
        Includes a limit on the total number of actual pipeline attempts.
        Records final metrics via MetricsManager.

        Args:
            code_file_path: Path to the source code file.
            metric: The managed metric dictionary reference for this test.
            allow_non_det_attempt: Flag to allow triggering the non-determinism forcing logic.

        Returns:
            A dictionary representing the final API result of the pipeline.
        """
        test_id = metric["test_id"]
        subject = metric["subject"]
        final_result_data = None  # Stores the dict to be returned
        actual_pipeline_attempts = 0  # Initialize counter for total attempts

        current_stage = "precompile"
        stage_input_data = None  # Will hold code_bytes, decompiled_bytes, binary_bytes

        # Read initial code file
        try:
            with open(code_file_path, "rb") as f:
                stage_input_data = f.read()
        except Exception as e:
            logger.error(
                f"Failed to read code file {code_file_path} for {test_id}: {e}"
            )
            failure_reason = f"Pipeline start failed: File read error: {e}"
            final_result_data = self._handle_failure(
                test_id,
                subject,
                "pipeline_start",
                failure_reason,
                system_error=True,
                forced_attempt_flag=metric.get(
                    "non_determinism_forcing_attempt", False
                ),
            )
            self.metrics_manager.record_test_completion(
                test_id,
                subject,
                final_result_data["status"],
                False,
                failure_reason,
                False,
                None,
                None,
                True,
                False,
                0.0,
            )
            return final_result_data

        # --- Pipeline Loop ---
        while current_stage is not None:
            # Check overall attempt limit
            actual_pipeline_attempts += 1
            if actual_pipeline_attempts > self.settings.MAX_ACTUAL_PIPELINE_ATTEMPTS:
                logger.error(
                    f"Pipeline for {test_id} ({subject}) exceeded max actual attempts ({self.settings.MAX_ACTUAL_PIPELINE_ATTEMPTS}). Aborting."
                )
                final_result_data = self._handle_max_attempts_reached(
                    test_id,
                    subject,
                    self.settings.MAX_ACTUAL_PIPELINE_ATTEMPTS,
                    metric.get("non_determinism_forcing_attempt", False),
                )
                # Record this specific failure in metrics
                self.metrics_manager.record_test_completion(
                    test_id,
                    subject,
                    final_result_data["status"],  # Should be FAILED
                    False,  # Not a success
                    final_result_data["error"],  # Specific error reason
                    False,  # Not non-deterministic
                    None,
                    None,
                    False,  # Not a QAN system error
                    False,  # Not a timeout
                    0.0,  # Duration updated later
                )
                # Update overall failure reason in metric to be specific
                latest_metric_info = self.metrics_manager.find_metric(test_id, subject)
                if latest_metric_info:
                    latest_metric_ref = latest_metric_info[0]
                    latest_metric_ref["failure_reason"] = final_result_data["error"]

                current_stage = None  # Break the loop
                continue  # Go to end of loop check

            # Record actual attempt for this specific stage run (existing logic)
            self.metrics_manager.record_actual_stage_attempt(
                test_id, subject, current_stage
            )

            # Update log message to include total attempts
            stage_log_msg = (
                (
                    f"(Fix Attempt {metric.get('stage_attempts', {}).get(current_stage, 0)}) "
                    f"(Actual Attempt {actual_pipeline_attempts}/{self.settings.MAX_ACTUAL_PIPELINE_ATTEMPTS})"
                )
                if metric.get("stage_attempts", {}).get(current_stage, 0) > 0
                else (
                    f"(Initial Attempt) (Actual Attempt {actual_pipeline_attempts}/{self.settings.MAX_ACTUAL_PIPELINE_ATTEMPTS})"
                )
            )
            logger.info(
                f"--- Running Stage: {current_stage.upper()} for {test_id} {stage_log_msg} ---"
            )

            # Execute the current stage
            qan_outcome = None
            try:
                if current_stage == "precompile":
                    qan_outcome = await self._execute_precompile_stage(stage_input_data)
                elif current_stage == "compile":
                    qan_outcome = await self._execute_compile_stage(stage_input_data)
                elif current_stage == "execute":
                    qan_outcome = await self._execute_execute_stage(stage_input_data)
                else:
                    raise ValueError(f"Unknown pipeline stage: {current_stage}")
            except Exception as stage_exec_err:
                logger.error(
                    f"Unexpected error executing stage '{current_stage}' for {test_id}: {stage_exec_err}",
                    exc_info=True,
                )
                failure_reason = (
                    f"Pipeline error during stage {current_stage}: {stage_exec_err}"
                )
                final_result_data = self._handle_failure(
                    test_id,
                    subject,
                    f"{current_stage}_exception",
                    failure_reason,
                    system_error=True,
                    forced_attempt_flag=metric.get(
                        "non_determinism_forcing_attempt", False
                    ),
                )
                self.metrics_manager.record_test_completion(
                    test_id,
                    subject,
                    final_result_data["status"],
                    False,
                    failure_reason,
                    False,
                    None,
                    None,
                    True,
                    False,
                    0.0,
                )
                return (
                    final_result_data  # Return immediately on stage execution exception
                )

            # --- Process QAN Outcome ---
            if qan_outcome.get("success"):
                logger.info(f"Stage '{current_stage}' successful for {test_id}.")
                self.metrics_manager.record_stage_success(
                    test_id, subject, current_stage
                )
                stage_input_data = qan_outcome.get(
                    "result"
                )  # Prepare input for the next stage

                # --- Stage Transition ---
                if current_stage == "precompile":
                    current_stage = "compile"
                elif current_stage == "compile":
                    current_stage = "execute"
                elif current_stage == "execute":
                    # --- Successful Execution ---
                    exec_result: ExecutionResult = stage_input_data
                    success_result_data = self._handle_success(
                        test_id,
                        subject,
                        exec_result,
                        metric.get("non_determinism_forcing_attempt", False),
                    )
                    # Record preliminary success in metrics
                    self.metrics_manager.record_test_completion(
                        test_id,
                        subject,
                        success_result_data["status"],
                        success_result_data["exit_code"] == 0,
                        success_result_data["error"],
                        False,
                        None,
                        None,
                        False,
                        False,
                        0.0,  # Duration updated later
                    )

                    # --- Attempt Force Non-Determinism ---
                    if allow_non_det_attempt:
                        final_result_data = (
                            await self._check_and_run_force_non_determinism(
                                code_file_path, metric, success_result_data
                            )
                        )
                        # Metrics are updated within the forcing logic calls
                    else:
                        final_result_data = (
                            success_result_data  # Use original success result
                        )

                    current_stage = None  # End the pipeline loop

            # --- Handle QAN Errors ---
            elif qan_outcome.get("non_deterministic"):
                logger.info(
                    f"Non-deterministic behavior detected during {current_stage} for {test_id}."
                )
                final_result_data = self._handle_non_deterministic(
                    test_id,
                    subject,
                    qan_outcome,
                    metric.get("non_determinism_forcing_attempt", False),
                )
                self.metrics_manager.record_test_completion(
                    test_id,
                    subject,
                    final_result_data["status"],
                    True,  # Non-det is a success
                    None,
                    True,
                    final_result_data["non_deterministic_detection_method"],
                    final_result_data["non_deterministic_detected_at_stage"],
                    False,
                    False,
                    0.0,
                )
                current_stage = None  # End the pipeline

            else:  # Handle regular errors (CompileError, ExecuteError, Timeout, System, Server, etc.)
                error_msg = qan_outcome.get("error", "Unknown QAN error")
                is_system_error = qan_outcome.get("system_error", False)
                is_timeout_error = qan_outcome.get("timeout_error", False)
                logger.warning(
                    f"Stage '{current_stage}' failed for {test_id}. Error: {error_msg}. Checking fix attempts..."
                )

                if is_system_error:
                    # System errors are not fixable
                    failure_reason = f"SystemError at {current_stage}: {error_msg}"
                    final_result_data = self._handle_failure(
                        test_id,
                        subject,
                        f"{current_stage}_system_error",
                        failure_reason,
                        stderr=qan_outcome.get("stderr"),
                        system_error=True,
                        forced_attempt_flag=metric.get(
                            "non_determinism_forcing_attempt", False
                        ),
                    )
                    self.metrics_manager.record_test_completion(
                        test_id,
                        subject,
                        final_result_data["status"],
                        False,
                        failure_reason,
                        False,
                        None,
                        None,
                        True,
                        False,
                        0.0,
                    )
                    current_stage = None  # End pipeline
                    continue  # Skip fix attempt logic

                # --- Attempt AI Fix ---
                should_retry_stage = await self._attempt_stage_fix(
                    stage=current_stage,
                    qan_outcome=qan_outcome,
                    code_file_path=code_file_path,
                    metric=metric,
                )

                if should_retry_stage:
                    # Fix attempt successful, re-read code and retry same stage
                    try:
                        with open(code_file_path, "rb") as f:
                            stage_input_data = f.read()
                        continue  # Go to next iteration of the while loop for the *same* stage
                    except Exception as e:
                        logger.error(
                            f"Failed to re-read modified code file {code_file_path} for {test_id} after fix: {e}"
                        )
                        failure_reason = f"File re-read error after successful fix at stage {current_stage}: {e}"
                        final_result_data = self._handle_failure(
                            test_id,
                            subject,
                            f"{current_stage}_fix_read",
                            failure_reason,
                            system_error=True,  # Treat as critical system error
                            forced_attempt_flag=metric.get(
                                "non_determinism_forcing_attempt", False
                            ),
                        )
                        self.metrics_manager.record_test_completion(
                            test_id,
                            subject,
                            final_result_data["status"],
                            False,
                            failure_reason,
                            False,
                            None,
                            None,
                            True,
                            False,
                            0.0,
                        )
                        current_stage = None  # End pipeline
                else:
                    # Fix attempt failed or max attempts reached
                    fix_error_reason = metric.get(
                        "fix_error", "Fix attempts exhausted or AI fix failed"
                    )
                    failure_reason = f"{current_stage.capitalize()} failed: {fix_error_reason}. Last QAN error: {error_msg}"
                    logger.error(f"{failure_reason} for {test_id}. Marking as failed.")
                    final_result_data = self._handle_failure(
                        test_id=test_id,
                        subject=subject,
                        stage=f"{current_stage}_failed_fix",
                        error_msg=failure_reason,
                        stdout=qan_outcome.get("stdout"),
                        stderr=qan_outcome.get("stderr"),
                        exit_code=qan_outcome.get("exit_code"),
                        system_error=False,  # Failure is due to QAN error + fix failure
                        timeout_error=is_timeout_error,
                        forced_attempt_flag=metric.get(
                            "non_determinism_forcing_attempt", False
                        ),
                    )
                    self.metrics_manager.record_test_completion(
                        test_id,
                        subject,
                        final_result_data["status"],
                        False,
                        failure_reason,
                        False,
                        None,
                        None,
                        False,
                        is_timeout_error,
                        0.0,
                    )
                    current_stage = None  # End pipeline

        # --- End of Pipeline Loop ---
        if final_result_data is None:
            # Should not happen if loop logic is correct, but handle defensively
            logger.error(
                f"Pipeline loop exited unexpectedly for {test_id}. Stage was: {current_stage}"
            )
            failure_reason = "Pipeline loop terminated unexpectedly"
            final_result_data = self._handle_failure(
                test_id,
                subject,
                "pipeline_logic",
                failure_reason,
                system_error=True,
                forced_attempt_flag=metric.get(
                    "non_determinism_forcing_attempt", False
                ),
            )
            self.metrics_manager.record_test_completion(
                test_id,
                subject,
                final_result_data["status"],
                False,
                failure_reason,
                False,
                None,
                None,
                True,
                False,
                0.0,
            )

        return final_result_data

    # --- Result Saving Helper ---
    def _save_api_result(self, result_data: Optional[Dict[str, Any]]) -> None:
        """Saves the final API result dictionary to a JSON file."""
        if not result_data or not isinstance(result_data, dict):
            logger.warning("Attempted to save invalid result data.")
            return

        subject = result_data.get("subject")
        test_id = result_data.get("test_id")

        if not subject or not test_id:
            logger.error(
                "Cannot save API result: Missing subject or test_id in result data."
            )
            return

        try:
            sanitized_subject = sanitize_filename(subject)
            results_filename = f"{sanitized_subject}_{test_id}_results.json"
            results_file = os.path.join(
                self.api_results_dir_for_scope, results_filename
            )

            if not save_to_json(result_data, results_file):
                logger.error(
                    f"Failed to write API results file {results_file} (save_to_json returned False)"
                )
            else:
                logger.debug(f"API result saved to {results_file}")
        except Exception as e:
            logger.error(
                f"Exception saving API result for {test_id} ({subject}): {e}",
                exc_info=True,
            )

    # --- Cleanup Helper ---
    async def _cleanup_extensionless_files(self, subject: str):
        """Cleans up files without extensions in the test results directory for the given subject."""
        try:
            sanitized_subject = sanitize_filename(subject)
            # Construct the target directory path: results/tests/<scope>/<subject>/<language>/
            target_dir = os.path.join(
                self.settings.TESTS_DIR,
                self.sanitized_scope_name,
                sanitized_subject,
                self.language,
            )

            if not os.path.isdir(target_dir):
                logger.debug(f"Cleanup skipped: Test directory not found: {target_dir}")
                return

            logger.info(f"Running cleanup for extensionless files in: {target_dir}")
            cleaned_count = 0
            for item_name in os.listdir(target_dir):
                item_path = os.path.join(target_dir, item_name)
                # Check if it's a file and if it has no extension (splitext returns empty string for extension)
                if os.path.isfile(item_path) and not os.path.splitext(item_name)[1]:
                    try:
                        os.remove(item_path)
                        logger.debug(f"Cleaned up extensionless file: {item_path}")
                        cleaned_count += 1
                    except OSError as e:
                        logger.warning(
                            f"Failed to remove extensionless file {item_path}: {e}"
                        )
            if cleaned_count > 0:
                logger.warning(
                    f"Cleanup finished: Removed {cleaned_count} extensionless files."
                )
            else:
                logger.debug(
                    "Cleanup finished: No extensionless files found to remove."
                )

        except Exception as e:
            logger.error(
                f"Error during test directory cleanup for subject '{subject}': {e}",
                exc_info=True,
            )

    # --- Test Processing Entry Point ---
    async def process_test(
        self,
        subject_plan: Dict[str, Any],
        test: Dict[str, Any],
        subject_index: int,
        subject_count: int,
        test_index: int,
        test_count: int,
    ) -> Dict[str, Any]:
        """Processes a single test: generate, pipeline, update metrics/state, save result, cleanup."""
        subject = subject_plan.get("subject", "Unknown Subject")
        test_id = test.get("id", "unknown_test_id")
        description = test.get("description", "No description")
        start_time = time.time()
        result_data = {}  # Initialize result dictionary

        logger.info(
            f"Processing Test [{subject_index}/{subject_count} Subj: '{subject}', {test_index}/{test_count} Test: {test_id}] Lang: {self.language}, Scope: '{self.scope_name}'"
        )

        # --- Code Generation ---
        code_file, metric_ref = await self._generate_code(subject, test)
        # _generate_code now handles metric creation/finding and recording generation result

        if metric_ref is None:  # Critical error during metric handling in generate_code
            logger.critical(
                f"Metric handling failed during generation for {test_id}. Aborting test."
            )
            # Cannot reliably call _handle_failure without metric
            result_data = {
                "status": "FAILED",
                "stage": "generation_metric_error",
                "error": "Metric handling failure",
                "test_id": test_id,
                "subject": subject,
                "scope": self.scope_name,
                "language": self.language,
                "total_duration": time.time() - start_time,
                "skipped": False,
            }
            # Cannot update metrics or state reliably here
            self._save_api_result(result_data)  # Save what we can
            await self._cleanup_extensionless_files(
                subject
            )  # Attempt cleanup even on early exit
            return result_data

        if not code_file or not metric_ref.get("generation", {}).get("success"):
            error_msg = metric_ref.get("generation", {}).get(
                "error", "Code generation failed"
            )
            logger.error(
                f"Code generation failed for {test_id}. Skipping QAN pipeline. Reason: {error_msg}"
            )
            # Format failure result
            result_data = self._handle_failure(
                test_id,
                subject,
                "generation",
                error_msg,
                forced_attempt_flag=metric_ref.get(
                    "non_determinism_forcing_attempt", False
                ),
            )
            # Record final state in metrics
            self.metrics_manager.record_test_completion(
                test_id,
                subject,
                result_data["status"],
                False,
                error_msg,
                False,
                None,
                None,
                False,
                False,
                metric_ref.get("generation", {}).get("duration", 0.0),
            )
        else:
            # --- QAN API Pipeline ---
            try:
                # _run_qan_pipeline handles its own metric recording for completion
                result_data = await self._run_qan_pipeline(code_file, metric_ref)
            except Exception as e:
                logger.error(
                    f"Unexpected critical error during QAN pipeline for {test_id}: {e}",
                    exc_info=True,
                )
                # Use the latest metric state available for failure handling
                latest_metric_info = self.metrics_manager.find_metric(test_id, subject)
                metric_for_fail = (
                    latest_metric_info[0] if latest_metric_info else metric_ref
                )  # Fallback
                failure_reason = f"Pipeline exception: {e}"
                result_data = self._handle_failure(
                    test_id,
                    subject,
                    "pipeline_exception",
                    failure_reason,
                    system_error=True,
                    forced_attempt_flag=metric_for_fail.get(
                        "non_determinism_forcing_attempt", False
                    ),
                )
                # Record this critical failure in metrics
                self.metrics_manager.record_test_completion(
                    test_id,
                    subject,
                    result_data["status"],
                    False,
                    failure_reason,
                    False,
                    None,
                    None,
                    True,
                    False,
                    0.0,  # Duration updated below
                )

        # --- Finalization ---
        total_duration = time.time() - start_time
        final_status = result_data.get("status", "UNKNOWN")

        # Update total duration in the metric object (if found)
        # Note: record_test_completion already set duration, but this updates with total wall-clock time
        final_metric_info = self.metrics_manager.find_metric(test_id, subject)
        if final_metric_info:
            final_metric_ref = final_metric_info[0]
            final_metric_ref["total_duration"] = total_duration
        else:
            logger.error(
                f"Could not find final metrics for {test_id} to save total duration."
            )

        # Add common fields and final duration for summary/API result
        result_data["scope"] = self.scope_name
        result_data["subject"] = subject
        result_data["language"] = self.language
        result_data["test_description"] = description  # Add description to result
        result_data["skipped"] = False  # Processed in this run
        result_data["total_duration"] = total_duration

        # --- Save API Result ---
        self._save_api_result(result_data)

        # --- Update Execution State ---
        if final_status not in ["PENDING", "UNKNOWN"]:
            self.state_manager.mark_completed(test_id, subject, final_status)

        # --- Run Cleanup ---
        await self._cleanup_extensionless_files(subject)

        # Optional delay
        await asyncio.sleep(0.05)
        return result_data

    # --- Summary Printing ---
    def _print_summary(self, results: List[Dict[str, Any]]):
        """Prints a summary of the test execution results."""
        # (Implementation remains the same as before, relies on result dict structure)
        print(
            f"\n{'=' * 20} Test Execution Summary ({self.scope_name} / {self.language}) {'=' * 20}"
        )
        if not results:
            print("No tests were processed in this run for this scope/language.")
            print("=" * (42 + len(self.scope_name) + len(self.language) + 3))
            return

        status_counts = {}
        skipped_count = 0
        processed_count = 0
        subjects_processed = set()
        non_deterministic_found = 0
        system_errors = 0
        timeout_errors = 0
        other_failures = 0
        successful_deterministic = 0
        total_duration_processed = 0.0
        max_attempts_reached = 0  # Track max attempts failures

        for result in results:
            if not isinstance(result, dict):
                logger.warning(f"Invalid item found in results list: {type(result)}")
                continue  # Skip non-dict items

            # Use .get() with defaults for safety
            status = result.get("status", "UNKNOWN")
            stage = result.get("stage", "unknown")
            subjects_processed.add(result.get("subject", "Unknown"))

            if result.get("skipped", False):  # Check the skipped flag
                skipped_count += 1
            else:
                processed_count += 1
                status_counts[status] = status_counts.get(status, 0) + 1
                total_duration_processed += result.get("total_duration", 0.0)

                # Categorize processed results
                if status == "NON_DETERMINISTIC":
                    non_deterministic_found += 1
                elif status == "SYSTEM_ERROR":
                    system_errors += 1
                elif status == "TIMEOUT":
                    timeout_errors += 1
                elif status == "SUCCESS":
                    successful_deterministic += 1
                elif (
                    status == "FAILED" and stage == "pipeline_max_attempts"
                ):  # Check for max attempts stage
                    max_attempts_reached += 1
                    other_failures += 1  # Also count as failure
                else:  # Includes other FAILED, EXECUTION_ERROR, UNKNOWN, etc.
                    other_failures += 1

        print(f"Total test plan entries considered: {len(results)}")
        print(f"Subjects involved in this run: {len(subjects_processed)}")
        print(f"Tests skipped (already completed): {skipped_count}")
        print(f"Tests processed in this run: {processed_count}")
        if processed_count > 0:
            avg_duration = total_duration_processed / processed_count
            print(
                f"Total duration for processed tests: {total_duration_processed:.2f}s (Avg: {avg_duration:.2f}s/test)"
            )

        if processed_count > 0:
            print("\n--- Processed Test Statuses ---")
            # Use status_counts for accuracy
            print(
                f"  🎯 NON_DETERMINISTIC:       {status_counts.get('NON_DETERMINISTIC', 0)}"
            )
            print(f"  ✅ SUCCESS (Deterministic): {status_counts.get('SUCCESS', 0)}")
            print(f"  ⏳ TIMEOUT:                 {status_counts.get('TIMEOUT', 0)}")
            print(
                f"  💥 SYSTEM_ERROR:            {status_counts.get('SYSTEM_ERROR', 0)}"
            )
            # Summing up remaining failure types
            gen_fail = status_counts.get("FAILED", 0)
            exec_err = status_counts.get("EXECUTION_ERROR", 0)
            unknown = status_counts.get("UNKNOWN", 0)
            # Subtract max_attempts_reached from the general FAILED count if we display it separately
            print(
                f"  ❌ FAILURES (Exec/Comp/Gen/Other): {gen_fail + exec_err + unknown - max_attempts_reached}"
            )
            print(
                f"  🚫 MAX_ATTEMPTS_REACHED:    {max_attempts_reached}"  # Display separately
            )

        print(
            f"\n{'=' * 20} Detailed Results ({self.scope_name} / {self.language}) {'=' * 20}"
        )
        # Filter out non-dict results before sorting
        valid_results = [r for r in results if isinstance(r, dict)]
        sorted_results = sorted(
            valid_results,
            key=lambda x: (x.get("subject", ""), x.get("test_id", "")),
        )

        if not sorted_results and skipped_count == len(results):
            print("All tests were skipped.")
        elif not sorted_results:
            print("No valid results to display.")

        for result in sorted_results:
            test_id = result.get("test_id", "unknown")
            status = result.get("status", "UNKNOWN")
            subject = result.get("subject", "Unknown Subject")
            skipped = result.get("skipped", False)
            stage = result.get("stage", "?").upper()
            duration = result.get("total_duration", -1.0)
            duration_str = f"{duration:.2f}s" if duration >= 0 else "N/A"
            base_info = f"[{subject} / {test_id}] ({duration_str}): "
            error_snip = (
                (str(result.get("error", "")) or "").replace("\n", " ").strip()[:60]
            )

            if skipped:
                # Fetch previous status from StateManager if possible (optional enhancement)
                # prev_status = self.state_manager.get_completed_status(test_id, subject) or status
                print(f"{base_info}⏩ SKIPPED (Previously: {status})")
            elif status == "NON_DETERMINISTIC":
                method = result.get("non_deterministic_detection_method", "?")
                stage_nd = result.get(
                    "non_deterministic_detected_at_stage", stage
                ).upper()
                forced = "(Forced)" if result.get("forced_attempt") else ""
                print(
                    f"{base_info}🎯 NON_DETERMINISTIC {forced} (at {stage_nd} via {method})"
                )
            elif status == "SUCCESS":
                print(f"{base_info}✅ SUCCESS (Deterministic, Exit 0)")
            elif status == "EXECUTION_ERROR":
                code = result.get("exit_code", "N/A")
                print(f"{base_info}❌ EXECUTION ERROR (Exit {code}) - {error_snip}...")
            elif status == "SYSTEM_ERROR":
                print(f"{base_info}💥 SYSTEM ERROR (at {stage}) - {error_snip}...")
            elif status == "TIMEOUT":
                print(f"{base_info}⏳ TIMEOUT (at {stage}) - {error_snip}...")
            elif (
                status == "FAILED" and stage == "PIPELINE_MAX_ATTEMPTS"
            ):  # Check for specific stage
                print(f"{base_info}🚫 MAX ATTEMPTS REACHED - {error_snip}...")
            elif status == "FAILED":
                print(f"{base_info}❌ FAILED (at {stage}) - {error_snip}...")
            else:  # Handle UNKNOWN or any other status
                print(f"{base_info}❓ {status} (at {stage}) - {error_snip}...")
        print("=" * (42 + len(self.scope_name) + len(self.language) + 3))

    # --- Main Execution Loop ---
    async def run(self):
        """Main asynchronous execution method for the ExecutionAgent instance."""
        overall_start_time = time.time()
        all_results_for_this_run = []

        try:
            # --- Read Test Plans ---
            try:
                plan_files_to_load = []
                if not os.path.exists(self.plan_dir_for_scope) or not os.path.isdir(
                    self.plan_dir_for_scope
                ):
                    raise FileNotFoundError(
                        f"Plan directory not found: {self.plan_dir_for_scope}"
                    )

                if self.sanitized_specific_subject:
                    subject_file_path = os.path.join(
                        self.plan_dir_for_scope,
                        f"{self.sanitized_specific_subject}.yaml",
                    )
                    if os.path.exists(subject_file_path):
                        plan_files_to_load.append(subject_file_path)
                    else:
                        raise FileNotFoundError(
                            f"Specified subject plan file not found: {subject_file_path}"
                        )
                else:
                    plan_files_to_load = [
                        os.path.join(self.plan_dir_for_scope, f)
                        for f in os.listdir(self.plan_dir_for_scope)
                        if f.endswith(".yaml")
                    ]

                subject_plans = []
                # Import locally where needed or move to top if used more often
                from nondeterministic_agent.utils.file_utils import load_from_yaml

                for file_path in plan_files_to_load:
                    plan_data = load_from_yaml(file_path)
                    if (
                        isinstance(plan_data, dict)
                        and "scope" in plan_data
                        and "subject" in plan_data
                        and "tests" in plan_data
                    ):
                        if plan_data.get("scope") == self.scope_name:
                            subject_plans.append(plan_data)
                        else:
                            logger.warning(
                                f"Scope mismatch in {file_path}. Expected '{self.scope_name}', found '{plan_data.get('scope')}'. Skipping."
                            )
                    else:
                        logger.warning(
                            f"Invalid format or missing keys in plan file {file_path}. Skipping."
                        )

                if not subject_plans:
                    target = (
                        f"subject '{self.specific_subject}' in "
                        if self.specific_subject
                        else ""
                    )
                    logger.warning(
                        f"No valid test plans found for {target}scope '{self.scope_name}'. Nothing to process."
                    )
                    self._print_summary([])  # Print empty summary
                    return []

            except FileNotFoundError as e:
                logger.critical(f"Required plan file or directory not found: {e}")
                print(f"\nError: {e}")
                print(
                    f"Ensure planning step ran for '{self.scope_name}' and '{self.plan_dir_for_scope}' exists."
                )
                sys.exit(1)
            except Exception as plan_load_err:
                logger.critical(f"Error loading plans: {plan_load_err}", exc_info=True)
                print(f"\nError loading plans: {plan_load_err}")
                sys.exit(1)

            # --- Prepare Skip Logic ---
            completed_test_keys = (
                self.state_manager.get_completed_test_keys()
                if not self.force_restart
                else set()
            )
            if completed_test_keys:
                logger.info(
                    f"Will skip {len(completed_test_keys)} previously completed tests for {self.scope_name}/{self.language}."
                )

            # --- Calculate Progress ---
            subject_count = len(subject_plans)
            total_tests_in_scope = sum(
                len(sp.get("tests", []))
                for sp in subject_plans
                if isinstance(sp.get("tests"), list)
            )
            processed_test_count_overall = 0
            processed_test_count_this_run = 0

            # --- Enter QAN Service Context ---
            async with self.qan_service:
                await self.qan_service.health()  # Initial health check

                # --- Iterate Through Subjects and Tests ---
                for subject_index, subject_plan in enumerate(subject_plans, 1):
                    subject = subject_plan.get("subject", "Unknown Subject")
                    tests = subject_plan.get("tests", [])
                    if not isinstance(tests, list):
                        logger.error(
                            f"Invalid 'tests' format for subject '{subject}'. Skipping subject."
                        )
                        continue
                    test_count_in_subject = len(tests)
                    logger.info(
                        f"--- Processing Subject {subject_index}/{subject_count}: '{subject}' ({test_count_in_subject} tests planned) ---"
                    )

                    for test_index, test in enumerate(tests, 1):
                        processed_test_count_overall += 1
                        progress_percent = (
                            (processed_test_count_overall / total_tests_in_scope * 100)
                            if total_tests_in_scope > 0
                            else 0
                        )

                        if not isinstance(test, dict):
                            logger.error(
                                f"Skipping invalid test entry {test_index} in '{subject}'."
                            )
                            # Create a minimal failure result for the summary
                            all_results_for_this_run.append(
                                {
                                    "test_id": f"invalid_entry_{subject}_{test_index}",
                                    "subject": subject,
                                    "scope": self.scope_name,
                                    "language": self.language,
                                    "status": "FAILED",
                                    "stage": "parsing",
                                    "error": "Invalid test entry format",
                                    "skipped": False,
                                    "total_duration": 0,
                                }
                            )
                            continue
                        test_id = test.get("id", "unknown_test_id")
                        if not test_id or test_id == "unknown_test_id":
                            logger.error(
                                f"Skipping test {test_index} in '{subject}' due to invalid 'id'."
                            )
                            all_results_for_this_run.append(
                                {
                                    "test_id": f"invalid_id_{subject}_{test_index}",
                                    "subject": subject,
                                    "scope": self.scope_name,
                                    "language": self.language,
                                    "status": "FAILED",
                                    "stage": "parsing",
                                    "error": "Invalid/missing test ID",
                                    "skipped": False,
                                    "total_duration": 0,
                                }
                            )
                            continue

                        current_test_key = f"{self.scope_name}_{subject}_{test_id}"

                        # --- Skip Logic Check ---
                        if (
                            not self.force_restart
                            and current_test_key in completed_test_keys
                        ):
                            logger.info(
                                f"[{progress_percent:.1f}%] Skipping Test ({subject_index}/{subject_count}, {test_index}/{test_count_in_subject}): {subject} / {test_id} - Previously completed."
                            )
                            # Retrieve status for summary
                            prev_status = (
                                self.state_manager.get_completed_status(
                                    test_id, subject
                                )
                                or "UNKNOWN (Completed)"
                            )
                            all_results_for_this_run.append(
                                {
                                    "test_id": test_id,
                                    "subject": subject,
                                    "scope": self.scope_name,
                                    "language": self.language,
                                    "status": prev_status,
                                    "skipped": True,
                                    "total_duration": 0,
                                }
                            )
                            continue

                        # --- Process the Test ---
                        processed_test_count_this_run += 1
                        logger.info(
                            f"--- [Run Progress: {processed_test_count_this_run} processed / {len(completed_test_keys)} skipped. Overall: {progress_percent:.1f}%] ---"
                        )
                        try:
                            # process_test now handles metric updates, pipeline, state updates, result saving, cleanup
                            result = await self.process_test(
                                subject_plan,
                                test,
                                subject_index,
                                subject_count,
                                test_index,
                                test_count_in_subject,
                            )
                            all_results_for_this_run.append(result)
                        except Exception as e:
                            # Catch critical errors *within* process_test if they bubble up
                            logger.error(
                                f"CRITICAL ERROR processing test {test_id} ({subject}): {e}",
                                exc_info=True,
                            )
                            # Construct failure result directly
                            failure_result = {
                                "test_id": test_id,
                                "subject": subject,
                                "scope": self.scope_name,
                                "language": self.language,
                                "status": "FAILED",
                                "stage": "main_loop_exception",
                                "error": f"Critical error in process_test: {e}",
                                "skipped": False,
                                "total_duration": time.time()
                                - overall_start_time,  # Approx duration
                            }
                            all_results_for_this_run.append(failure_result)
                            # Attempt to update metrics/state defensively
                            try:
                                self.state_manager.mark_completed(
                                    test_id, subject, "FAILED"
                                )
                                self.metrics_manager.record_test_completion(
                                    test_id,
                                    subject,
                                    "FAILED",
                                    False,
                                    failure_result["error"],
                                    False,
                                    None,
                                    None,
                                    True,
                                    False,
                                    failure_result["total_duration"],
                                )
                            except Exception as final_save_err:
                                logger.error(
                                    f"Failed saving state/metrics after critical process_test error for {test_id}: {final_save_err}"
                                )

            # --- Final Saving (outside QAN context) ---
            logger.info("Saving final state and metrics...")
            self.state_manager.save_state()
            self.metrics_manager.save_all_metrics()

            # Save run summary (optional, but useful)
            summary_filename = (
                f"run_summary_{self.language}_{time.strftime('%Y%m%d_%H%M%S')}.json"
            )
            summary_file = os.path.join(
                self.state_manager.state_dir_for_scope, summary_filename
            )
            # Ensure data is serializable (filter out non-dicts just in case)
            serializable_results = [
                r for r in all_results_for_this_run if isinstance(r, dict)
            ]
            if not save_to_json(serializable_results, summary_file):
                logger.error(f"Failed to save run summary {summary_file}")
            else:
                logger.info(f"Run summary saved to {summary_file}")

            # --- Print Summary ---
            self._print_summary(all_results_for_this_run)
            logger.info(
                f"Total execution time for run: {time.time() - overall_start_time:.2f} seconds."
            )
            return all_results_for_this_run

        except Exception as e:
            logger.critical(
                f"FATAL UNEXPECTED ERROR running execution agent: {str(e)}",
                exc_info=True,
            )
            # Attempt final save on fatal error
            try:
                if hasattr(self, "state_manager") and self.state_manager:
                    self.state_manager.save_state()
                if hasattr(self, "metrics_manager") and self.metrics_manager:
                    self.metrics_manager.save_all_metrics()
                logger.info(
                    "Attempted state/metrics save during shutdown after fatal error."
                )
            except Exception as save_e:
                logger.error(
                    f"Failed state/metrics save during fatal shutdown: {save_e}"
                )
            raise  # Re-raise the original exception
