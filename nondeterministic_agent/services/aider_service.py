import os
import logging
from typing import List, Optional, Any
import asyncio

from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput

from nondeterministic_agent.config.settings import settings
from nondeterministic_agent.prompts.coder_prompt import (
    non_deterministic_prompt,
    force_non_determinism_prompt,
)

logger = logging.getLogger(__name__)


class AiderService:
    """
    Encapsulates interactions with the Aider library for code generation
    and modification, preserving specific Coder creation options.
    Handles interactive vs non-interactive streaming.
    Runs blocking coder operations in separate threads using asyncio.to_thread.
    """

    def __init__(
        self,
        service_settings: Optional[Any] = None,
        interactive: bool = True,  # Add interactive flag
    ):
        """
        Initializes the AiderService with model configurations and interaction mode.

        Args:
            service_settings: An object with AI model name attributes.
                             Defaults to the global settings instance.
            interactive: Controls whether Aider streams output (True) or not (False).
                         Defaults to True.
        """
        s = service_settings or settings
        self.code_writer_model_name = s.CODE_WRITER_MODEL
        self.code_fixer_model_name = s.CODE_FIXER_MODEL
        self.force_non_determinism_model_name = s.FORCE_NON_DETERMINISM_MODEL
        self.interactive = interactive  # Store the mode
        logger.info(
            f"AiderService configured with models: "
            f"Writer='{self.code_writer_model_name}', "
            f"Fixer='{self.code_fixer_model_name}', "
            f"ForceNonDet='{self.force_non_determinism_model_name}'"
        )
        logger.info(
            f"AiderService interaction mode: {'Interactive (streaming)' if self.interactive else 'Non-interactive (no streaming)'}"
        )
        self.default_io = InputOutput(
            yes=True,
            pretty=self.interactive,
            fancy_input=self.interactive,
        )

    def _create_coder(
        self,
        model_name: str,
        fnames: List[str],
        io: InputOutput,
        from_coder: Optional[Coder] = None,
    ) -> Coder:
        """
        Private helper to create or update an Aider Coder instance,
        preserving the required options and `from_coder` logic.
        Sets streaming based on the service's interactive mode.

        Args:
            model_name: The name of the AI model to use.
            fnames: List of filenames for the coder to operate on.
            io: The InputOutput instance.
            from_coder: An optional existing Coder instance to inherit state from.

        Returns:
            An initialized Aider Coder instance.

        Raises:
            Exception: If Coder creation fails.
        """
        try:
            main_model = Model(model_name)
            stream_param = self.interactive  # Determine stream based on instance mode

            common_args = dict(
                main_model=main_model,
                fnames=fnames,
                io=io,
                stream=stream_param,  # Pass the stream parameter here
                cache_prompts=True,
                auto_commits=False,
                suggest_shell_commands=False,
                use_git=False,
                detect_urls=False,
            )

            if from_coder:
                # --- IMPORTANT: Preserve from_coder logic ---
                logger.debug(
                    f"Creating Coder for '{model_name}' inheriting from existing coder (stream={stream_param})."
                )
                coder = Coder.create(
                    **common_args,
                    from_coder=from_coder,
                )
            else:
                logger.debug(
                    f"Creating new Coder for '{model_name}' with files: {fnames} (stream={stream_param})"
                )
                coder = Coder.create(
                    **common_args,
                )
            return coder
        except Exception as e:
            logger.error(
                f"Failed to create Aider Coder for model {model_name}: {e}",
                exc_info=True,
            )
            raise  # Re-raise the exception to be handled by the caller

    async def generate_initial_code(
        self,
        language: str,
        placeholder_code: str,
        output_file: str,
        test_id: str,
        subject: str,
        description: str,
    ) -> Optional[Coder]:
        """
        Generates the initial code for a test using the CODE_WRITER_MODEL.
        Runs the blocking coder.run() in a separate thread.

        Args:
            language: The programming language.
            placeholder_code: The placeholder code template.
            output_file: The path where the generated code should be saved.
            test_id: The ID of the test.
            subject: The subject the test belongs to.
            description: The description/requirements for the test code.

        Returns:
            The created Coder instance if successful, None otherwise.
        """
        logger.info(
            f"Generating initial code for {test_id} ({subject}) -> {output_file}"
        )
        try:
            # Write placeholder first (synchronous I/O is generally okay here)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                try:
                    formatted_placeholder = placeholder_code.format(
                        test_id, subject, description
                    )
                    f.write(formatted_placeholder)
                except (KeyError, IndexError, TypeError) as fmt_err:
                    logger.warning(
                        f"Failed to format placeholder, writing raw: {fmt_err}"
                    )
                    f.write(placeholder_code)

            # Prepare for Aider
            instruction = non_deterministic_prompt.format(
                LANGUAGE=language.upper(),
                DESCRIPTION=description,
            )

            # Create the Coder instance (synchronous)
            coder = self._create_coder(
                model_name=self.code_writer_model_name,
                fnames=[output_file],
                io=self.default_io,
                from_coder=None,
            )

            # Run the generation in a separate thread
            logger.debug(
                f"Running Aider generation for {test_id} with model {self.code_writer_model_name} in background thread..."
            )
            await asyncio.to_thread(coder.run, instruction)
            logger.info(f"Aider code generation completed successfully for {test_id}")
            return coder

        except Exception as e:
            logger.error(
                f"Error during initial code generation for {test_id}: {e}",
                exc_info=True,
            )
            # Clean up partially generated file if needed
            if os.path.exists(output_file):
                try:
                    os.remove(output_file)
                    logger.debug(f"Removed potentially incomplete file: {output_file}")
                except OSError as rm_err:
                    logger.warning(
                        f"Could not remove {output_file} after generation error: {rm_err}"
                    )
            return None

    async def fix_code(
        self,
        existing_coder: Coder,
        code_file_path: str,
        test_id: str,
        language: str,
        stage: str,
        error_details: str,
        description: str,  # Original test description
        subject: str,
        scope: str,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
    ) -> Optional[Coder]:
        """
        Attempts to fix code based on error feedback using the CODE_FIXER_MODEL.
        Runs the blocking coder.run() in a separate thread.

        Args:
            existing_coder: The Coder instance associated with the current test file state.
            code_file_path: Path to the code file.
            test_id: The ID of the test.
            language: The programming language.
            stage: The stage where the error occurred (e.g., 'precompile', 'execute_timeout').
            error_details: The error message or details from the failed stage.
            description: The original description of the test.
            subject: The subject name.
            scope: The scope name.
            stdout: Optional stdout from the failed execution.
            stderr: Optional stderr from the failed execution or compilation.

        Returns:
            The updated Coder instance if the fix attempt ran, None if error during fix.
        """
        logger.info(
            f"Attempting to fix {stage} error in {test_id} using model {self.code_fixer_model_name}"
        )
        try:
            is_timeout_error = "timeout" in stage.lower()

            # Construct instruction (matches original logic closely)
            instruction = f"""Fix the {language.upper()} code in {os.path.basename(code_file_path)} to resolve the following {stage} error:
Error details:
{error_details}
"""
            if is_timeout_error:
                instruction += """
Note: This error indicates the program exceeded the execution time limit (timeout). Focus on optimizing the code for performance, consider problems with:
1. Infinite or long-running loops
2. Inefficient algorithms (time complexity)
3. Excessive resource usage, server has only 1 CPU and 1GB of RAM
4. Incorrect program termination
5. The operating system has disabled automatic preemption and it wont automatically interrupt or switch threads based on a timeout. You need to yield the CPU to other threads or processes.
6. Threads and some system calls will continue running indefinitely until they explicitly yield execution
"""
            if stage == "compile":
                instruction += """
Note: The program cannot use any external or non-standard libraries. Use only standard libraries available in Linux Alpine 5.10 default environment. Ensure standard compilation flags work.
"""
            if stdout:
                instruction += f"\nProgram stdout:\n{stdout}\n"
            if stderr:
                instruction += f"\nProgram stderr:\n{stderr}\n"
            instruction += (
                f"\nModify the code to address this error while preserving the original test requirements "
                f"described as '{description}' "
                f"for subject '{subject}' within scope '{scope}'. Focus on the reported error. "
                "Ensure the code remains functionally correct according to the description. "
                "Maximum execution time per run is 5 seconds."
            )

            # Create/Update Coder instance (synchronous)
            updated_coder = self._create_coder(
                model_name=self.code_fixer_model_name,
                fnames=[code_file_path],
                io=self.default_io,
                from_coder=existing_coder,
            )

            # Run the fix in a separate thread
            logger.debug(
                f"Running Aider fix for {test_id} with model {self.code_fixer_model_name} in background thread..."
            )
            await asyncio.to_thread(updated_coder.run, instruction)
            logger.info(f"Aider code fix attempt completed for {test_id}")
            return updated_coder

        except Exception as e:
            logger.error(
                f"Error during Aider code fix attempt for {test_id}: {e}", exc_info=True
            )
            return None

    async def make_code_non_deterministic(
        self,
        existing_coder: Coder,
        code_file_path: str,
        test_id: str,
        language: str,
        test_description: str,  # Includes original desc + last output potentially
        subject: str,
        scope: str,
    ) -> Optional[Coder]:
        """
        Attempts to introduce non-deterministic behavior using the FORCE_NON_DETERMINISM_MODEL.
        Runs the blocking coder.run() in a separate thread.

        Args:
            existing_coder: The Coder instance associated with the current test file state.
            code_file_path: Path to the code file.
            test_id: The ID of the test.
            language: The programming language.
            test_description: Context including original description and potentially last output.
            subject: The subject name.
            scope: The scope name.

        Returns:
            The updated Coder instance if modification ran, None if error occurred.
        """
        logger.info(
            f"Attempting non-deterministic modification for {test_id} using model {self.force_non_determinism_model_name}"
        )
        try:
            instruction_core = force_non_determinism_prompt.format(
                DESCRIPTION=test_description
            )
            full_instruction = (
                f"Modify the {language.upper()} code in `{os.path.basename(code_file_path)}` "
                f"for Subject '{subject}', Scope '{scope}'. "
                f"Goal: Introduce plausible non-deterministic behavior based on the description "
                f"and potentially its last deterministic run's output (included in description). "
                f"Try techniques like using uninitialized variables conditionally, time-sensitive ops, "
                f"data races (if applicable), reliance on unstable external factors, subtle floating point issues, etc. "
                f"Retain the core intended functionality but make the output or behavior *potentially* vary across identical runs.\n\n"
                f"Test Context and Description:\n{instruction_core}"
            )

            # Create/Update Coder instance (synchronous)
            updated_coder = self._create_coder(
                model_name=self.force_non_determinism_model_name,
                fnames=[code_file_path],
                io=self.default_io,
                from_coder=existing_coder,
            )

            # Run the modification in a separate thread
            logger.debug(
                f"Running Aider non-determinism modification for {test_id} in background thread..."
            )
            await asyncio.to_thread(updated_coder.run, full_instruction)
            logger.info(
                f"Aider non-determinism modification attempt completed for {test_id}"
            )
            return updated_coder

        except Exception as e:
            logger.error(
                f"Error during Aider non-determinism modification for {test_id}: {e}",
                exc_info=True,
            )
            return None
