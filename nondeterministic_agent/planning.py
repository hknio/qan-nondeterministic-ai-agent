import argparse
import sys
import os
import logging
import asyncio
from typing import Optional

# Configuration and Utilities
from nondeterministic_agent.config.settings import settings
from nondeterministic_agent.utils.log_config import configure_logger
from nondeterministic_agent.utils.cli_helpers import (
    get_single_choice,
    get_yes_no_input,
    get_multiline_input,
)
from nondeterministic_agent.utils.string_utils import sanitize_filename

# Services
from nondeterministic_agent.services.llm_service import LLMService

# Agent
from nondeterministic_agent.agents.planning_agent import PlanningAgent

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="NonDeterministic Planner: Scope Analysis and Test Plan Generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scope",
        type=str,
        default=None,
        help="Specify a scope name. If exists, prompts for action. If not, starts analysis.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help=f"Override AI model for planning (Default from settings: '{settings.PLANNING_MODEL}').",
        metavar="MODEL_NAME",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable interactive prompts. Requires --scope. Errors if scope file doesn't exist.",
    )
    return parser.parse_args()


async def run_planning_workflow(
    agent: PlanningAgent, scope_name_arg: Optional[str], non_interactive: bool
):
    """Handles the interactive or non-interactive planning workflow."""
    logger = logging.getLogger(__name__)
    selected_scope_name = scope_name_arg
    scope_data = None

    # --- Step 0: Scope Selection / Loading ---
    if not selected_scope_name:
        if non_interactive:
            print("Error: --scope is required in non-interactive mode.")
            logger.critical("Scope missing in non-interactive mode.")
            sys.exit(1)

        # Interactive Scope Selection
        existing_scopes = agent.list_existing_scopes()  # list_existing_scopes is sync
        if existing_scopes:
            print("\nFound existing scopes:")
            choices = existing_scopes + ["Create a new scope", "Exit"]
            action = await get_single_choice("Select a scope or create new:", choices)
            if action == "Exit" or action is None:
                sys.exit(0)
            elif action == "Create a new scope":
                selected_scope_name = None
            else:
                selected_scope_name = action
        else:
            print("\nNo existing scopes found.")
            selected_scope_name = None

        if selected_scope_name is None:
            scope_input_name = await get_multiline_input(
                "Enter a name for the new scope:", default_value=""
            )
            # get_multiline_input returns None on cancel
            if scope_input_name is None:
                print("Operation cancelled.")
                sys.exit(0)
            if not scope_input_name.strip():
                print("Scope name cannot be empty.")
                sys.exit(1)
            selected_scope_name = scope_input_name.strip()
            logger.info(f"User provided new scope name: '{selected_scope_name}'")
            scope_data = None  # Ensure analysis runs for new scope
        else:  # User selected existing scope
            scope_data = agent.load_scope_analysis(selected_scope_name)  # load is sync
            if not scope_data:
                print(
                    f"Error loading existing scope '{selected_scope_name}'. Please check file or re-analyze."
                )
                reanalyze = await get_yes_no_input(
                    f"Re-analyze scope '{selected_scope_name}'?", default=False
                )
                if reanalyze:  # Returns True, False, or None (treat None as No/Cancel)
                    scope_data = None  # Trigger re-analysis
                else:
                    print("Exiting.")
                    sys.exit(1)
            else:
                print(f"\nLoaded existing scope: '{selected_scope_name}'")
                sub_choices = [
                    "Generate/Update test plans",
                    "Re-analyze scope (overwrite)",
                    "Exit",
                ]
                sub_action = await get_single_choice(
                    "Action for this scope?", sub_choices
                )
                if sub_action == sub_choices[1]:  # Re-analyze
                    scope_data = None  # Trigger re-analysis
                elif (
                    sub_action == sub_choices[2] or sub_action is None
                ):  # Exit or Cancel
                    print("Exiting.")
                    sys.exit(0)
                # Else: proceed to plan generation with loaded scope_data

    else:  # Scope name provided via argument
        logger.info(f"Processing specified scope: '{selected_scope_name}'")
        scope_data = agent.load_scope_analysis(selected_scope_name)  # load is sync
        if scope_data:
            print(f"\nLoaded existing scope: '{selected_scope_name}'")
            if non_interactive:
                pass  # Proceed directly to plan generation
            else:  # Ask interactively even if scope was passed via arg
                sub_choices = [
                    "Generate/Update test plans",
                    "Re-analyze scope (overwrite)",
                    "Exit",
                ]
                # --- CHANGE: Add await ---
                sub_action = await get_single_choice(
                    "Action for this scope?", sub_choices
                )
                if sub_action == sub_choices[1]:
                    scope_data = None
                elif sub_action == sub_choices[2] or sub_action is None:
                    print("Exiting.")
                    sys.exit(0)
                # Else: proceed to plan generation
        else:  # Scope provided via arg does not exist
            if non_interactive:
                print(
                    f"Error: Scope file for '{selected_scope_name}' not found in non-interactive mode."
                )
                logger.critical(
                    f"Scope file not found for '{selected_scope_name}' in non-interactive mode."
                )
                sys.exit(1)
            else:
                print(
                    f"Scope '{selected_scope_name}' not found or failed to load. Proceeding to analysis."
                )
                scope_data = None  # Ensure analysis runs

    # --- Step 1: Scope Analysis (if needed) ---
    if scope_data is None:
        if not selected_scope_name:
            logger.critical("Logic error: Scope name not set before analysis step.")
            sys.exit(1)
        if non_interactive:
            logger.critical(
                "Reached analysis step unexpectedly in non-interactive mode."
            )
            sys.exit(1)

        print(f"\n--- Scope Analysis for '{selected_scope_name}' ---")
        default_prompt_text = f"Analyze potential non-deterministic behavior within the '{selected_scope_name}' area of the Linux Kernel. Identify distinct sub-areas or concepts prone to such issues."
        subject_prompt = await get_multiline_input(
            f"Enter the detailed prompt/subject area to analyze for scope '{selected_scope_name}':",
            default_value=default_prompt_text,
        )
        # Handle cancellation from multiline input
        if subject_prompt is None:
            print("Analysis cancelled.")
            sys.exit(0)
        if not subject_prompt.strip():
            print("Analysis prompt cannot be empty.")
            sys.exit(1)

        try:
            print(
                f"\nStarting analysis for scope '{selected_scope_name}' (This may take a moment)..."
            )
            scope_data = agent.create_scope_analysis(
                selected_scope_name, subject_prompt
            )
            if scope_data is None:
                print(
                    f"Scope analysis failed for '{selected_scope_name}'. Check logs. Exiting."
                )
                sys.exit(1)

            print("\nAnalysis completed. Subjects identified:")
            subjects = scope_data.get("subjects", [])
            if subjects:
                for i, subject in enumerate(subjects):
                    print(f"  {i + 1}. {subject}")
            else:
                print("  (No subjects identified)")
        except Exception as e:
            logger.error(f"Error during scope analysis call: {str(e)}", exc_info=True)
            print(f"Error during analysis: {str(e)}")
            sys.exit(1)

    # --- Step 2: Test Plan Generation ---
    if not scope_data:
        print(
            "\nScope data is missing (analysis might have failed). Cannot proceed to plan generation."
        )
        sys.exit(1)

    if scope_data.get("subjects"):
        generate_plans = False
        proceed_with_generation = True  # Assume yes unless interactively changed

        # --- Resume Logic / Interaction Moved Here ---
        subjects = scope_data.get("subjects", [])
        sanitized_scope_dirname = sanitize_filename(selected_scope_name)  # Use util
        scope_plan_dir = os.path.join(settings.PLAN_DIR, sanitized_scope_dirname)
        subject_to_sanitized = {s: sanitize_filename(s) for s in subjects}  # Use util
        existing_files = {}
        pending_subjects_for_resume_check = []
        completed_subjects_count = 0
        if os.path.exists(scope_plan_dir):
            try:
                for filename in os.listdir(scope_plan_dir):
                    if filename.endswith(".yaml"):
                        basename = os.path.splitext(filename)[0]
                        existing_files[basename] = os.path.join(
                            scope_plan_dir, filename
                        )
            except OSError as e:
                logger.warning(f"Could not list existing plan files: {e}")

        for subject, sanitized_name in subject_to_sanitized.items():
            if sanitized_name in existing_files:
                completed_subjects_count += 1
            else:
                pending_subjects_for_resume_check.append(subject)

        remaining_subjects_count = len(pending_subjects_for_resume_check)
        total_subjects_count = len(subjects)

        if not non_interactive and completed_subjects_count > 0:
            print(
                f"\nFound {completed_subjects_count} existing test plans out of {total_subjects_count}."
            )
            if remaining_subjects_count > 0:
                print(f"Remaining subjects to process: {remaining_subjects_count}")
                gen_remaining = await get_yes_no_input(
                    f"Generate the remaining {remaining_subjects_count} test plans?",
                    default=True,
                )
                if not gen_remaining:  # Handles False and None (cancel)
                    print("Skipping test plan generation.")
                    proceed_with_generation = False
            else:
                print("All plans already seem to exist.")
                proceed_with_generation = False  # Don't run generation if all exist

        elif non_interactive:
            # In non-interactive mode, always proceed if there are subjects
            proceed_with_generation = True
        else:  # Interactive, no existing plans found
            gen_new = await get_yes_no_input(
                f"\nGenerate test plans for {total_subjects_count} subjects in '{selected_scope_name}'?",
                default=True,
            )
            if not gen_new:
                print("Skipping test plan generation.")
                proceed_with_generation = False
        # --- End Resume Logic / Interaction ---

        if proceed_with_generation:
            print(f"\n--- Generating Test Plans for Scope '{selected_scope_name}' ---")
            try:
                # Assuming agent.create_test_plans_for_scope is synchronous or handles its own async calls internally
                num_plans = agent.create_test_plans_for_scope(
                    selected_scope_name, scope_data
                )
                print(
                    f"\nFinished: {num_plans} plans were generated/updated (or found existing)."
                )
                print(f"Plan files saved under: {scope_plan_dir}")
            except Exception as e:
                logger.error(
                    f"Error during plan generation call: {str(e)}", exc_info=True
                )
                print(f"Error during plan generation: {str(e)}")
                sys.exit(1)
    else:
        print("\nNo subjects found in scope data. Cannot generate test plans.")

    print(f"\nPlanning process for scope '{selected_scope_name}' finished.")


async def main():
    """Main asynchronous entry point for planning script."""
    args = parse_arguments()
    configure_logger()
    logger = logging.getLogger(__name__)

    # --- Initialize Services ---
    logger.info("Initializing LLM service...")
    try:
        model_name = args.model or settings.PLANNING_MODEL
        llm_service = LLMService(default_model=model_name)
        print(f"Using Planning Model: '{model_name}'")
    except Exception as e:
        logger.critical(f"Failed to initialize LLMService: {e}", exc_info=True)
        print(f"\nInitialization Error (LLMService): {e}")
        sys.exit(1)

    # --- Initialize Agent ---
    try:
        # Pass settings explicitly if PlanningAgent needs it, otherwise it uses global
        agent = PlanningAgent(llm_service, settings)
    except Exception as e:
        logger.critical(f"Failed to initialize PlanningAgent: {e}", exc_info=True)
        print(f"\nInitialization Error (PlanningAgent): {e}")
        sys.exit(1)

    # --- Run Workflow ---
    try:
        await run_planning_workflow(agent, args.scope, args.non_interactive)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user (KeyboardInterrupt).")
        print("\nPlanning interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.critical(
            f"Unhandled exception during planning workflow: {e}", exc_info=True
        )
        print(f"\nAn unexpected critical error occurred: {e}")
        print("Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        # Handle nested loop error if running in env like Jupyter
        if "Cannot run the event loop while another loop is running" in str(e):
            logger.warning("Asyncio loop already running. Attempting direct await.")
            # This might need adjustment based on the specific environment
            loop = asyncio.get_event_loop()
            # Ensure loop is running if in specific context like notebook that might close it
            if not loop.is_running():
                loop.run_until_complete(main())
            else:
                # If loop is running, just create task? This is complex.
                # Simplest might be to just run the main coroutine if a loop exists
                # but this depends heavily on the environment's loop management.
                # The run_until_complete might be sufficient.
                logger.warning("Running loop detected, using loop.run_until_complete.")
                loop.run_until_complete(main())

        else:
            # Re-raise other RuntimeErrors
            raise e
