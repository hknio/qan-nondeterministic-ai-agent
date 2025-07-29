import asyncio
import argparse
import sys
import logging
import signal

# Configuration and Utilities
from nondeterministic_agent.config.settings import settings
from nondeterministic_agent.utils.log_config import configure_logger
from nondeterministic_agent.utils.cli_helpers import (
    get_single_choice,
    list_available_scopes,
)

# Services
from nondeterministic_agent.services.qan_service import QANService
from nondeterministic_agent.services.aider_service import AiderService

# Managers
from nondeterministic_agent.managers.metrics_manager import MetricsManager
from nondeterministic_agent.managers.state_manager import StateManager

# Agent
from nondeterministic_agent.agents.execution_agent import ExecutionAgent

# Configure logger at the module level after imports
configure_logger()
logger = logging.getLogger(__name__)

# Flag to prevent recursive shutdown calls
shutdown_requested = False


def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="NonDeterministic Agent: Generate, execute, and analyze code using QAN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scope",
        type=str,
        default=None,
        help="Specify the planning scope name (directory name in results/plan/). Interactive if omitted.",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default=None,
        choices=settings.SUPPORTED_LANGUAGES,
        help="Specify the programming language. Interactive if omitted.",
        metavar="LANG",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Optionally, process only a single subject within the scope.",
        metavar="SUBJECT_NAME",
    )
    parser.add_argument(
        "-f",
        "--force-restart",
        action="store_true",
        help="Ignore saved execution state AND reset runtime metrics.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable interactive prompts. Requires --scope and --language. Disables Aider streaming.",
    )
    return parser.parse_args()


async def main():
    global shutdown_requested
    args = parse_arguments()

    loop = asyncio.get_running_loop()
    main_task = asyncio.current_task()

    def handle_interrupt():
        global shutdown_requested
        if not shutdown_requested:
            shutdown_requested = True
            logger.warning("Interrupt signal received. Initiating graceful shutdown...")
            print("\nInterrupt received, attempting graceful shutdown...")
            # Cancel the main task
            if main_task and not main_task.done():
                main_task.cancel()
            # Remove the handler so subsequent Ctrl+C raises KeyboardInterrupt directly
            # Allowing force quit if graceful shutdown hangs
            try:
                loop.remove_signal_handler(signal.SIGINT)
                loop.remove_signal_handler(signal.SIGTERM)  # Also handle SIGTERM
            except (NotImplementedError, ValueError):
                # Ignore if not supported or already removed
                pass
        else:
            logger.warning("Shutdown already in progress. Ignoring additional signal.")

    # Add the signal handlers
    try:
        loop.add_signal_handler(signal.SIGINT, handle_interrupt)
        loop.add_signal_handler(
            signal.SIGTERM, handle_interrupt
        )  # Handle TERM signal too
    except NotImplementedError:
        # Windows doesn't support add_signal_handler
        logger.warning(
            "Signal handlers not supported on this platform (likely Windows). Ctrl+C behavior may differ."
        )

    selected_scope = args.scope
    selected_language = args.language

    # --- Interactive Selection ---
    if not args.non_interactive:
        try:
            if selected_scope is None:
                available_scopes = list_available_scopes()
                if not available_scopes:
                    print(
                        f"\nError: No scopes found in plan directory ({settings.PLAN_DIR}). Run planning first."
                    )
                    sys.exit(1)
                print("\n--- Scope Selection ---")
                choices = available_scopes + ["Exit"]
                selected_scope = await get_single_choice(
                    "Choose scope:", choices, default=0 if available_scopes else None
                )
                if selected_scope is None or selected_scope == "Exit":
                    print("Operation cancelled.")
                    sys.exit(0)
                logger.info(f"User selected scope: '{selected_scope}'")

            if selected_language is None:
                print("\n--- Language Selection ---")
                choices = settings.SUPPORTED_LANGUAGES + ["Exit"]
                try:
                    default_lang_index = settings.SUPPORTED_LANGUAGES.index("python")
                except ValueError:
                    default_lang_index = 0 if settings.SUPPORTED_LANGUAGES else None
                selected_language = await get_single_choice(
                    "Choose language:", choices, default=default_lang_index
                )
                if selected_language is None or selected_language == "Exit":
                    print("Operation cancelled.")
                    sys.exit(0)
                logger.info(f"User selected language: '{selected_language}'")
        except asyncio.CancelledError:
            logger.warning("Shutdown requested during interactive prompts.")
            print("\nShutdown requested during setup.")
            return  # Exit main gracefully

    # --- Non-Interactive Validation ---
    if args.non_interactive:
        if not selected_scope:
            print("\nError: --scope is required in --non-interactive mode.")
            logger.critical("Scope missing in non-interactive mode.")
            sys.exit(1)
        if not selected_language:
            print("\nError: --language is required in --non-interactive mode.")
            logger.critical("Language missing in non-interactive mode.")
            sys.exit(1)
        logger.info("Running in non-interactive mode. Aider streaming disabled.")

    # Determine Aider interaction mode based on command-line flag
    is_interactive_mode = not args.non_interactive

    # --- Initialize Services and Managers ---
    logger.info("Initializing services and managers...")
    agent = None  # Initialize agent to None
    qan_service = None
    aider_service = None
    metrics_manager = None
    state_manager = None

    try:
        # Initialize components
        qan_service = QANService(settings)
        aider_service = AiderService(settings, interactive=is_interactive_mode)
        metrics_manager = MetricsManager(settings, selected_scope, selected_language)
        state_manager = StateManager(settings, selected_scope, selected_language)

        # --- Initialize and Run Agent ---
        print("\nStarting Execution Agent for:")
        print(f"  Scope    : '{selected_scope}'")
        print(f"  Language : {selected_language}")
        if args.subject:
            print(f"  Subject  : '{args.subject}' (Single subject)")
        if args.force_restart:
            print("  Mode     : Force Restart")
        if args.non_interactive:
            print("  Mode     : Non-interactive (Streaming disabled)")
        else:
            print("  Mode     : Interactive (Streaming enabled)")
        # Print model names used by AiderService
        print(
            f"  Models   : Writer='{aider_service.code_writer_model_name}', Fixer='{aider_service.code_fixer_model_name}', ForceNonDet='{aider_service.force_non_determinism_model_name}'"
        )

        agent = ExecutionAgent(
            scope_name=selected_scope,
            language=selected_language,
            qan_service=qan_service,
            aider_service=aider_service,
            metrics_manager=metrics_manager,
            state_manager=state_manager,
            force_restart=args.force_restart,
            specific_subject=args.subject,
        )

        logger.info("Starting agent run...")
        # Wrap the core agent run in a try/except CancelledError
        try:
            await agent.run()
            logger.info(
                f"Agent run completed normally for {selected_scope}/{selected_language}"
            )
            print("\nExecution finished.")
        except asyncio.CancelledError:
            logger.warning("Main agent task cancelled by interrupt.")
            print("\nExecution interrupted.")
            # The finally block below will handle saving

    except asyncio.CancelledError:
        # Catch cancellation that might happen during component initialization
        logger.warning("Shutdown requested during initialization.")
        print("\nShutdown requested during initialization.")
        # Fall through to finally block for cleanup
    except ValueError as ve:  # Catch init errors from Agent components
        logger.error(f"Initialization Error: {ve}", exc_info=True)
        print(f"\nConfiguration Error: {ve}")
        sys.exit(1)
    except Exception as e:
        logger.critical(
            f"Unhandled exception during agent execution: {e}", exc_info=True
        )
        print(f"\nAn unexpected critical error occurred: {e}")
        print("Check logs for details.")
        # Fall through to finally block for cleanup
        # Consider sys.exit(1) here if you want errors to halt immediately after saving
    finally:
        # This block runs on normal completion, exception, or cancellation
        logger.info("Entering finally block for cleanup and saving...")

        # Attempt final save if managers were initialized
        if state_manager or metrics_manager:
            print("Attempting to save progress...")
            try:
                if state_manager:
                    state_manager.save_state()
                if metrics_manager:
                    metrics_manager.save_all_metrics()
                print("Progress saved.")
                logger.info("State/metrics saved during cleanup.")
            except Exception as save_e:
                logger.error(f"Failed to save state/metrics during cleanup: {save_e}")
                print("Warning: Failed to save progress.")
        else:
            logger.info("Managers were not initialized, skipping save.")

        # Remove signal handlers explicitly before exiting main (good practice)
        try:
            loop.remove_signal_handler(signal.SIGINT)
            loop.remove_signal_handler(signal.SIGTERM)
        except (
            NotImplementedError,
            ValueError,
        ):  # ValueError if already removed or never added
            pass
        logger.info("Cleanup finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except asyncio.CancelledError:
        # This could happen if the loop itself is cancelled externally
        print("\nAsyncio loop execution cancelled.")
        logger.warning("Asyncio CancelledError caught at top level.")
        sys.exit(1)  # General error exit code
    except RuntimeError as e:
        if "Cannot run the event loop while another loop is running" in str(e):
            logger.warning("Asyncio loop already running. Attempting direct await.")
            # If running in an env like Jupyter where a loop exists
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule main as a task if loop is already running
                    # Note: This might not block script exit depending on environment
                    async def run_main_task():
                        await main()

                    loop.create_task(run_main_task())
                    # Need a way to wait for it or run forever if in interactive session
                    # This part is context-dependent and might need adjustments
                    print("Task scheduled on existing loop. Script might exit.")
                else:
                    # If loop exists but isn't running, run until complete
                    loop.run_until_complete(main())
            except Exception as loop_err:
                logger.error(
                    f"Error running main on existing loop: {loop_err}", exc_info=True
                )
                print(f"Error interacting with existing event loop: {loop_err}")
                sys.exit(1)

        else:
            logger.critical(f"Unhandled RuntimeError: {e}", exc_info=True)
            print(f"An unexpected runtime error occurred: {e}")
            sys.exit(1)  # General error exit code
    except Exception as e:
        logger.critical(f"Unhandled top-level exception: {e}", exc_info=True)
        print(f"An unexpected top-level error occurred: {e}")
        sys.exit(1)  # General error exit code
