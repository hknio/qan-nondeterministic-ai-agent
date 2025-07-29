#!/usr/bin/env python3
"""
review.py

Review Agent script:
- Scans `results/api_results` for JSON test result files.
- Categorizes errors using an LLM.
- Groups errors into directories under `results/reviewed/<error_type>/<category>/`.
- Copies associated test files from `results/tests`.

Usage:
  python3 review.py [error_type]
  
  error_type: Optional. Either 'system_error' or 'non_deterministic' (default)
"""

import os
import sys
import yaml
import json
import shutil
import signal
import difflib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from nondeterministic_agent.services.llm_service import LLMService
from nondeterministic_agent.planning import sanitize_filename
from nondeterministic_agent.utils.log_config import configure_logger
from nondeterministic_agent.utils.string_utils import extract_result_content

logger = configure_logger(__name__)

LANGUAGE_DETAILS = {
    "typescript": {"extension": ".ts", "placeholder_module": "ts"},
    "scala": {"extension": ".scala", "placeholder_module": "scala"},
    "rust": {"extension": ".rs", "placeholder_module": "rust"},
    "ruby": {"extension": ".rb", "placeholder_module": "ruby"},
    "php": {"extension": ".php", "placeholder_module": "php"},
    "perl": {"extension": ".pl", "placeholder_module": "perl"},
    "kotlin": {"extension": ".kt", "placeholder_module": "kotlin"},
    "javascript": {"extension": ".js", "placeholder_module": "js"},
    "java": {"extension": ".java", "placeholder_module": "java"},
    "golang": {"extension": ".go", "placeholder_module": "go"},
    "cxx": {"extension": ".cpp", "placeholder_module": "cpp"},
    "csharp": {"extension": ".cs", "placeholder_module": "csharp"},
    "c": {"extension": ".c", "placeholder_module": "c"},
    "python": {"extension": ".py", "placeholder_module": "py"},
}

def find_test_file(scope: str, subject: str, test_id: str, tests_dir: str, language: Optional[str] = None) -> Optional[str]:
    """
    Attempt to locate a test file within tests_dir based on scope, subject, test_id, and optional language.
    Subdirectories are named after sanitized scope/subject, then possibly a folder matching the language key
    from LANGUAGE_DETAILS. Failing that, we look for a file whose name contains test_id or a file containing
    the test_id in its content.

    Returns the first matching file path or None if not found.
    """
    sanitized_scope = sanitize_filename(scope)
    sanitized_subject = sanitize_filename(subject)

    # If language is known, try that subfolder first
    if language and language.lower() in LANGUAGE_DETAILS:
        likely_lang_path = os.path.join(tests_dir, sanitized_scope, sanitized_subject, language.lower())
        if os.path.exists(likely_lang_path):
            # 1) Check direct filename
            for f in os.listdir(likely_lang_path):
                if test_id in f:
                    return os.path.join(likely_lang_path, f)
            # 2) Check file contents
            for f in os.listdir(likely_lang_path):
                fullp = os.path.join(likely_lang_path, f)
                if os.path.isfile(fullp):
                    try:
                        with open(fullp, "r") as ff:
                            cont = ff.read()
                            if any(p in cont for p in [
                                f"id: {test_id}",
                                f"\"id\": \"{test_id}\"",
                                f"test_id: {test_id}",
                                f"\"test_id\": \"{test_id}\""
                            ]):
                                return fullp
                    except:
                        pass

    # Fallback #1: Check scope/subject directory for language subfolders recognized by LANGUAGE_DETAILS
    sdir = os.path.join(tests_dir, sanitized_scope, sanitized_subject)
    if os.path.exists(sdir):
        for item in os.listdir(sdir):
            # If item is a subdirectory whose name is in LANGUAGE_DETAILS
            if os.path.isdir(os.path.join(sdir, item)) and item.lower() in LANGUAGE_DETAILS:
                sub_path = os.path.join(sdir, item)
                # a) check filename
                for f in os.listdir(sub_path):
                    if test_id in f:
                        return os.path.join(sub_path, f)
                # b) check file content
                for f in os.listdir(sub_path):
                    fp = os.path.join(sub_path, f)
                    if os.path.isfile(fp):
                        try:
                            with open(fp, "r") as ff:
                                cont = ff.read()
                                if any(p in cont for p in [
                                    f"id: {test_id}",
                                    f"\"id\": \"{test_id}\"",
                                    f"test_id: {test_id}",
                                    f"\"test_id\": \"{test_id}\""
                                ]):
                                    return fp
                        except:
                            pass

        # Fallback #2: Maybe no language subfolder, so check directly in the subject directory
        for f in os.listdir(sdir):
            fp = os.path.join(sdir, f)
            if test_id in f:
                return fp
            # Or check content
            if os.path.isfile(fp):
                try:
                    with open(fp, "r") as ff:
                        cont = ff.read()
                        if any(p in cont for p in [
                            f"id: {test_id}",
                            f"\"id\": \"{test_id}\"",
                            f"test_id: {test_id}",
                            f"\"test_id\": \"{test_id}\""
                        ]):
                            return fp
                except:
                    pass

    # Fallback #3: Do a filename match across entire tests_dir
    matches = []
    for root, _, files in os.walk(tests_dir):
        for fl in files:
            if test_id in fl:
                matches.append(os.path.join(root, fl))
    if matches:
        return matches[0]

    # Fallback #4: search by content in code files
    for root, _, files in os.walk(tests_dir):
        for fl in files:
            # Skip obvious non-code
            if fl.endswith((".json", ".yaml", ".yml", ".md", ".txt")):
                continue
            fp = os.path.join(root, fl)
            try:
                with open(fp, "r") as ff:
                    cont = ff.read()
                    if any(p in cont for p in [
                        f"id: {test_id}",
                        f"\"id\": \"{test_id}\"",
                        f"test_id: {test_id}",
                        f"\"test_id\": \"{test_id}\""
                    ]):
                        return fp
            except:
                pass
    return None

class ReviewAgent:
    """
    ReviewAgent:
    - Scans results/api_results for JSON test result files
    - For each error, queries an LLM to determine a category
    - Groups errors by category in results/reviewed/<error_type>/<category>/
    - Copies the associated .c (or other language) test file if found
    - Respects Ctrl+C to allow for partial progress saving
    """

    def __init__(self,
                tests_dir: str = "results/tests",
                reviewed_dir: str = "results/reviewed",
                error_type: str = "non_deterministic",
                model_name: Optional[str] = None):
        """
        :param tests_dir: Path to directory containing test source files.
        :param reviewed_dir: Directory to place categorized results and category info.
        :param error_type: Type of errors to process ('system_error' or 'non_deterministic').
        :param model_name: Model name override (if not provided, use default).
        """
        self.api_results_dir = "results/api_results"  # Hard-coded now, no param
        self.tests_dir = tests_dir
        self.reviewed_dir = reviewed_dir
        # Map between input error_type and JSON status
        error_type_map = {
            "system_error": "SYSTEM_ERROR",
            "non_deterministic": "NON_DETERMINISTIC"
        }
        self.error_type = error_type.lower()
        if self.error_type not in error_type_map:
            logger.warning(f"Invalid error_type: {error_type}, defaulting to 'non_deterministic'")
            self.error_type = "non_deterministic"
        self.error_status = error_type_map[self.error_type]
        
        # Create base reviewed directory
        os.makedirs(self.reviewed_dir, exist_ok=True)
        # Create error type specific directory
        self.error_type_dir = os.path.join(self.reviewed_dir, self.error_type)
        os.makedirs(self.error_type_dir, exist_ok=True)

        if model_name is None:
            # fallback to environment or default
            model_name = os.getenv("REVIEW_MODEL", "deepseek/deepseek-chat")
        self.llm = LLMService(default_model=model_name)

        # Load categories from directories
        self.categories = self._load_categories_from_dirs()
        self.processed_files = set()

        logger.info("ReviewAgent initialized.")
        logger.info(f"  - JSON results dir: {self.api_results_dir}")
        logger.info(f"  - Tests dir: {self.tests_dir}")
        logger.info(f"  - Reviewed results dir: {self.reviewed_dir}")
        logger.info(f"  - Error type: {self.error_type}")
        logger.info(f"  - Error status: {self.error_status}")
        logger.info(f"  - model: {model_name}")

    def _load_categories_from_dirs(self) -> List[Dict]:
        """
        Load categories by scanning the reviewed directory for subdirectories and reading 
        category-info.yaml files from each.
        Also loads the list of already processed files from processed_files.txt
        
        If a categories.yaml file exists in the error_type_dir with a non-empty list,
        those categories will be used exclusively instead of scanning for directories.
        """
        categories = []
        
        # Skip if error_type dir doesn't exist yet
        if not os.path.exists(self.error_type_dir):
            os.makedirs(self.error_type_dir, exist_ok=True)
            return categories
        
        # First check if there's a categories.yaml file with predefined categories
        categories_yaml_path = os.path.join(self.error_type_dir, "categories.yaml")
        if os.path.exists(categories_yaml_path):
            try:
                with open(categories_yaml_path, "r", encoding="utf-8") as f:
                    predefined_categories = yaml.safe_load(f)
                
                # If we have predefined categories, use those exclusively
                if predefined_categories and isinstance(predefined_categories, list) and len(predefined_categories) > 0:
                    logger.info(f"Using {len(predefined_categories)} predefined categories from {categories_yaml_path}")
                    
                    # Convert predefined categories into our expected format
                    for cat in predefined_categories:
                        if isinstance(cat, dict) and 'name' in cat:
                            # Create folder name from category name if it doesn't exist
                            folder_name = cat.get('folder', sanitize_filename(cat['name']))
                            cat_entry = {
                                "name": cat['name'],
                                "detail": cat.get('detail', ''),
                                "folder": folder_name
                            }
                            categories.append(cat_entry)
                            
                            # Make sure the folder exists
                            os.makedirs(os.path.join(self.error_type_dir, folder_name), exist_ok=True)
                    
                    # Load processed files and return the predefined categories
                    self._load_processed_files()
                    return categories
            except Exception as e:
                logger.warning(f"Failed to load predefined categories from {categories_yaml_path}: {e}")
                # Fall back to directory-based categories
        
        # If no valid predefined categories, load from directories as before
        logger.info(f"Loading categories from {self.error_type_dir} directories.")
            
        # Load processed files list if it exists
        self._load_processed_files()
            
        # Iterate through each directory in error_type_dir
        for item in os.listdir(self.error_type_dir):
            dir_path = os.path.join(self.error_type_dir, item)
            
            # Skip non-directories and special files/directories
            if not os.path.isdir(dir_path) or item.startswith('.'):
                continue
                
            # Look for category-info.yaml in each directory
            info_file = os.path.join(dir_path, "category-info.yaml")
            if os.path.exists(info_file):
                try:
                    with open(info_file, "r", encoding="utf-8") as f:
                        info = yaml.safe_load(f) or {}
                    
                    # Create category entry with folder name
                    category = {
                        "name": info.get("name", item),
                        "detail": info.get("detail", ""),
                        "folder": item
                    }
                    categories.append(category)
                except Exception as e:
                    logger.warning(f"Failed to load category info from {info_file}: {e}")
        
        logger.info(f"Loaded {len(categories)} categories from {self.error_type} directories.")
        return categories
        
    def _load_processed_files(self):
        """
        Load the list of processed files from a text file in the error_type directory
        """
        processed_files_path = os.path.join(self.error_type_dir, "processed_files.txt")
        if os.path.exists(processed_files_path):
            try:
                with open(processed_files_path, "r", encoding="utf-8") as f:
                    self.processed_files = set(line.strip() for line in f if line.strip())
                logger.info(f"Loaded {len(self.processed_files)} processed files from {processed_files_path}")
            except Exception as e:
                logger.warning(f"Failed to load processed files from {processed_files_path}: {e}")
                self.processed_files = set()
        else:
            self.processed_files = set()

    def _save_processed_files(self):
        """
        Save the list of processed files to a text file in the error_type directory
        """
        processed_files_path = os.path.join(self.error_type_dir, "processed_files.txt")
        try:
            with open(processed_files_path, "w", encoding="utf-8") as f:
                for filepath in sorted(self.processed_files):
                    f.write(f"{filepath}\n")
            logger.info(f"Saved {len(self.processed_files)} processed files to {processed_files_path}")
        except Exception as e:
            logger.error(f"Failed to save processed files: {e}")

    def run_review(self):
        logger.info(f"Starting review process for {self.error_type} errors.")
        
        # Refresh categories from directories to ensure we have the latest
        self.categories = self._load_categories_from_dirs()
        logger.info(f"Refreshed {len(self.categories)} categories from {self.error_type} directories.")
        
        # handle Ctrl+C
        original_handler = signal.getsignal(signal.SIGINT)

        def handle_sigint(sig, frame):
            logger.warning("SIGINT caught, saving processed files and exiting.")
            self._save_processed_files()
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_sigint)

        # find all JSON in results/api_results
        json_files = []
        for root, _, files in os.walk(self.api_results_dir):
            for fl in files:
                if fl.endswith(".json"):
                    json_files.append(os.path.join(root, fl))
        logger.info(f"Found {len(json_files)} JSON files in {self.api_results_dir}.")

        processed_count = 0
        for fpath in json_files:
            if fpath in self.processed_files:
                continue
            assigned = self._review_single_file(fpath)
            if assigned:
                processed_count += 1
            self.processed_files.add(fpath)

        logger.info(f"Review done. {processed_count} files assigned to categories.")
        self._save_processed_files()

        # restore old signal handler
        signal.signal(signal.SIGINT, original_handler)

    def _parse_differences(self, exec_results: List[str]) -> str:
        """
        Parse execution results and create a git-diff style display 
        showing what changed between the executions.
        
        Args:
            exec_results: List of execution result strings
        
        Returns:
            A string containing diff-style output showing differences
        """
        try:
            if not exec_results or not isinstance(exec_results, list):
                return "Error: Could not parse differences data"
            
            # We need at least two results to compare
            if len(exec_results) <= 1:
                return "No differences found to compare"
            
            # Take the first two execution results for comparison
            result1 = exec_results[0]
            result2 = exec_results[1]
            
            # Parse the stdout from both results
            # Extract just the stdout part if the strings are in ExecutionResult format
            stdout1 = result1
            stdout2 = result2
            
            stdout1_match = re.search(r"ExecutionResult\(stdout=(.*?), stderr=", result1, re.DOTALL)
            stdout2_match = re.search(r"ExecutionResult\(stdout=(.*?), stderr=", result2, re.DOTALL)
            
            if stdout1_match and stdout2_match:
                # Get the raw stdout content (which may still have \n escaped)
                stdout1 = stdout1_match.group(1)
                stdout2 = stdout2_match.group(1)
                
                # If these contain escaped newlines, replace them
                if "\\n" in stdout1:
                    stdout1 = stdout1.replace("\\n", "\n")
                if "\\n" in stdout2:
                    stdout2 = stdout2.replace("\\n", "\n")
            
            # First, split both outputs into lines
            lines1 = stdout1.splitlines()
            lines2 = stdout2.splitlines()
            
            # First try: build a colored diff for improved readability
            diff_output = "Diff between executions:\n"
            
            # Use difflib to create a unified diff
            diff_lines = list(difflib.unified_diff(
                lines1,
                lines2,
                fromfile='Execution 1',
                tofile='Execution 2',
                lineterm='',
                n=3  # Context lines
            ))
            
            # Format the diff output
            if diff_lines:
                # Count how many lines have actual changes (+ or -)
                change_lines = [line for line in diff_lines if line.startswith('+') or line.startswith('-')]
                
                # If there are a lot of changes, focus on showing just the differences
                if len(change_lines) > 20:
                    # Extract sections with changes (the line with the change and some context)
                    important_sections = []
                    current_section = []
                    in_change_section = False
                    
                    for line in diff_lines:
                        # Skip the file headers
                        if line.startswith('---') or line.startswith('+++'):
                            continue
                            
                        # If this is a hunk header or a change line, we're in a change section
                        if line.startswith('@@') or line.startswith('+') or line.startswith('-'):
                            in_change_section = True
                            
                        if in_change_section:
                            current_section.append(line)
                            
                        # If we're in a change section but hit an unchanged line, 
                        # add one line of trailing context then end the section
                        if in_change_section and not (line.startswith('@@') or line.startswith('+') or line.startswith('-')):
                            in_change_section = False
                            important_sections.append(current_section)
                            current_section = []
                    
                    # Add any remaining section
                    if current_section:
                        important_sections.append(current_section)
                    
                    # Combine the important sections into the diff output
                    for i, section in enumerate(important_sections):
                        if i > 0:
                            diff_output += "\n[...]\n"
                        diff_output += "\n".join(section)
                    
                    # Add a note about truncation
                    if len(important_sections) > 1:
                        diff_output += f"\n\n(Showing {len(important_sections)} key difference sections)"
                else:
                    # If not too many changes, show the full diff
                    diff_output += "\n".join(diff_lines)
            else:
                diff_output += "\nNo line differences found. This may indicate differences in whitespace or invisible characters.\n"
                
                # If no visible differences in the standard diff, let's check for non-printable/whitespace chars
                if stdout1 != stdout2:
                    diff_output += "\nDetected differences in characters:\n"
                    # Compare character by character, looking for first few differences
                    char_diffs = []
                    for i, (c1, c2) in enumerate(zip(stdout1, stdout2)):
                        if c1 != c2:
                            char_diffs.append(f"Position {i}: '{repr(c1)}' vs '{repr(c2)}'")
                            if len(char_diffs) >= 5:  # Limit to first 5 differences
                                char_diffs.append("...")
                                break
                    diff_output += "\n".join(char_diffs)
            
            return diff_output
        
        except Exception as e:
            logger.warning(f"Error creating diff visualization: {e}")
            return f"Error creating diff: {str(e)}"

    def _review_single_file(self, json_path: str) -> bool:
        """
        Parse the JSON, check if status matches the configured error_type,
        then ask LLM for category and copy to folder.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Could not parse JSON {json_path}: {e}")
            return False

        status = data.get("status", "")
        if status != self.error_status:
            # Not the type of error we want to categorize
            return False

        test_id = data.get("test_id", "")
        scope = data.get("scope", "")
        subject = data.get("subject", "")
        stage = data.get("stage", "")
        stderr = data.get("stderr", "")

        # Parse NonDeterministic stderr for differences visualization
        diff_output = ""
        if self.error_status == "NON_DETERMINISTIC" and stderr:
            try:
                # First attempt: Try to extract the ExecutionResult directly using regex
                pattern = r"(ExecutionResult\(stdout=.*?exit_code=\d+\))"
                exec_results = re.findall(pattern, stderr, re.DOTALL)
                
                # If that didn't work, try to extract from the differences section
                if not exec_results or len(exec_results) < 2:
                    # Try to extract the differences field
                    differences_match = re.search(r"differences=({.*?}),\s*message=", stderr, re.DOTALL)
                    if differences_match:
                        differences_str = differences_match.group(1)
                        # Try different patterns to extract execution results
                        
                        # Pattern 1: Look for 'ExecutionResult...': ['server1', 'server2']
                        pattern1 = r"'(ExecutionResult\(.*?exit_code=\d+\))': \[(.*?)\]"
                        matches = re.findall(pattern1, differences_str, re.DOTALL)
                        if matches:
                            exec_results = [key for key, _ in matches]
                
                # If we found execution results, process them
                if exec_results and len(exec_results) >= 2:
                    # Replace escaped newlines for better display
                    exec_results = [res.replace("\\n", "\n") for res in exec_results]
                    logger.info(f"Found {len(exec_results)} different execution results")
                    
                    # Generate diff between the first two results
                    diff_output = self._parse_differences(exec_results[:2])
                else:
                    logger.warning("Failed to extract execution results for comparison")
                    diff_output = "Error: Could not extract execution results for comparison"
            except Exception as e:
                logger.warning(f"Failed to parse differences in stderr: {e}")
                diff_output = f"Error: Failed to parse differences from stderr: {str(e)}"

        short_summary = (
            f"Status: {status}\n"
            f"Stderr: {stderr}\n"
        )
        
        # Add diff visualization if available
        if diff_output:
            short_summary += f"\nDiff Analysis:\n{diff_output}\n"

        # Check if we're using predefined categories
        using_predefined = self._is_using_predefined_categories()

        existing_cats_text = ""
        for cat in self.categories:
            existing_cats_text += f"- Category Name: {cat['name']}\n  Description: {cat.get('detail','')}\n\n"
        if not existing_cats_text.strip():
            existing_cats_text = "(none yet)"

        # Attempt to find test file
        language = data.get("language")
        test_path = None
        test_contents = ""
        if scope and subject and test_id:
            test_path = find_test_file(scope, subject, test_id, self.tests_dir, language)
            if test_path:
                try:
                    with open(test_path, "r", encoding="utf-8") as f:
                        test_contents = f.read()
                except Exception as e:
                    logger.warning(f"Could not read test file {test_path}: {e}")
        else:
            logger.warning(f"No scope, subject, or test_id for {json_path}")

        to_categorize = ""
        category_instructions = ""
        
        if self.error_status == "SYSTEM_ERROR":
            to_categorize = "You are reviewing a hermit error to assign it to an existing or new category. You need to find the root cause of the error as use it as a category name. Then you need to provide a detailed description of the category so other similar errors can be assigned to it."
            if using_predefined:
                category_instructions = "You MUST select one of the existing categories - DO NOT create new categories."
            else:
                category_instructions = "If it fits an existing category, re-use that exact category_name. Otherwise propose new. Keep as little unique categories as possible."
        elif self.error_status == "NON_DETERMINISTIC":
            to_categorize = "You are reviewing a nondeterministic error to assign it to an existing or new category. You need to find the root cause of the nondeterminism as use it as a category name. Then you need to provide a detailed description of the category so other similar errors can be assigned to it."
            if using_predefined:
                category_instructions = "You MUST select one of the existing categories - DO NOT create new categories."
            else:
                category_instructions = "If it fits an existing category, re-use that exact category_name. Otherwise propose new. Keep as little unique categories as possible."

        # Construct prompt
        prompt_msg = f"""
    You are a helpful assistant that reviews test results and assigns them to categories.
    {to_categorize}
    
    Error details:
    {short_summary}

    Test file contents:
    ```{language}
    {test_contents}
    ```

    Existing categories:
    {existing_cats_text}

    Before returning category_name and category_detail, review the error details and test file contents to understand the root cause of the error.

    Then return a JSON with:
    "category_name" : up to 60 chars
    "category_detail": up to 512 chars

    {category_instructions}

    Respond with json block:
    ```json
    {{
    "category_name": "...",
    "category_detail": "..."
    }}
    ```
    """
        
        # call LLM
        try:
            raw_response = self.llm.run_single_completion(prompt_msg, max_tokens=32000)
            extracted = extract_result_content(raw_response) or raw_response
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return False

        category_name = ""
        category_detail = ""
        try:
            parsed = json.loads(extracted.strip())
            category_name = parsed.get("category_name", "").strip()
            category_detail = parsed.get("category_detail", "").strip()
        except Exception as e:
            logger.warning(f"Couldn't parse response as JSON. Response:\n{extracted}\nErr: {e}")
            return False

        # Trim category name if needed
        if not category_name:
            logger.warning(f"No category name returned for file {json_path}")
            return False

        category_name = category_name[:60]
        category_detail = category_detail[:1024]

        # Check if we're using predefined categories
        using_predefined = self._is_using_predefined_categories()

        existing_cat = self._find_existing_category(category_name)
        if existing_cat is None:
            # If using predefined categories but got a non-matching category name
            if using_predefined:
                logger.warning(f"Category '{category_name}' doesn't match any predefined category")
                # Try to find a close match among predefined categories
                best_match = None
                for cat in self.categories:
                    # If the returned category name contains or is contained by a predefined category
                    if cat["name"].lower() in category_name.lower() or category_name.lower() in cat["name"].lower():
                        best_match = cat
                        break
                
                if best_match:
                    logger.info(f"Using closest predefined category match: '{best_match['name']}'")
                    existing_cat = best_match
                else:
                    # Use the first category as a fallback
                    if self.categories:
                        logger.info(f"Using first predefined category as fallback: '{self.categories[0]['name']}'")
                        existing_cat = self.categories[0]
                    else:
                        logger.error("No predefined categories available as fallback")
                        return False
            else:
                # Normal case for dynamic categories - create a new one
                folder_name = sanitize_filename(category_name)
                used_folders = [cat["folder"] for cat in self.categories]
                base_folder = folder_name
                c = 1
                while folder_name in used_folders:
                    folder_name = f"{base_folder}_{c}"
                    c += 1
                new_cat = {"name": category_name, "detail": category_detail, "folder": folder_name}
                self.categories.append(new_cat)
                logger.info(f"Created new category: {category_name}")
                cat_dir = os.path.join(self.error_type_dir, folder_name)
                os.makedirs(cat_dir, exist_ok=True)
                # write category-info.yaml
                info_path = os.path.join(cat_dir, "category-info.yaml")
                info_data = {"name": category_name, "detail": category_detail}
                try:
                    with open(info_path, "w", encoding="utf-8") as ff:
                        yaml.dump(info_data, ff, default_flow_style=False, allow_unicode=True, sort_keys=False)
                except Exception as ee:
                    logger.error(f"Error writing category-info.yaml: {ee}")
                existing_cat = new_cat
        
        # At this point, existing_cat should always be set
        # Use it to determine the folder and copy files
        assigned_folder = os.path.join(self.error_type_dir, existing_cat["folder"])
        logger.info(f"Assigned to category '{existing_cat['name']}'")

        # Copy JSON to that category folder
        base_json_name = os.path.basename(json_path)
        target_json = os.path.join(assigned_folder, base_json_name)
        try:
            shutil.copy2(json_path, target_json)
        except Exception as e:
            logger.error(f"Failed to copy JSON to category folder: {e}")

        if test_path and os.path.isfile(test_path):
            # Extract the original extension from the test file
            test_extension = os.path.splitext(test_path)[1]
            # Use the same base name as the JSON file but keep the original extension
            json_base_no_ext = os.path.splitext(base_json_name)[0]
            test_target_name = f"{json_base_no_ext}{test_extension}"
            target_test = os.path.join(assigned_folder, test_target_name)
            try:
                shutil.copy2(test_path, target_test)
                logger.info(f"Copied test file {test_path} -> {target_test}")
            except Exception as e:
                logger.error(f"Failed to copy test file: {e}")
        else:
            logger.info(f"No matching test file found for test_id={test_id}, scope={scope}, subject={subject}")

        return True

    def _find_existing_category(self, name: str) -> Optional[Dict]:
        for cat in self.categories:
            if cat["name"].lower() == name.lower():
                return cat
        return None
        
    def _is_using_predefined_categories(self) -> bool:
        """
        Check if we're using predefined categories from categories.yaml
        """
        categories_yaml_path = os.path.join(self.error_type_dir, "categories.yaml")
        if os.path.exists(categories_yaml_path):
            try:
                with open(categories_yaml_path, "r", encoding="utf-8") as f:
                    yaml_cats = yaml.safe_load(f)
                if yaml_cats and isinstance(yaml_cats, list) and len(yaml_cats) > 0:
                    return True
            except Exception:
                pass
        return False

# Entry point for script execution
def main():
    # Parse command line arguments
    error_type = "non_deterministic"  # Default error type
    if len(sys.argv) > 1:
        arg_error_type = sys.argv[1].lower()
        if arg_error_type in ["system_error", "non_deterministic"]:
            error_type = arg_error_type
        else:
            print(f"Invalid error type: {arg_error_type}")
            print("Valid options: system_error, non_deterministic")
            print("Defaulting to non_deterministic")
    
    logger.info(f"Launching ReviewAgent for {error_type} errors.")
    # ReviewAgent now loads categories directly from reviewed directories
    agent = ReviewAgent(
        tests_dir="results/tests",
        reviewed_dir="results/reviewed",
        error_type=error_type,
        model_name=None,
    )
    agent.run_review()


if __name__ == "__main__":
    main()
