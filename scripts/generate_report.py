#!/usr/bin/env python3

import os
import json
import shutil
import re
from pathlib import Path

# Import the sanitize_filename function from the nondeterministic_agent module
from nondeterministic_agent.planning import sanitize_filename

# Define the base directory paths
RESULTS_DIR = "results"
REPORT_DIR = "report"
NON_DETERMINISTIC_DIR = os.path.join(REPORT_DIR, "non_deterministic")
SYSTEM_ERROR_DIR = os.path.join(REPORT_DIR, "system_error")

def find_api_results_dir(base_dir=RESULTS_DIR):
    """Recursively search for the api_results directory."""
    print(f"Searching for API results directory in {base_dir}...")
    for root, dirs, _ in os.walk(base_dir):
        if "api_results" in dirs:
            api_results_dir = os.path.join(root, "api_results")
            print(f"Found API results directory: {api_results_dir}")
            return api_results_dir
    print("Warning: api_results directory not found in results.")
    return None

def find_tests_dir(base_dir=RESULTS_DIR):
    """Recursively search for the tests directory."""
    print(f"Searching for tests directory in {base_dir}...")
    for root, dirs, _ in os.walk(base_dir):
        if "tests" in dirs:
            tests_dir = os.path.join(root, "tests")
            print(f"Found tests directory: {tests_dir}")
            return tests_dir
        # Also look for 'plan' directory which might contain tests
        if "plan" in dirs:
            plan_dir = os.path.join(root, "plan")
            print(f"Found plan directory: {plan_dir}")
            return plan_dir
    print("Warning: tests directory not found in results.")
    return None

def find_test_file(scope, subject, test_id, tests_dir, language=None):
    """
    Find the test file based on scope, subject, and test ID.
    Recursively searches through the tests directory for matching files.
    """
    # Sanitize scope and subject to match directory naming
    sanitized_scope = sanitize_filename(scope)
    sanitized_subject = sanitize_filename(subject)
    
    # If we have language information, use it for more targeted search
    if language:
        # Try the most likely path with language subdirectory
        likely_lang_path = os.path.join(tests_dir, sanitized_scope, sanitized_subject, language)
        if os.path.exists(likely_lang_path):
            # Try looking for a file that has the test_id in its name
            print(f"Checking likely path with language: {likely_lang_path}")
            for file in os.listdir(likely_lang_path):
                if test_id in file:
                    file_path = os.path.join(likely_lang_path, file)
                    print(f"Found test file by name match: {file_path}")
                    return file_path
                
            # If not found by name, check file contents
            for file in os.listdir(likely_lang_path):
                file_path = os.path.join(likely_lang_path, file)
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            if any(pattern.format(test_id=test_id) in content for pattern in 
                                   [f"id: {test_id}", f"\"id\": \"{test_id}\"", 
                                    f"test_id: {test_id}", f"\"test_id\": \"{test_id}\""]):
                                print(f"Found test file by content match: {file_path}")
                                return file_path
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")
    
    # Try common language subdirectories under the subject folder
    likely_path = os.path.join(tests_dir, sanitized_scope, sanitized_subject)
    if os.path.exists(likely_path):
        print(f"Checking likely path: {likely_path}")
        
        # Check if there are language subdirectories
        for item in os.listdir(likely_path):
            lang_dir = os.path.join(likely_path, item)
            if os.path.isdir(lang_dir) and item.lower() in ['c', 'cpp', 'java', 'python', 'js', 'go', 'rust']:
                print(f"Found language directory: {lang_dir}")
                
                # Search in language directory for filename match
                for file in os.listdir(lang_dir):
                    if test_id in file:
                        file_path = os.path.join(lang_dir, file)
                        print(f"Found test file by name match: {file_path}")
                        return file_path
                
                # If not found by name, check file contents in language directory
                for file in os.listdir(lang_dir):
                    file_path = os.path.join(lang_dir, file)
                    if os.path.isfile(file_path):
                        try:
                            with open(file_path, 'r') as f:
                                content = f.read()
                                if any(pattern.format(test_id=test_id) in content for pattern in 
                                       [f"id: {test_id}", f"\"id\": \"{test_id}\"", 
                                        f"test_id: {test_id}", f"\"test_id\": \"{test_id}\""]):
                                    print(f"Found test file by content match: {file_path}")
                                    return file_path
                        except Exception as e:
                            print(f"Error reading file {file_path}: {e}")
        
        # Also check for test files directly in subject directory (no language subdirectory)
        for file in os.listdir(likely_path):
            file_path = os.path.join(likely_path, file)
            if os.path.isfile(file_path):
                # Check if test_id is in the filename
                if test_id in file:
                    print(f"Found test file by name match in subject dir: {file_path}")
                    return file_path
                
                # Also check file contents
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        if any(pattern.format(test_id=test_id) in content for pattern in 
                               [f"id: {test_id}", f"\"id\": \"{test_id}\"", 
                                f"test_id: {test_id}", f"\"test_id\": \"{test_id}\""]):
                            print(f"Found test file by content match in subject dir: {file_path}")
                            return file_path
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

    # If tests still not found, do a filename match across the entire tests directory
    print(f"Test not found in expected locations. Searching by filename pattern: '*{test_id}*'...")
    matched_files = []
    for root, _, files in os.walk(tests_dir):
        for file in files:
            if test_id in file:
                matched_files.append(os.path.join(root, file))
    
    if matched_files:
        print(f"Found {len(matched_files)} possible test files by name match:")
        for file in matched_files[:5]:  # Show first 5 matches
            print(f"  - {file}")
        
        # Return the first match
        print(f"Using the first matched file: {matched_files[0]}")
        return matched_files[0]
    
    # Last resort: wider search by content (only for files that are likely code)
    print(f"No filename matches found. Performing wider search for test ID in file contents: {test_id}...")
    for root, _, files in os.walk(tests_dir):
        for file in files:
            # Skip obvious non-test files
            if file.endswith(('.json', '.yaml', '.yml', '.md', '.txt')):
                continue
                
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    if any(pattern.format(test_id=test_id) in content for pattern in 
                           [f"id: {test_id}", f"\"id\": \"{test_id}\"", 
                            f"test_id: {test_id}", f"\"test_id\": \"{test_id}\""]):
                        print(f"Found test file during wider content search: {file_path}")
                        return file_path
            except Exception as e:
                # Silent for wide search to reduce noise
                pass
    
    print(f"Warning: No test file found for ID: {test_id}")
    return None

def get_test_id(data):
    """
    Extract test ID from data, handling different field names.
    Checks both 'id' and 'test_id' fields.
    """
    return data.get('test_id') or data.get('id')

def process_api_results():
    """
    Process JSON files from API results directory and organize them based on their status.
    """
    # First, locate the API results and tests directories
    api_results_dir = find_api_results_dir()
    tests_dir = find_tests_dir()
    
    if not api_results_dir:
        print("Error: Could not locate API results directory. Exiting.")
        return
    
    if not tests_dir:
        print("Warning: Could not locate tests directory. Will process JSON files but cannot copy test files.")
    
    # Create report directories if they don't exist
    os.makedirs(NON_DETERMINISTIC_DIR, exist_ok=True)
    os.makedirs(SYSTEM_ERROR_DIR, exist_ok=True)
    
    # Count variables for summary
    total_files = 0
    non_deterministic_count = 0
    system_error_count = 0
    test_files_found = 0
    skipped_files = 0
    
    # Recursively find all JSON files in the API results directory
    json_files = []
    for root, _, files in os.walk(api_results_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    if not json_files:
        print(f"No JSON files found in {api_results_dir} or its subdirectories.")
        return
    
    print(f"Found {len(json_files)} JSON files to process.")
    
    # Process each JSON file
    for json_path in json_files:
        filename = os.path.basename(json_path)
        total_files += 1
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Check if status is NON_DETERMINISTIC or SYSTEM_ERROR
            status = data.get('status')
            if status not in ['NON_DETERMINISTIC', 'SYSTEM_ERROR']:
                skipped_files += 1
                continue
            
            # Get test information - handle different field names
            test_id = get_test_id(data)
            scope = data.get('scope')
            subject = data.get('subject')
            language = data.get('language')  # Get language if it exists
            
            if not test_id or not scope or not subject:
                print(f"Warning: Missing required fields in {filename}")
                print(f"  test_id: {test_id}, scope: {scope}, subject: {bool(subject)}")
                continue
            
            # Determine target directory based on status
            target_dir = NON_DETERMINISTIC_DIR if status == 'NON_DETERMINISTIC' else SYSTEM_ERROR_DIR
            if status == 'NON_DETERMINISTIC':
                non_deterministic_count += 1
            else:
                system_error_count += 1
            
            # Copy the JSON file to the target directory
            target_json_path = os.path.join(target_dir, filename)
            shutil.copy2(json_path, target_json_path)
            print(f"Copied JSON file: {filename} to {target_dir}")
            
            # Find and copy the test file if tests directory was found
            if tests_dir:
                test_file_path = find_test_file(scope, subject, test_id, tests_dir, language)
                if test_file_path:
                    test_files_found += 1
                    # Get the extension from the test file
                    _, extension = os.path.splitext(test_file_path)
                    # Create the target path with the same name as the JSON but with the test file extension
                    base_name = os.path.splitext(filename)[0]
                    target_test_path = os.path.join(target_dir, f"{base_name}{extension}")
                    # Copy the test file
                    shutil.copy2(test_file_path, target_test_path)
                    print(f"Copied test file for {test_id} to {target_test_path}")
                else:
                    print(f"No test file found for {test_id}")
                    
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Print summary
    print("\nReport Generation Summary:")
    print(f"Total files processed: {total_files}")
    print(f"Files skipped (no matching status): {skipped_files}")
    print(f"Non-deterministic issues found: {non_deterministic_count}")
    print(f"System errors found: {system_error_count}")
    print(f"Test files found and copied: {test_files_found}")
    print(f"\nReports stored in:")
    print(f"- Non-deterministic: {os.path.abspath(NON_DETERMINISTIC_DIR)}")
    print(f"- System errors: {os.path.abspath(SYSTEM_ERROR_DIR)}")

if __name__ == "__main__":
    print("Generating report from API results...")
    process_api_results()
    print("Report generation completed.") 