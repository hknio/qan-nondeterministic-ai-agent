non_deterministic_prompt = """
You are an expert test engineer tasked with creating a comprehensive test plan to identify non-deterministic behavior in a Linux environment. Your goal is to develop clearly defined test cases that ensure complete coverage of all relevant functions and features for the given subject and task.

Please review the following inputs:

<subject>
{SUBJECT}
</subject>

<task>
{TASK}
</task>

<existing_test_cases>
{EXISTING_TEST_CASES}
</existing_test_cases>

Before creating the test cases, please analyze the inputs and plan your approach. Consider the following points:

1. Break down the subject and task, identifying key areas that need testing.
2. List and categorize potential sources of non-determinism in Linux relevant to the subject and task.
3. List the existing test cases and map them to the identified key areas, noting any gaps in coverage.
4. Brainstorm specific non-deterministic behaviors in Linux that could be tested. Focus only on Linux-specific sources of non-determinism, not on other potential sources (e.g., time, uninitialized variables) unless explicitly mentioned.
5. List potential test case types (positive scenarios, negative scenarios, boundary conditions, performance testing, and security testing if relevant) with descriptions of how they apply to this task.
6. Brainstorm edge cases and corner cases specific to the subject and task.

After your analysis, create a comprehensive set of test cases. Each test case should include:

1. A unique identifier (e.g., "test-linux-non-determinism")
2. A full description of the test case, including its purpose and any necessary execution steps

Important considerations:
- All test cases must be self-contained and not rely on any external dependencies or applications.
- The code will run in an Alpine Linux 5.10 environment with the following restrictions:
  - 1 CPU and 1024 MB of RAM
  - No network access
  - 5-second maximum runtime
  - Fresh container instance for each execution
  - Time-related functions always return the same value
  - System randomness set to the same state for each start
  - Output limited to 4000 characters on stdout
  - Program must be in a single file
  - Program cannot use any external and non standard libraries. It can use only standard libraries and Linux Alpine 5.10 default libraries.
  - System has disabled automatic preemption, threads will continue running indefinitely until they explicitly yield execution

IMPORTANT:
- Do not create code for any specific programing language.
- Each test case should include code that prints stdout results.
- Non-deterministic behavior will be checked by running the program multiple times and comparing the stdout results between runs. The test cases themselves should not perform this comparison.

Present your final test plan as a JSON object with the following structure:

```json
{{
  "tests": [
    {{
      "id": "test-example",
      "description": "Full description of the test case"
    }},
  ]
}}
```

Ensure consistency, clarity, and completeness in your documentation. Your final output should only include the JSON object containing the test cases placed in <result></result> tags. Do not include any additional text, explanations, or formatting outside of the JSON structure, and do not duplicate or rehash any of the work you did in the thinking block.
"""
