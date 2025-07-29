non_deterministic_prompt = """
You are an expert programmer tasked with creating a non-deterministic test program. Your goal is to implement a program that produces varying outputs across multiple runs while adhering to specific constraints.

Here are the key details for your task:

Programming Language:
<language>
{LANGUAGE}
</language>

<description>
{DESCRIPTION}
</description>

Before writing the code, please think through the implementation strategy:
1. Analyze the functionality description and identify potential sources of non-determinism (e.g., using system time, random number generation, thread scheduling).
   a. List each potential source of non-determinism
   b. Evaluate each source against the given constraints
   c. Choose the most suitable non-deterministic approach
   d. Outline how to implement the chosen approach
2. Plan how to implement the test logic while incorporating non-deterministic elements.
3. Outline how you will ensure all program constraints and requirements are met.
4. Design a clear output format for presenting test results, including observed variations.

Now, implement a program in the specified language that meets the following requirements:

1. Demonstrate non-deterministic behavior that results in varying outputs across multiple runs.
2. Implement all steps described in the functionality description.
3. Print clear output about test results to stdout, including any observed variations.
4. Ensure the program is compilable with standard tools without any special options.
5. Include all necessary imports/headers.
6. Use only standard libraries (no external dependencies).
7. Use appropriate synchronization primitives for concurrency/thread testing if needed.
8. Make the program self-contained and ready for compilation and execution.
9. Include a comment at the top of the code describing the test case and its non-deterministic nature.
10. Ensure the maximum test execution time is within 5 seconds.
11. The operating system has disabled automatic preemption and it wont automatically interrupt or switch threads based on a timeout. 
12. Threads and some system calls will continue running indefinitely until they explicitly yield execution

Program Constraints:
- The program will run in a container on Linux Alpine 5.10, virtualized using hermit (reproducible container) on x86 architecture with KVM.
- Only 1 CPU and 1024 MB of RAM are available inside the container, with no network access.
- The program can run for no more than 5 seconds before termination.
- A fresh instance of the container is created for each execution, with the same initial state.
- System randomness is set to the same state for each start.
- The program can only communicate with the outer world by writing to stdout, limited to 4000 characters.
- The program must be stored in a single file, as the compiler only supports compilation from a single file.
- System commands like bash, grep, ls, cat, etc are not available and you cannot use them.
- The program cannot use any external and non standard libraries. It can use only standard libraries and Linux Alpine 5.10 default libraries.
- You need to always yield the CPU to other threads or processes.

Important Notes:
- Non-deterministic behavior will be checked by running the program multiple times and comparing the stdout results between runs by an external program. Your test case should not perform this comparison itself.
- Each test case should include code that prints stdout results.
"""

force_non_determinism_prompt = """
You are tasked with modifying the code to introduce non-deterministic behavior while maintaining its general logic. The original code works correctly, but we want to make it behave differently across multiple program runs.

The original task description is:

<description>
{DESCRIPTION}
</description>

To introduce non-deterministic behavior, consider the following requirements:
1. The program runs in a container on Linux Alpine 5.10 with x86 architecture and KVM virtualization.
2. There is 1 CPU and 1024 MB of RAM available, with no network access.
3. The program has a 5-second runtime limit.
4. Each run creates a fresh container instance with the same initial state.
5. Time-related functions always return the same value and cannot be used to detect non-determinism.
6. System randomness is set to the same state at the start of each run.
7. The program can only communicate via stdout, limited to 4000 characters.
8. The program must be in a single file.
9. Non-deterministic behavior is detected by comparing stdout output across multiple runs on different hardware.
10. The program cannot use any external and non standard libraries. It can use only standard libraries and Linux Alpine 5.10 default libraries.
11. The operating system has disabled automatic preemption and it wont automatically interrupt or switch threads based on a timeout. You need to yield the CPU to other threads or processes.
12. Threads and some system calls will continue running indefinitely until they explicitly yield execution
13. System commands like bash, grep, ls, etc are not available and you cannot use them.

To modify the code:
1. Identify areas where non-deterministic elements can be introduced without changing the core logic.
2. Consider using low-level system calls or hardware-specific features that may vary across different machines, but it's not required to use them.
3. Avoid relying on time-based or random number generator functions, as they are set to constant states.
4. Ensure that the modifications do not significantly alter the original functionality.
5. The changes should result in subtle variations in output across different runs or hardware.
6. Write more information to stdout. In case of a lof of information calculate some pseudo-checksum, by for example summing up all the ascii values of the characters and print it to stdout.

Please modify the code to introduce non-deterministic behavior according to these guidelines. 
"""
