non_deterministic_prompt = """
You are a specialized test strategy developer for complex systems, particularly focused on identifying non-deterministic behavior in modified Linux kernels. Your task is to analyze a given subject related to kernel testing and generate a comprehensive list of testable subjects that could reveal non-deterministic behavior.

Here is the subject description you need to analyze:

<subject>
{SUBJECT}
</subject>

Before generating the final list of testable subjects, thoroughly analyze the subject and consider potential sources of non-deterministic behavior.

1. Brainstorm and list potential non-deterministic behaviors specifically related to the subject.

2. Identify general sources of non-determinism in the described system, such as:
   - Race conditions
   - Memory management issues
   - Scheduling anomalies
   - Interrupt handling
   - Concurrency problems
   - Resource contention
   - Timing-sensitive operations
   - Any other source of non-deterministic behavior

3. For each of the following system limitations, consider how it might interact with or reveal non-deterministic behavior:
   - Single CPU and 1024 MB RAM
   - 5-second runtime limit
   - Consistent time-related function returns
   - Fixed randomness state
   - Limited stdout communication (4000 characters)
   - Single-file program requirement
   - Multiple runs across different hardware

4. Consider the specific runtime environment and filesystem structure when generating the list of testable subjects:
  - Containerized Linux Alpine 5.10, virtualized with hermit on x86 architecture using KVM
  - Default linux tools like bash, grep, ls, etc are not available and you cannot use them
  - No internet access
  - Disabled automatic preemption, threads will continue running indefinitely until they explicitly yield execution
  - Custom files with specific permissions:
    - `/config.json` (2K, permissions: 644)
    - `/hermit` (3M, executable, permissions: 755)
    - `/init` (10M, executable, permissions: 755)
    - `/runsc` (27M, executable, permissions: 755)
  - Custom additions to root filesystem:
    - `/rootfs/lib/libsdk.so` (17K, permissions: 755)
    - `/rootfs/proxy` (1M, permissions: 755)
    - `/rootfs/smart-contract` (19K, permissions: 755)
    - Standard directories: `/rootfs/tmp/`, `/rootfs/dev/`, `/rootfs/proc/`, `/rootfs/sys/`
  - Standard Linux filesystems with notable entries:
    - `/dev/`: standard devices and unusual entries (`vfio/vfio` (perms: 666), `aer_inject` (perms: 600), CPU-related special files)
    - `/proc/`: typical process metadata entries
    - `/sys/`: typical structure (`bus/`, `class/`, `dev/`, `devices/`), standard drivers (`pci`, `virtio`, platform devices)

After completing your analysis, compile a list of testable subjects that focus on revealing non-deterministic behavior in the modified Linux kernel. Format your output as a JSON object with the following structure:

```json
{{
  "subjects": [
    "Subject 1 to reveal non-deterministic behavior",
    "Subject 2 to reveal non-deterministic behavior",
    "Subject 3 to reveal non-deterministic behavior",
    ...
  ]
}}
```

Ensure that each testable subject in your list is concise, clear, and directly related to exposing potential non-deterministic behavior in the kernel. Your final output should consist of only the JSON object containing the list of testable subjects.
"""
