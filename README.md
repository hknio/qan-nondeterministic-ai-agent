# Non-Deterministic Behavior Detection AI Agent

A sophisticated AI-powered tool designed to systematically identify and analyze non-deterministic behaviors in programs and APIs, particularly in modified Linux kernel environments. This tool leverages advanced Large Language Models (LLMs) to automatically generate, execute, and analyze test code across multiple programming languages.

## 🎯 Purpose

Non-deterministic behavior in software systems can lead to unpredictable results, making debugging difficult and potentially causing security vulnerabilities. This tool helps developers:

- **Identify** potential sources of non-determinism in their systems
- **Generate** comprehensive test cases to expose these behaviors
- **Execute** tests across multiple environments to detect variations
- **Analyze** results to understand the root causes of non-determinism

## 🏗️ Architecture Overview

The tool employs a sophisticated multi-stage pipeline:

1. **Planning Stage**: AI-driven analysis to identify potential non-deterministic behaviors
2. **Code Generation**: Automatic test code creation using LLMs
3. **Execution Stage**: Multi-server test execution using QAN (Quick Analysis Network)
4. **Analysis & Review**: Categorization and review of detected behaviors

## 🚀 Key Features

- **Multi-Language Support**: Test generation for 14+ programming languages including C, C++, Java, Python, Go, Rust, and more
- **AI-Powered Analysis**: Uses state-of-the-art LLMs for intelligent test planning and code generation
- **Automated Execution**: Fully automated test execution with error recovery and retry mechanisms
- **Non-Determinism Detection**: Specialized algorithms to detect variations across multiple test runs
- **Comprehensive Reporting**: Detailed metrics and categorization of detected behaviors
- **Containerized Testing**: Secure execution in isolated Linux Alpine containers with KVM virtualization

## 📋 Requirements

- Python 3.12+
- Poetry (for dependency management)
- API keys for supported LLM providers (OpenAI, Anthropic, Perplexity, DeepSeek, etc.)
- Access to QAN servers for test execution (configurable via environment)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd nondeterministic-agent
```

### 2. Install Dependencies

```bash
poetry install
```

### 3. Environment Configuration

Create a `.env` file in the project root with your configuration:

```bash
# API Keys
DEEPSEEK_API_KEY="your_deepseek_api_key"
OPENAI_API_KEY="your_openai_api_key"
PERPLEXITYAI_API_KEY="your_perplexity_api_key"

# Model names
PLANNING_MODEL="perplexity/r1-1776"
CODE_WRITER_MODEL="perplexity/r1-1776" 
CODE_FIXER_MODEL="perplexity/r1-1776"
FORCE_NON_DETERMINISM_MODEL="perplexity/r1-1776"
REVIEW_MODEL="perplexity/r1-1776"

API_SERVER_URLS="s1.server.com:8000,s2.server.com:8000,s3.server.com:8000"
```

### Model Compatibility

This tool leverages [LiteLLM](https://github.com/BerriAI/litellm) for model interactions, meaning you can use any model provider supported by LiteLLM, including:

- OpenAI
- Anthropic
- Perplexity AI
- DeepSeek
- Mistral AI
- Gemini (Google)
- Azure OpenAI
- And many more

Specify your preferred model in the .env file using the appropriate format required by LiteLLM (e.g., for Perplexity R1 "perplexity/r1-1776" or "anthropic/claude-3-7-sonnet-20250219", etc.).

---

## 📖 Usage Guide

The tool operates in three main stages:

### 1. Planning Stage - Identifying Non-Deterministic Behaviors

Run the planning script to analyze your target system:

```bash
python nondeterministic_agent/planning.py
```

#### Command Line Options:

```
--scope SCOPE_NAME    Specify a scope name for the analysis
-m, --model MODEL     Override the default planning model
--non-interactive     Run without interactive prompts (requires --scope)
```

#### Interactive Workflow:

1. **Describe Your Target**: When prompted, provide a description of the system or API you want to analyze. The AI will identify potential sources of non-determinism such as:
   - Race conditions and concurrency issues
   - Memory management variations
   - CPU cache and TLB effects
   - Timing-dependent operations
   - System call variations
   - Floating-point precision issues

2. **Review Generated Subjects**: The tool generates a comprehensive list of test subjects categorized by type (e.g., CPU microarchitecture, concurrency, filesystem, etc.)

3. **Customize Test Plans**: Review and optionally modify the generated test plans in:
   ```
   results/scope/<scope_name>/
   ```

### 2. Execution Stage - Generating and Running Tests

Execute the generated test plans for a specific programming language:

```bash
python nondeterministic_agent/execution.py -l <language>
```

#### Supported Languages:

- **Compiled Languages**: `c`, `cxx`, `csharp`, `golang`, `java`, `kotlin`, `rust`, `scala`
- **Interpreted Languages**: `javascript`, `perl`, `php`, `python`, `ruby`, `typescript`

#### Command Line Options:

```
-l, --language LANG      Required. Target programming language
--scope SCOPE_NAME       Specify which scope to execute (interactive if omitted)
-f, --force-restart      Force restart, ignoring saved execution state
-m, --model MODEL        Override the code generation model
--limit N                Limit to first N test plans
--languages              List all supported languages
```

#### Execution Process:

The tool automatically:

1. **Generates Test Code**: Creates language-specific implementations for each test plan
2. **Compiles Code**: For compiled languages, builds the executable
3. **Executes Tests**: Runs tests multiple times across different QAN servers
4. **Analyzes Results**: Compares outputs to detect non-deterministic behavior
5. **Auto-Fixes Errors**: Attempts to fix compilation or runtime errors
6. **Forces Non-Determinism**: If initial tests are deterministic, tries to introduce variations

#### Example Workflow:

```bash
# Run tests for Python
python nondeterministic_agent/execution.py -l python

# Run tests for C++ with custom model
python nondeterministic_agent/execution.py -l cxx -m "anthropic/claude-3-opus-20240219"

# Resume interrupted execution
python nondeterministic_agent/execution.py -l java

# Force restart for Rust tests
python nondeterministic_agent/execution.py -l rust -f
```

### 3. Review Stage - Analyzing Results

After execution, use the review tool to categorize and analyze detected behaviors:

```bash
python nondeterministic_agent/review.py [error_type]
```

Where `error_type` is either:
- `non_deterministic` (default) - Review non-deterministic behaviors
- `system_error` - Review system-level errors

The review process:
1. Scans execution results
2. Uses AI to categorize errors by type
3. Groups similar behaviors together
4. Copies relevant test files for easy analysis

## 📊 Output Structure

The tool generates comprehensive outputs organized as follows:

```
results/
├── scope/                     # Planning stage outputs
│   └── <scope_name>/         # Test subjects for each scope
├── plan/                     # Generated test plans
│   └── <scope_name>/        
│       └── test_plans.json
├── generated_code/           # Generated test programs
│   └── <language>/
│       └── <subject>/
│           └── test_*.ext
├── api_results/              # Raw execution results
│   └── <test_id>_<language>.json
├── metrics_<scope>_<language>.json  # Execution metrics
├── execution_state_<language>.json  # Execution progress tracking
└── reviewed/                 # Categorized results
    ├── non_deterministic/
    └── system_error/
```

## 🔧 Advanced Features

### Batch Execution

Run tests for all scopes or plans using the provided scripts:

```bash
# Execute all scopes
./scripts/run_all_scopes.sh

# Execute all plans for a specific scope
./scripts/run_all_plans.sh
```

### Custom Test Environments

The tool executes tests in a specialized environment:
- **Container**: Linux Alpine 5.10 virtualized with Hermit on x86 architecture using KVM
- **Resources**: Single CPU, 1024 MB RAM
- **Runtime Limit**: 5 seconds per test
- **Special Features**:
  - Consistent time function returns
  - Fixed randomness state
  - Disabled automatic preemption
  - No internet access

### Report Generation

Generate comprehensive reports of test results:

```bash
python scripts/generate_report.py
```

This creates a structured report directory with:
- Categorized non-deterministic behaviors
- System errors grouped by type
- Associated test code for each finding

## 🏗️ Architecture Details

![Non-Deterministic Agent Workflow](images/Mermaid%20Chart-2025-03-30-120212.svg)
*Figure: Visual representation of the non-deterministic agent workflow and test execution pipeline*

### Core Components

1. **Planning Agent** (`agents/planning_agent.py`): Analyzes subjects and generates test plans
2. **Execution Agent** (`agents/execution_agent.py`): Manages code generation and execution
3. **QAN Service** (`services/qan_service.py`): Handles multi-server test execution
4. **LLM Service** (`services/llm_service.py`): Interfaces with various AI models
5. **State Manager** (`managers/state_manager.py`): Tracks execution progress
6. **Metrics Manager** (`managers/metrics_manager.py`): Collects and analyzes results

### Test Execution Pipeline

1. **Code Generation**: AI creates test code based on plan specifications
2. **Precompilation**: Validates and preprocesses code
3. **Compilation**: Builds executables for compiled languages
4. **Multi-Server Execution**: Runs tests across multiple QAN servers
5. **Result Analysis**: Compares outputs to detect variations
6. **Auto-Recovery**: Attempts to fix errors and retry
7. **Non-Determinism Forcing**: Modifies code to expose hidden variations

## Acknowledgments

This tool leverages:
- [LiteLLM](https://github.com/BerriAI/litellm) for unified LLM access
- [Aider](https://github.com/paul-gauthier/aider) for AI-assisted code generation
- Various open-source libraries listed in `pyproject.toml`

## Contact

For questions, issues, or contributions:
- Author: Yuriy Babyak <yuriy.babyak@outlook.com>
- GitHub: https://github.com/yuriyward/
