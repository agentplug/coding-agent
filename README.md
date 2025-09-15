# Coding Agent

**Version**: 1.0.0  
**Author**: agentplug  
**License**: MIT  

## Description

The Coding Agent generates Python code based on natural language prompts and can explain existing Python code. It uses AI to provide intelligent code generation and analysis capabilities.

## Methods

### `generate_code(prompt: str) -> str`

Generates Python code based on a natural language prompt.

**Parameters:**

- `prompt` (string, required): Natural language description of code to generate

**Returns:**

- Generated Python code as a string

**Example:**

```bash
python agent.py '{"method": "generate_code", "parameters": {"prompt": "Create a function that adds two numbers"}}'
```

### `explain_code(code: str) -> str`

Explains what a piece of Python code does.

**Parameters:**

- `code` (string, required): Python code to explain

**Returns:**

- Explanation of what the code does

**Example:**

```bash
python agent.py '{"method": "explain_code", "parameters": {"code": "def add_numbers(a, b): return a + b"}}'
```

## Dependencies

- `aisuite[openai]>=0.1.7` - AI service integration
- `python-dotenv>=1.0.0` - Environment variable management
- `docstring-parser>=0.17.0` - Required by aisuite

## Setup

1. **Create virtual environment:**

   ```bash
   uv venv .venv
   source .venv/bin/activate  # Unix/macOS
   # or .venv\Scripts\activate  # Windows
   ```

2. **Install dependencies:**

   ```bash
   uv pip install -e .
   ```

3. **Set up API key (optional):**

   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   # or create ~/.agenthub/.env file
   ```

4. **Configure the agent (optional):**
   Edit `config.json` to customize:
   - AI model selection
   - Temperature and other AI parameters
   - System prompts for different methods
   - Error messages

## Usage

The agent accepts JSON input via command line and returns JSON output:

```bash
# Activate virtual environment first
source .venv/bin/activate

# Generate code
python agent.py '{"method": "generate_code", "parameters": {"prompt": "Create a function that calculates factorial"}}'

# Explain code
python agent.py '{"method": "explain_code", "parameters": {"code": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"}}'
```

## Configuration

The agent can be configured via the `config.json` file. Here's the structure:

```json
{
  "ai": {
    "model": "openai:gpt-4.1",
    "temperature": 0.1,
    "max_tokens": null,
    "timeout": 30
  },
  "system_prompts": {
    "generate_code": "You are a Python code generator...",
    "explain_code": "You are a Python code explainer...",
    "validate_code": "You are a code validator..."
  },
  "error_messages": {
    "generate_code": "# Error generating code: {error}...",
    "explain_code": "Error explaining code: {error}...",
    "validate_code": "Error validating code: {error}..."
  }
}
```

### Configuration Options:

- **ai.model**: The AI model to use (e.g., "openai:gpt-4.1", "openai:gpt-3.5-turbo")
- **ai.temperature**: Controls randomness (0.0 to 1.0, lower = more deterministic)
- **ai.max_tokens**: Maximum tokens in response (null for no limit)
- **ai.timeout**: Request timeout in seconds
- **system_prompts**: Custom prompts for each method
- **error_messages**: Custom error messages with {error} placeholder

## Error Handling

The agent gracefully handles:

- Missing API keys (provides fallback responses)
- Invalid method names
- Missing parameters
- Network connectivity issues
- AI service errors
- Missing or invalid config.json (uses default configuration)

All errors are returned in JSON format with an `error` field.

## Tags

- code-generation
- python
- ai-assistant
