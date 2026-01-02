---
applyTo: '**'
---
Provide project context and coding guidelines that AI should follow when generating code, answering questions, or reviewing changes.

# Terminal Execution Rules

## Python Environment
- **ALWAYS Activate Virtual Environment**: Before running any Python script or pip command in the terminal, you MUST explicitly activate the virtual environment first.
- **Conda Environment**:
  - If you know the Conda environment name (e.g., from `environment.yml` or user instructions), use: `conda activate <env_name>`
  - If the environment name is unknown, **ASK the user** for the environment name before running code.
- **Chaining Commands**: Since terminal sessions might not persist state perfectly between tool calls depending on the environment, prefer chaining the activation command with the execution command.
  - **Conda (Windows)**: `conda activate <env_name> & python your_script.py` (Note: use `&` or `;` depending on shell)
  - **Conda (Linux/macOS)**: `source activate <env_name> && python your_script.py`
  - **Standard venv (Windows)**: `.\venv\Scripts\activate ; python your_script.py`
  - **Standard venv (Linux/macOS)**: `source venv/bin/activate && python your_script.py`
- If you are unsure of the virtual environment folder name (e.g., `venv`, `.venv`, `env`), check the directory structure first.