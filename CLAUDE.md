# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands
- **Lint**: `ruff check --fix --line-length=120 --select=I <path>`
- **Format**: `ruff format --line-length=120 <path>`
- **Test**: `python -m pytest <path_to_test_file>`
- **Run single test**: `python -m pytest path/to/test_file.py::test_function_name`
- **Run example**: `python examples/<example_file>.py`
- **Test Mesh agent**: `python mesh/tests/<agent_test_file>.py`
- **Install dependencies with uv**: `uv add <package_name>`

## Code Style Guidelines
- **Formatting**: 120-char line length, use Ruff formatter
- **Imports**: System imports first, then third-party, then local; use relative imports within packages
- **Naming**: snake_case for variables/functions, PascalCase for classes; agents named with `ServiceNameFeatureAgent`
- **Types**: Use type hints where possible
- **Error handling**: Use specific exceptions with try/except blocks; use loguru for logging
- **Async patterns**: Use asyncio and async/await syntax; properly manage resources with async context managers
- **Documentation**: Include docstrings for classes and functions

## Project Structure
- **agents/**: Base agent implementations
- **core/**: Framework components (LLM, search, tools, clients)
- **mesh/**: Heurist Mesh specialized agent implementations
- **clients/**: API client implementations
- **examples/**: Example code and tests

## Mesh Agent Development Guidelines
- **Base class**: Inherit from `mesh.mesh_agent.MeshAgent`
- **Agent naming**: Use descriptive PascalCase names ending with "Agent" (e.g., `AlloraPricePredictionAgent`)
- **Metadata**: Include complete metadata in `__init__` (name, version, author, description, inputs, outputs, etc.)
- **API access**: Store sensitive API keys in environment variables, never hardcode
- **Testing**: Create a test file in `mesh/tests/` with example input/output saved as YAML
- **Tools**: Define tools using `get_tool_schemas()` method
- **External APIs**: Use `@with_retry` and `@with_cache` decorators for network operations
- **Resource management**: Implement `__aenter__` and `__aexit__` for proper cleanup
- **Documentation**: Include examples and clear descriptions in metadata
