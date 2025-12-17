# Contributing to CLaRa vs RAG Demo

Thank you for your interest in contributing! This is an educational project designed to help developers understand the differences between CLaRa and Normal RAG approaches.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/clara-vs-rag-demo.git`
3. Install dependencies: `pip install -r requirements.txt`
4. Run tests: `pytest tests/ -v`

## Code Structure

- **`src/models.py`**: Model loading - modify here to add new models
- **`src/rag.py`**: RAG pipeline - modify here to change retrieval logic
- **`src/clara.py`**: CLaRa pipeline - modify here to adapt to CLaRa API changes
- **`src/utils.py`**: Utility functions - add helper functions here
- **`src/app.py`**: UI code - modify here to change the Gradio interface

## Adding Features

### Adding New Models

1. Add model loading code in `src/models.py`
2. Create a new pipeline file (e.g., `src/new_model.py`)
3. Add the option to `src/app.py` UI

### Improving Tests

Add new test cases to `tests/test_frontend.py` or create new test files following the naming convention `test_*.py`.

### Adding Examples

Add example documents to `examples/` directory with descriptive filenames.

## Code Style

- Follow PEP 8 style guide
- Use type hints where possible
- Add docstrings to functions
- Keep functions focused and small

## Testing

Before submitting a PR:
1. Run tests: `pytest tests/ -v`
2. Test the UI: `python app.py` and verify it works
3. Check for linting errors

## Submitting Changes

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Commit with clear messages: `git commit -m "Add feature: description"`
4. Push to your fork: `git push origin feature/your-feature-name`
5. Open a Pull Request

## Questions?

Feel free to open an issue for questions or discussions!

