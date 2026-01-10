# Contributing to Chess Transformers

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/chess-transform.git
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e chess_seq/
   pip install -r requirements.txt
   ```

## Development Setup

### Running Tests

```bash
pytest tests/
```

### Code Style

- Follow PEP 8 guidelines
- Use descriptive variable names
- Add docstrings to functions and classes
- Keep functions focused and modular

## Making Changes

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit with clear messages:
   ```bash
   git commit -m "Add: brief description of changes"
   ```

3. Push to your fork and create a Pull Request

## Pull Request Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Ensure all tests pass
- Update documentation if needed

## Areas for Contribution

- **Bug fixes**: Check the issues tab for known bugs
- **Documentation**: Improve README, docstrings, or add tutorials
- **Features**: Length generalization, new architectures, RL improvements
- **Testing**: Add unit tests for uncovered code
- **Probing tools**: Interpretability research

## Questions?

Feel free to open an issue for questions or discussions about potential contributions.
