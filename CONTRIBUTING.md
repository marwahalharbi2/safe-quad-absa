# Contributing to SAFE

Thank you for your interest in contributing to SAFE!

## Development Setup

```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/safe-quad-absa.git
cd safe-quad-absa

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=safe_metric --cov-report=html

# Run specific test file
pytest tests/test_core.py -v
```

## Code Style

We use:
- **Black** for code formatting
- **Flake8** for linting

```bash
# Format code
black safe_metric/ scripts/ tests/

# Check linting
flake8 safe_metric/ scripts/ tests/
```

## Adding New Features

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Write tests for your feature in `tests/`
3. Implement your feature
4. Ensure all tests pass: `pytest`
5. Update documentation if needed
6. Submit a pull request

## Reporting Issues

Please use GitHub Issues to report:
- Bugs
- Feature requests
- Documentation improvements

Include:
- Python version
- SAFE version
- Steps to reproduce (for bugs)
- Expected vs actual behavior

## Questions?

Open a GitHub Discussion or email safe.eval.2026@gmail.com
