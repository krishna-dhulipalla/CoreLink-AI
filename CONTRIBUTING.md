# Contributing to CoreLink AI

Thank you for your interest in contributing to CoreLink AI! We welcome contributions from everyone.

## How to Contribute

1.  **Fork the Repository:** Create a personal fork of the project on GitHub.
2.  **Clone the Fork:** Clone your fork to your local machine.
3.  **Install Dependencies:** Run `uv sync --extra test` to install runtime and test dependencies.
4.  **Create a Branch:** Create a new branch for your feature or bug fix.
5.  **Make Changes:** Implement your changes and ensure they follow the project's coding style.
6.  **Run Tests:** Ensure all unit tests pass by running `uv run pytest tests -q -p no:cacheprovider --ignore=tests/benchmarks`.
7.  **Submit a Pull Request:** Push your branch to your fork and submit a pull request to the main repository.

## Coding Standards

-   Use `uv` for dependency management.
-   Follow PEP 8 for Python code.
-   Ensure all new features include appropriate tests.
-   Update documentation for any significant changes.

## Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub.
