# Publishing Checklist

1. Update version in `pyproject.toml`.
2. Ensure tests pass locally: `pytest`.
3. Build: `python -m build`.
4. Validate package metadata: `twine check dist/*`.
5. Publish to TestPyPI (recommended first):
   `twine upload --repository testpypi dist/*`
6. Publish to PyPI:
   `twine upload dist/*`
7. Create a GitHub release tag matching the package version.
