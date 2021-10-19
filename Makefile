.PHONY: test install format

# Run tests for the library

test:
	poetry run python -m unittest discover tests/


# Install the project

install:
	poetry install
	# temporary fix: run this line to force installing nmslib from source
	poetry run pip install --force-reinstall --no-binary :all: nmslib


# format

format:
	poetry run pre-commit run --all-files
