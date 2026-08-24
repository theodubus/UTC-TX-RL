.PHONY: install install-dev run test lint clean

install:
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt

run:
	python main.py

test:
	python -m pytest tests/ -v

lint:
	flake8 src/ main.py --max-line-length=120 --ignore=E501

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -f outputs/agent.gif
