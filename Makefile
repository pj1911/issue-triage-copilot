PYTHON=.venv/bin/python
PIP=.venv/bin/pip

setup:
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

download-data:
	$(PYTHON) -m src.data.download

eda:
	$(PYTHON) -m src.data.eda
