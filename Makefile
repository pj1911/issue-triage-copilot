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

inspect:
	$(PYTHON) -m src.data.inspect_schema

clean-labels:
	$(PYTHON) -m src.data.clean_labels

split:
	$(PYTHON) -m src.data.split

preprocess:
	$(PYTHON) -m src.data.preprocess
