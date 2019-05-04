setup:
	python3 -m venv ./venv/

install:
	python -m pip install -r requirements.txt

doc:
	cd docs && make html

test: FORCE
	python3 test/test.py

FORCE:
