##############################
# Create a virtual environment
##############################
setup:
	python3 -m venv ./venv/

##############################
# Install dependencies
#
# You probably want to run `source venv/bin/activate` before this!
##############################
install:
	python -m pip install -r requirements.txt

##############################
# Build Sphinx documentation
##############################
doc:
	cd docs && make html

##############################
# Run unit tests
##############################
test: FORCE
	python3 test/test.py

FORCE:
