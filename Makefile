doc:
	cd docs && make html

test: FORCE
	python3 test/test.py

FORCE:
