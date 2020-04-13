install:
	pip3 install -r requirements.txt

test:
	PYTHONPATH=. pytest