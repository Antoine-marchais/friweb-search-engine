install:
	pip3 install virtualenv
	virtualenv venv --python=python3
	. venv/bin/activate
	pip3 install -r requirements.txt

test:
	. venv/bin/activate
	PYTHONPATH=. pytest