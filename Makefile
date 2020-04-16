install:
	pip install -r requirements.txt
	python -m nltk.downloader popular
test:
	pytest