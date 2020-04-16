# RI project - Search engine on the Stanford collection

![](https://github.com/Antoine-marchais/friweb-search-engine/workflows/.github/workflows/tests.yml/badge.svg?branch=master)

A search engine project using the collection provided by Stanford for their online course : Information Retrieval and Web Search (cs276)
A link to the course can be found [here](http://web.stanford.edu/class/cs276/), and the collection can be downloaded by following this [link](http://web.stanford.edu/class/cs276/pa/pa1-data.zip)

## Setup

We're using **Python 3.7+**.

Download the collection and extract it in the `data` directory, under a `collection directory` :

```bash
.
├── data
│   └── collection
|       ├── 0
|       ├── ...
|       └── 9
├── preprocess.py
...
```

We assume you're in a correct new *Virtual Environnement*, with `python` launching a Python 3.7+.

You can run `make install`.

Or Install the necessary modules : `pip install -r requirements.txt`, and the minimal *nltk* modules : `python -m nltk.downloader popular`.

## Testing

Run `make test`.
