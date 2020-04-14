# RI project - Search engine on the Stanford collection

A search engine project using the collection provided by Stanford for their online course : Information Retrieval and Web Search (cs276)
A link to the course can be found [here](http://web.stanford.edu/class/cs276/), and the collection can be downloaded by following this [link](http://web.stanford.edu/class/cs276/pa/pa1-data.zip)

## Setup

We're using **Python 3.7+**.

Download the collection and extract it in the `data` directory, under a `collection directory` :
```
.
├── data
│   └── collection
|       ├── 0
|       ├── ...
|       └── 9
├── preprocess.py
...
```
install the necessary modules : `pip install -r requirements.txt`

You can also run `make install`.

## Testing

Run `make test`.

