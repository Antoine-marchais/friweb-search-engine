# RI project - Search engine on the Stanford collection

A search engine project using the collection provided by Stanford for their online course : Information Retrieval and Web Search (cs276).
A link to the course can be found [here](http://web.stanford.edu/class/cs276/), and the collection can be downloaded by following this [link](http://web.stanford.edu/class/cs276/pa/pa1-data.zip).

The goal of this project, which is developped for the **Information Research for the Web** course, is to build a small search engine around the CS276 dataset. More specifically, it must construct an inverted index from the dataset, in order to be able to respond at queries on this database. We support two types of queries:

- boolean queries: cats OR (dogs AND duck)
- vectorial queries: cats dogs duck

The performance bottlenecks are important to monitor since the goal of this course is also to observe the challenges of the real scale of the web.

## Setup

We're using **Python 3.7+**, with some of the last data structures implemented (`dataclasses`).

As the dataset is heavy, it is not included in the repositery. Therefore, download the [collection](http://web.stanford.edu/class/cs276/pa/pa1-data.zip) and extract it in the `data` directory, under a `collection directory` :

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

**Caution:** We assume you're in a correct new *Virtual Environnement*, with `python` and `pip` refering to Python 3.7+.

To install the modules, you can simply run `make install`.

Or Install the necessary modules : `pip install -r requirements.txt`, and the minimal *nltk* modules : `python -m nltk.downloader popular`.

We specifically relie on two modules for this project :

- `nltk`, a natural language processing toolkit, used for tokenization and lemmatization. We use *wordnet*, so you'll need to download additional data.
- `ttable`, a library to work with Boolean expressionss

## Usage

TODO, with interface.py

## Dataset analysis ?

Notebook ?

## Preprocessing

### Files loading

We simply open all files in the `collection` folder, loading the content in memory into a dictionnary.

### Tokenization

The data set is already tokenized, but we still this process for queries.
We use the tokenization provided with `nltk`, which is more precise than a simple `split` on spaces, for numbers or `'s` for example.

### Stop words

**TODO**: some stop words analysis

We remove stop words before *lemmatization* but also after.

### Lemmatization

For a better quality of lemmatization, we use the *context* of a token in a sentence, with the `pos_tag` function in `nltk`, which can indicate the grammatical use of a word : *fly* can be either a verb or a noun. The *WordNet* lemmatizer can take as argument this categorisation to give a precise result.

### Inversed index construction

Building the inversed index consist on applying these differents steps to the whole collection, and then iterate over all documents and tokens, counting and storing the index in a dictionnary.

We support three index types:

- *document index*: For each term of the collection, it returns the ids of the documents in which the term appears.

```python
{"information": [1, 4]} # the term 'information' appears in docs 1 and 4
```

- *frequency index*: For each term of the collection, for each document in which the term appears, it returns both the id of the document and the term frequency of the term in the document.

```python
{"information": {1: 2, 4: 19}} # the term 'information' appears in docs 1 (2 times) and 4 (19 times)
```

- *position index*, For each term of the collection, for each document in which the term appears, it returns both the id of the document and the positions of the term in the document.

```python
{"information": {1: [1, 19], 4: [0, 2, 5]}} # the term 'information' appears in docs 1 (at position 1 and 19) and 4 (at position at 0, 2 and 5)
```

## Querying

### Loading the index

**TODO** the time it takes

### Boolean querying

If the request is entered as *boolean*, we therefore except that it is syntactically correct. We support three logical operator :

- **AND** : `term1 AND term2` will return documents containing both `term1` or `term2`
- **OR** : `term1 AND term2` will return documents containing `term1`, `term2` or both
- **NAND** : which means **AND NOT**, `term1 NAND term2` will return documents containing `term1` but not `term2`.

You can also use *parenthesis* for a better query expression. Otherwise, by default, `ttable` has the following default priorities (or *[precedence](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Operator_Precedence)*) for operators : `NAND > AND > OR`, ie `cats OR dogs NAND ducks AND squid` will be interpreted as `(cats OR ((dogs NAND ducks) AND squid))`

First, we tokenize, remove stop words and lemmatize the query, except for the *logical operators*. Then, we use `ttable` to transform the query into its *postfix syntax*, which will remove parenthesis and change the order of tokens and logical operator. It is then easy to process the query : read the postfix query sequentially. If you have a token, add the list of documents with this token (with the inverted index). Ortherwise, its a logical operator, apply a merge with this operator and the last two elements of the stack.

If you don't specify any logical operator, we assume it's an **AND** request : `cats dog duck` --> `cats AND dog AND duck`.

### Vectorial querying

## Testing

Run `make test`.
