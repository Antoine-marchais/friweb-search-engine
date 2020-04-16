from collections import OrderedDict

from config import PATH_DATA, PATH_STOP_WORDS
from preprocess import create_corpus_from_files, build_inverted_index, InvertedIndex

import pickle
import pytest

# only called once in the test file
@pytest.fixture(scope="module")
def corpus():
    return create_corpus_from_files(PATH_DATA, dev=True, dev_iter=100)

@pytest.mark.xfail(reason="data is not imported in the repo +  pickle doesn't load properly the data")
@pytest.mark.parametrize(
    "index_type",
    [1, 2, 3],
)
def test_build_dev_inverted_index_type(corpus, index_type):
    test_index: InvertedIndex = None
    built_index = build_inverted_index(corpus, PATH_STOP_WORDS, type_index=index_type)
    with open(f"tests/test_index_type{index_type}_100.pkl", "rb") as f:
        test_index = pickle.load(f)
    assert test_index == built_index

# TODO: we could optimize the test speed by lemmatizing the corpus once, and then creating the three different index