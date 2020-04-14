from collections import OrderedDict

from config import PATH_DATA, PATH_STOP_WORDS
from preprocess import create_corpus_from_files, build_inverted_index, InvertedIndex

import pickle
import pytest

@pytest.mark.xfail(reason="data is not imported in the repo +  pickle doesn't load properly the data")
def test_build_dev_inverted_index():
    
    corpus = create_corpus_from_files(PATH_DATA, dev=True, dev_iter=100)
    test_index: InvertedIndex = None

    index_type1 = build_inverted_index(corpus, PATH_STOP_WORDS, type_index=1)
    with open("tests/test_index_type1_100.pkl", "rb") as f:
        test_index = pickle.load(f)
    assert test_index == index_type1

    index_type2 = build_inverted_index(corpus, PATH_STOP_WORDS, type_index=2)
    with open("tests/test_index_type2_100.pkl", "rb") as f:
        test_index = pickle.load(f)
    assert test_index == index_type2

    index_type3 = build_inverted_index(corpus, PATH_STOP_WORDS, type_index=3)
    with open("tests/test_index_type3_100.pkl", "rb") as f:
        test_index = pickle.load(f)
    assert test_index == index_type3

