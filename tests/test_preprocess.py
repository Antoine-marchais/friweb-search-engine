from src.config import PATH_DATA, PATH_STOP_WORDS
from src.preprocess import create_corpus_from_files, build_inverted_index

import pickle

def test_build_dev_inverted_index_type_1():
    corpus = create_corpus_from_files(PATH_DATA, dev=True, dev_iter=100)
    index = build_inverted_index(corpus, PATH_STOP_WORDS, type_index=1)
    
    test_index = dict()
    with open("tests/test_index_type1_100.pkl", "rb") as f:
        test_index = pickle.load(f)

    assert test_index == index