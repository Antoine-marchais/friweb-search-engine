from collections import OrderedDict

from config import PATH_STOP_WORDS
from preprocess import build_inverted_index, InvertedIndex
from mock_data import COLLECTION, INVERTED_INDEX_1, INVERTED_INDEX_2

import pickle
import pytest

@pytest.mark.xfail(reason="data is not imported in the repo +  pickle doesn't load properly the data")
def test_build_dev_inverted_index():
    index_type1 = build_inverted_index(COLLECTION, PATH_STOP_WORDS, type_index=1)
    assert INVERTED_INDEX_1 == index_type1

    index_type2 = build_inverted_index(COLLECTION, PATH_STOP_WORDS, type_index=2)
    assert INVERTED_INDEX_2 == index_type2

