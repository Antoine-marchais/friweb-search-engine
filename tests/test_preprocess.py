from collections import OrderedDict

from config import PATH_STOP_WORDS
from preprocess import build_inverted_index, InvertedIndex
from mock_data import COLLECTION, get_index

import pickle
import pytest

@pytest.mark.parametrize(
    "index_type",
    [1, 2],
)
def test_build_dev_inverted_index(index_type):
    inverted_index = build_inverted_index(COLLECTION, PATH_STOP_WORDS, type_index=index_type)
    assert get_index(index_type).index == inverted_index.index
    assert get_index(index_type).itype == inverted_index.itype
    assert get_index(index_type).stats == inverted_index.stats
