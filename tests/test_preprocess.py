from collections import OrderedDict

from config import PATH_STOP_WORDS
from preprocess import build_inverted_index, InvertedIndex
from mock_data import COLLECTION, get_index

import pickle
import pytest

@pytest.mark.xfail(reason="data is not imported in the repo +  pickle doesn't load properly the data")
@pytest.mark.parametrize(
    "index_type",
    [1, 2],
)
def test_build_dev_inverted_index(index_type):
    index_type1 = build_inverted_index(COLLECTION, PATH_STOP_WORDS, type_index=1)
    assert get_index(index_type) == index_type1