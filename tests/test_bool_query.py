from typing import Dict

from bool_query import *

import pytest

TEST_INVERTED_INDEX: Dict[str, int] = {
    "cat": [1, 2, 3],
    "dog": [1, 3, 5],
    "duck": [1, 5],
    "squid": [1, 2, 3, 6],
}

def test_lemmatization():

    assert lemmatize_query("cats or any dogs nand duck") == ["cat", "or", "dog", "nand", "duck"]

def test_postfix():
    assert query_to_postfix(["cat", "or", "dog", "and", "not", "duck"]) == ['cat', 'dog', 'duck', 'not', 'and', 'or']
    assert query_to_postfix(["cat", "or", "dog", "nand", "duck"]) == ['cat', 'dog', 'duck', "nand", 'or']

def test_list_merging():
    assert merge_or([1, 2, 3], [1, 3, 5]) == [1, 2, 3, 5]
    assert merge_or([1, 2, 5], [1, 3]) == [1, 2, 3, 5]
    assert merge_and([1, 2, 3], [1, 3, 5]) == [1, 3]
    assert merge_nand([1, 2, 3, 6], [1, 3, 5]) == [2, 6]

    assert boolean_operator_merge("or", [1, 2, 3], [1, 3, 5]) == [1, 2, 3, 5]
    assert boolean_operator_merge("or", [1, 2, 5], [1, 3]) == [1, 2, 3, 5]
    assert boolean_operator_merge("and", [1, 2, 3], [1, 3, 5]) == [1, 3]
    assert boolean_operator_merge("nand", [1, 2, 3, 6], [1, 3, 5]) == [2, 6]

def test_process_postfix_query():
    # this is like (cat or (dog nand duck))
    assert process_postfix_query(['cat', 'dog', 'duck', "nand", 'or'], TEST_INVERTED_INDEX) == [1, 2, 3]

    # squid nand (cat and dog)
    assert process_postfix_query(["squid", "cat", "dog", "and", "nand"], TEST_INVERTED_INDEX) == [2, 6]

    # lemu and (cat or dog)
    assert process_postfix_query(["lemu", "cat", "dog", "or", "and"], TEST_INVERTED_INDEX) == []

    # (cat or dog) and squid
    assert process_postfix_query(["cat", "dog", "or", "squid", "and"], TEST_INVERTED_INDEX) == [1, 2, 3]

    with pytest.raises(Exception):
        # missing a bool operator
        process_postfix_query(["cat", "dog", "lemu", "and"], TEST_INVERTED_INDEX)

    with pytest.raises(IndexError):
        # missing an operand
        process_postfix_query(["cat", "and"], TEST_INVERTED_INDEX)