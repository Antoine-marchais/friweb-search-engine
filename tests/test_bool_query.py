from src.bool_query import *

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