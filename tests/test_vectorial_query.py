import pytest
import vectorial_query as vq 
from mock_data import INVERTED_INDEX_2

def test_lemmatize():
    assert vq.lemmatize_query("this is a test query") == ["test", "query"]

def test_get_scores():
    # here we check that a test query gives the expected result
    test_query_1 = vq.lemmatize_query("scientific papers in the media")
    scores = vq.get_scores(test_query_1, INVERTED_INDEX_2, "tf_idf", "tf_idf_log_normalize")
    assert sorted(list(scores.items()), key=lambda score:score[1], reverse=True)[0][0] == 6

    # here we check that query results are ordered as expected
    test_query_2 = vq.lemmatize_query("dumb test query")
    scores = vq.get_scores(test_query_2, INVERTED_INDEX_2, "tf_idf", "tf_idf_log_normalize")
    assert scores[0] > scores[4]
