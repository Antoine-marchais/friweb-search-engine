from preprocess import InvertedIndex
from interface import retrieve_docs_from_bool_query

TEST_INVERTED_INDEX_1 = InvertedIndex(
    itype=1,
    index={
        "cat": [1, 2, 3],
        "dog": [1, 3, 5],
        "duck": [1, 5],
        "squid": [1, 2, 3, 6],
        },
    mapping={
        1: "Everything about animals",
        2: "Why cats love seafood",
        3: "Every animal but birds",
        4: "The trial - Kafka",
        5: "How to hunt with a dog",
        6: "Vingt mille lieues sous les mers"
    },
    stats=None
)

def test_bool_query_no_operator():
    assert retrieve_docs_from_bool_query("cats dog squid", TEST_INVERTED_INDEX_1) == ["Everything about animals", "Every animal but birds"]
    assert retrieve_docs_from_bool_query("cats lemu", TEST_INVERTED_INDEX_1) == []

def test_bool_query_with_operator():

    # priorities : NAND > AND > OR

    # this is like (cat or (dog nand duck))
    assert retrieve_docs_from_bool_query("cats or dogs nand duck", TEST_INVERTED_INDEX_1) == ["Everything about animals", "Why cats love seafood", "Every animal but birds"]
   
    # this is like (squid nand(cat and dog))
    assert retrieve_docs_from_bool_query("squid nand cat and dog", TEST_INVERTED_INDEX_1) == ["Why cats love seafood", "Vingt mille lieues sous les mers"]
    
    # this is like ((lemu and cat) or dog))
    assert retrieve_docs_from_bool_query("lemu and cat or dog", TEST_INVERTED_INDEX_1) == ["Everything about animals", "Every animal but birds","How to hunt with a dog"]
    
    # this is like (lemu and (cat or dog))
    assert retrieve_docs_from_bool_query("lemu and ( cat or dog )", TEST_INVERTED_INDEX_1) == []

    #  squid and ( cat or dog )
    assert retrieve_docs_from_bool_query("squid and ( cat or dog )", TEST_INVERTED_INDEX_1) == ["Everything about animals", "Why cats love seafood", "Every animal but birds"]