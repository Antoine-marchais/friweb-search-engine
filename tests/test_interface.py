from preprocess import InvertedIndex
from interface import retrieve_docs_from_bool_query

TEST_INVERTED_INDEX_TYPE1 = InvertedIndex(
    itype=1,
    index={
        "cat": {1: True, 2: True, 3: True},
        "dog": {1:True, 3:True, 5:True},
        "duck": {1:True, 5:True},
        "squid": {1:True, 2:True, 3:True, 6:True},
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
    assert retrieve_docs_from_bool_query("cats dog squid", TEST_INVERTED_INDEX_TYPE1, True) == ["Everything about animals", "Every animal but birds"]
    assert retrieve_docs_from_bool_query("cats lemu", TEST_INVERTED_INDEX_TYPE1, True) == []

def test_bool_query_with_operator():

    # priorities : NAND > AND > OR

    # this is like (cat or (dog nand duck))
    assert retrieve_docs_from_bool_query("cats or dogs nand duck", TEST_INVERTED_INDEX_TYPE1, True) == ["Everything about animals", "Why cats love seafood", "Every animal but birds"]
   
    # this is like (squid nand(cat and dog))
    assert retrieve_docs_from_bool_query("squid nand cat and dog", TEST_INVERTED_INDEX_TYPE1, True) == ["Why cats love seafood", "Vingt mille lieues sous les mers"]
    
    # this is like ((lemu and cat) or dog))
    assert retrieve_docs_from_bool_query("lemu and cat or dog", TEST_INVERTED_INDEX_TYPE1, True) == ["Everything about animals", "Every animal but birds","How to hunt with a dog"]
    
    # this is like (lemu and (cat or dog))
    assert retrieve_docs_from_bool_query("lemu and ( cat or dog )", TEST_INVERTED_INDEX_TYPE1, True) == []

    #  squid and ( cat or dog )
    assert retrieve_docs_from_bool_query("squid and ( cat or dog )", TEST_INVERTED_INDEX_TYPE1, True) == ["Everything about animals", "Why cats love seafood", "Every animal but birds"]