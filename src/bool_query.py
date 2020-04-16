import tt

from typing import List, Any, OrderedDict as OrdDict
from enum import Enum

from config import PATH_STOP_WORDS
from preprocess import tokenize_document, remove_stop_words_from_document, load_stop_words, lemmatize_document, Union

class LOGICAL_TOKENS(Enum):
    AND = "and"
    OR = "or"
    NAND = "nand"
    # NOT = "not"

LOGICAL_TOKENS_VALUES = [x.value for x in LOGICAL_TOKENS]


def lemmatize_query(query: str) -> List[str]:
    """lemmatize a single boolean query
    
    Arguments:
        query {str} -- base query string, as input by the user
    
    Returns:
        List[str] -- processed query, after lemmatization and removing stop words
    """
    
    tokens = tokenize_document(query)
    stop_words = load_stop_words(PATH_STOP_WORDS)

    # first pass
    tokens = remove_stop_words_from_document(tokens, stop_words, LOGICAL_TOKENS_VALUES)

    lemmatized_query = lemmatize_document(tokens)

    assert len(lemmatized_query) == len(tokens), Exception("lemmatization should not remove tokens")

    # put back logical operator in case they have been changed with lemmatization
    for idx, tok in enumerate(tokens):
        if tok in LOGICAL_TOKENS_VALUES:
            lemmatized_query[idx] = tok
       
    # second pass
    return remove_stop_words_from_document(lemmatized_query, stop_words, LOGICAL_TOKENS_VALUES)

def query_to_postfix(query: List[str]) -> List[str]:
    """transform a query into its postfix form
    
    Arguments:
        query {List[str]} -- processed query
    
    Returns:
        List[str] -- postfix query
    """    
    return tt.BooleanExpression(" ".join(query)).postfix_tokens

def merge_or(a: List[int], b: List[int]) -> List[int]:
    """Merge two list with an OR operation, which are sorted !
    
    Arguments:
        a {List[int]} -- left list
        b {List[int]} -- right list
    
    Returns:
        List[int] -- left OR right
    """

    res = []
    n = len(a)
    m = len(b)
    i, j = 0, 0
    while i < n and j < m:
        if a[i] == b[j]:
            res.append(a[i])
            i += 1
            j += 1
        else:
            if a[i] < b[j]:
                res.append(a[i])
                i += 1
            else:
                res.append(b[j])
                j += 1
    
    while i < n:
        res.append(a[i])
        i += 1
    
    while j < m:
        res.append(b[j])
        j += 1
    
    return res

def merge_and(a: List[int], b: List[int]) -> List[int]:
    """Merge two list with an AND operation, which are sorted !
    
    Arguments:
        a {List[int]} -- left list
        b {List[int]} -- right list
    
    Returns:
        List[int] -- left AND right
    """
    res = []
    n = len(a)
    m = len(b)
    i, j = 0, 0
    while i < n and j < m:
        if a[i] == b[j]:
            res.append(a[i])
            i += 1
            j += 1
        else:
            if a[i] < b[j]:
                i += 1
            else:
                j += 1
    return res

def merge_nand(a: List[int], b: List[int]) -> List[int]:
    """Merge two list with an NAND operation, which are sorted !
    
    Arguments:
        a {List[int]} -- left list
        b {List[int]} -- right list
    
    Returns:
        List[int] -- left NAND right (left AND (NOT right))
    """
    res = []
    n = len(a)
    m = len(b)
    i, j = 0, 0
    while i < n and j < m:
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            if a[i] < b[j]:
                res.append(a[i])
                i += 1
            else:
                j += 1

    
    while i < n:
        res.append(a[i])
        i += 1

    return res

def boolean_operator_merge(boolOperator: str, posting_term1: List[int], posting_term2: List[int]) ->  List[int]:
    """merge two list of int, given a logical operator
    
    Arguments:
        boolOperator {str} -- logical operator to use
        posting_term1 {List[int]} -- left list
        posting_term2 {List[int]} -- right list
    
    Raises:
        Exception: if the given boolOperator is not supported
    
    Returns:
        List[int] -- merged list = left OPERATOR right
    """
    try:
        logicalOperator = LOGICAL_TOKENS(boolOperator)
        if logicalOperator == LOGICAL_TOKENS.AND:
            return merge_and(posting_term1, posting_term2)
        elif logicalOperator == LOGICAL_TOKENS.OR:
            return merge_or(posting_term1, posting_term2)
        elif logicalOperator == LOGICAL_TOKENS.NAND:
            return merge_nand(posting_term1, posting_term2)
    except ValueError:
        raise Exception(f"unsupported BoolOperator: {boolOperator}")
    

def process_postfix_query(postfix_query: List[str], inverted_index: OrdDict[str, int]) -> List[int]:
    """get relevant documents ids from a postfix query
    
    Arguments:
        postfix_query {List[str]} -- query in postfix form
        inverted_index {OrdDict[str, int]} -- invertedIndex "index", ie InvertedIndex.index
    
    Returns:
        List[int] -- List of relevant document ids
    """
    relevant_docs_stack: List[List[int]] = []    
    for term in postfix_query:
        if term in LOGICAL_TOKENS_VALUES:
            op_2 = relevant_docs_stack.pop()
            op_1 = relevant_docs_stack.pop()
            relevant_docs_stack.append(boolean_operator_merge(term, op_1, op_2))
        else:
            if term in inverted_index:
                relevant_docs_stack.append(inverted_index[term])
            else: 
                relevant_docs_stack.append([])

    assert len(relevant_docs_stack) == 1, Exception(f"error while processing postfix query {postfix_query}. Should obtain a result of len 1 but got {relevant_docs_stack}")
    return relevant_docs_stack.pop()
