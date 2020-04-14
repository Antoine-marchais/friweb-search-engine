import tt

from typing import List, Any, Dict
from enum import Enum

from config import PATH_STOP_WORDS
from preprocess import tokenize_document, remove_stop_words_from_document, load_stop_words, lemmatize_one_token, get_lemmatizer

class LOGICAL_TOKENS(Enum):
    AND = "and"
    OR = "or"
    NAND = "nand"
    # NOT = "not"

LOGICAL_TOKENS_VALUES = [x.value for x in LOGICAL_TOKENS]


def lemmatize_query(query: str) -> tt.ExpressionTreeNode:
    tokens = tokenize_document(query)
    stop_words = load_stop_words(PATH_STOP_WORDS)
    tokens = remove_stop_words_from_document(tokens, stop_words, LOGICAL_TOKENS_VALUES)


    lemmatized_query = []
    lemmatizer = get_lemmatizer()
    for tok in tokens:
        if tok in LOGICAL_TOKENS_VALUES:
            lemmatized_query.append(tok)
        else:
            lemmatized_query.append(lemmatize_one_token(tok, lemmatizer))
    
    return lemmatized_query

def query_to_postfix(query: List[str]) -> List[str]:
    return tt.BooleanExpression(" ".join(query)).postfix_tokens

def merge_or(a: List[int], b: List[int]) -> List[int]:
    """Merge two list with an OR operation, which are sorted !
    
    Arguments:
        a {List[int]} -- [description]
        b {List[int]} -- [description]
    
    Returns:
        List[int] -- [description]
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
        a {List[int]} -- [description]
        b {List[int]} -- [description]
    
    Returns:
        List[int] -- [description]
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
        a {List[int]} -- [description]
        b {List[int]} -- [description]
    
    Returns:
        List[int] -- [description]
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
    

def process_postfix_query(postfix_query: List[str], inverted_index: Dict[str, int]) -> List[int]:
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
