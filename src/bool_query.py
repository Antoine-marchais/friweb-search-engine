import tt

from typing import List, Any

from src.config import PATH_STOP_WORDS
from src.preprocess import tokenize_document, remove_stop_words_from_document, load_stop_words, lemmatize_tok, get_lemmatizer

LOGICAL_TOKENS = ["and", "or", "not"]

def lemmatize_query(query: str) -> tt.ExpressionTreeNode:
    tokens = tokenize_document(query)
    stop_words = load_stop_words(PATH_STOP_WORDS)
    tokens = remove_stop_words_from_document(tokens, stop_words, LOGICAL_TOKENS)


    lemmatized_query = []
    lemmatizer = get_lemmatizer()
    for tok in tokens:
        if tok in LOGICAL_TOKENS:
            lemmatized_query.append(tok)
        else:
            lemmatized_query.append(lemmatize_tok(tok, lemmatizer))
    
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
