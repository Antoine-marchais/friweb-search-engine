from config import WEIGHT_DOCUMENT, WEIGHT_QUERY, PATH_STOP_WORDS
from collections import Counter
import math
from preprocess import InvertedIndex, StatCollection, lemmatize_document, tokenize_document, load_stop_words, remove_stop_words_from_document
from typing import Optional, Dict, List, Union, Tuple

def lemmatize_query(query: str) -> List[str]:
    tokens = tokenize_document(query)
    stop_words = load_stop_words(PATH_STOP_WORDS)
    tokens = remove_stop_words_from_document(tokens, stop_words, [])
    tokens = lemmatize_document(tokens)
    return tokens

def get_scores(
    query: List[str], 
    inverted_index: InvertedIndex, 
    weighting_scheme_document=WEIGHT_DOCUMENT, 
    weighting_scheme_query=WEIGHT_QUERY) -> Dict[int,float]:
    """
    compute score for all documents using the vectorial model with the config parameters
    
    Arguments:
        query {List[str]} -- preprocessed query 
        inverted_index {InvertedIndex} -- inverted index constructed on the collection
    
    Keyword Arguments:
        weighting_scheme_document {str} -- weighting scheme for the document (default: {WEIGHT_DOCUMENT})
        weighting_scheme_query {str} -- weighting scheme for the query (default: {WEIGHT_QUERY})
    
    Returns:
        Dict[int,float] -- similarity between the query and each document
    """
    scores = {}
    document_norms = {}
    query_norm = 0
    words = Counter(query)
    stats_collection = inverted_index.stats
    assert inverted_index.itype == 2
    frequency_index = inverted_index.index
    for term in words:
        if weighting_scheme_query == "binary":
            weight_query = 1
            
        #fall back to term frequency for the query
        else : 
            weight_query = words[term]
        query_norm += weight_query**2
        if term in frequency_index :
            for doc_ID, frequency in frequency_index[term].items():
                if weighting_scheme_document == "binary":
                    weight_document = 1
                elif weighting_scheme_document == "frequency":
                    weight_document = get_tf(term, doc_ID, frequency_index)
                elif weighting_scheme_document == "tf_idf_normalize":
                    tf_document = get_tf_normalize(term, doc_ID, frequency_index, stats_collection)
                    idf_document = get_idf(term, frequency_index, stats_collection.nb_docs)
                    weight_document = tf_document*idf_document
                elif weighting_scheme_document == "tf_idf_logarithmic":
                    tf_document = get_tf_logarithmique(term, doc_ID, frequency_index)
                    idf_document = get_idf(term, frequency_index, stats_collection.nb_docs)
                    weight_document = tf_document*idf_document
                
                #fall back to tf_idf_log_normalize
                else:
                    tf_document = get_tf_logarithme_normalize(term, doc_ID, frequency_index, stats_collection)
                    idf_document = get_idf(term, frequency_index, stats_collection.nb_docs)
                    weight_document = tf_document*idf_document
                if doc_ID in scores:
                    scores[doc_ID] += weight_query * weight_document
                    document_norms[doc_ID] += weight_document**2
                else:
                    scores[doc_ID] = weight_query * weight_document
                    document_norms[doc_ID] = weight_document**2
    for doc_ID in scores:
        scores[doc_ID] = scores[doc_ID]/(math.sqrt(query_norm)*math.sqrt(document_norms[doc_ID]))
    return scores

def get_tf(term: str, doc_ID: int, index_frequence: Dict[str, Dict[int, int]]) -> float:
    """
    returns the term frequency of a term in a document
    
    Arguments:
        term {str} -- term to search tf for
        doc_ID {int} -- id of the coresponding document
        index_frequence {Dict[str, Dict[int, int]]} -- frequency index
    
    Returns:
        float -- term frequency
    """
    postings_list = index_frequence[term].items()
    tf = 0
    for elt in postings_list:
        if elt[0] == doc_ID:
            tf = elt[1]
            break
    return tf

def get_tf_logarithmique(term: str ,doc_ID: int, index_frequence: Dict[str, Dict[int, int]]) -> float:
    """
    returns the logarithmic term frequency of a term in a document
    
    Arguments:
        term {str} -- term to search tf for
        doc_ID {int} -- id of the coresponding document
        index_frequence {Dict[str, Dict[int, int]]} -- frequency index
    
    Returns:
        float -- logarithmic term frequency
    """
    tf = get_tf(term, doc_ID, index_frequence)
    log_tf = 1 + math.log(tf) if tf > 0 else 0
    return log_tf

def get_tf_normalize(term: str ,doc_ID: int, index_frequence: Dict[str, Dict[int, int]], stats_collection: StatCollection) -> float:
    """
    returns the normalized term frequency of a term in a document
    
    Arguments:
        term {str} -- term to search tf for
        doc_ID {int} -- id of the coresponding document
        index_frequence {Dict[str, Dict[int, int]]} -- frequency index
        stats_collection {StatCollection} -- collection statistics
    
    Returns:
        float -- normalized term frequency
    """
    tf = get_tf(term, doc_ID, index_frequence)
    return tf/stats_collection.doc_stats[doc_ID]["freq_max"]

def get_tf_logarithme_normalize(term: str, doc_ID: int, index_frequence: Dict[str, Dict[int, int]], stats_collection: StatCollection) -> float:
    """
    returns the normalized logarithmic term frequency of a term in a document
    
    Arguments:
        term {str} -- term to search tf for
        doc_ID {int} -- id of the coresponding document
        index_frequence {Dict[str, Dict[int, int]]} -- frequency index
        stats_collection {StatCollection} -- collection statistics
    
    Returns:
        float -- normalized logarithmic term frequency
    """
    log_tf = get_tf_logarithmique(term, doc_ID, index_frequence)
    return log_tf/(1+math.log(stats_collection.doc_stats[doc_ID]["moy_freq"]))

def get_idf(term: str, index_frequence: Dict[str, Dict[int, int]], nb_doc: int) -> float:
    """
    returns the inverse document frequency of a term in a document
    
    Arguments:
        term {str} -- term to search tf for
        index_frequence {Dict[str, Dict[int, int]]} -- frequency index
        nb_doc {int} -- number of documents in the collection

    Returns:
        float -- inverse document frequency
    """
    df = len(index_frequence[term])
    return math.log(nb_doc/df)