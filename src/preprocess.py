import os
import pickle as pkl

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple, Any, OrderedDict as OrdDict

from config import PATH_DATA, DEV_MODE, DEV_ITER, PATH_INDEX, PATH_STOP_WORDS, PATH_DATA_BIN
from nltk import pos_tag
from nltk.stem.api import StemmerI
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize


def create_corpus_from_files(path: str, dev: bool =False, dev_iter: Optional[int]=None) -> OrdDict[str, str]:
    """Read file and import tokens into a new collection
    
    Arguments:
        path {string} -- path of the collection
    
    Keyword Arguments:
        dev {bool} -- activate dev mode (default: {False})
        dev_iter {Optional[int]} -- number of iterations in dev mode (default: {None})
    
    Returns:
        Dict[str, str] -- The loaded corpus
    """
    corpus = OrderedDict()
    for n_dir in os.listdir(path):
        dir_path = os.path.join(path, n_dir)
        index_file = 0
        list_files = os.listdir(dir_path)
        while index_file < len(list_files) and not (dev and index_file >= dev_iter):
            filename = list_files[index_file]
            index_file += 1
            file_path = os.path.join(dir_path, filename)
            with open(file_path, "r") as f:
                corpus[os.path.join(n_dir,filename)] = f.read()
    return corpus

def load_corpus_from_binary(path: str) -> OrdDict[str, str]:
    """Load corpus from binary file
    
    Arguments:
        path {string} -- path of the binary file to load corpus from
    
    Returns:
        [type] -- [description]
    """
    with open(path,"rb") as f:
        corpus = pkl.load(f)
    return corpus
    
def tokenize_document(document: str) -> List[str]:
    """tokenize a document with nltk
    
    Arguments:
        document {str} -- a str representing the document, query, etc..
    
    Returns:
        List[str] -- lowered tokens
    """
    tokens = word_tokenize(document.lower())
    return [tok.lower() for tok in tokens]

def tokenize_collection(collection: OrdDict[str, str]) -> OrdDict[str, List[str]]:
    """tokenize a collection
    
    Arguments:
        collection {OrdDict[str, str]} -- the collection to tokenize
    
    Returns:
        OrdDict[str, List[str]] -- tokenized collection
    """
    new_collection = OrderedDict()    
    for key in collection:
        new_collection[key] = tokenize_document(collection[key])

    return new_collection

def load_stop_words(stop_word_path: str) -> List[str]:
    """load the list of stop words from a file
    
    Arguments:
        stop_word_path {str} -- path for the stop words files
    
    Returns:
        List[str] -- list of stop words
    """    
    with open(stop_word_path,"r") as f:
        stp = [word.lower() for word in f.read().split("\n") if word != ""]
    return stp

def remove_stop_words_from_document(d: List[str], stop_words: List[str], exceptions: List[str] = []) -> List[str]:
    """remove stop words from a list, except for tokens specified in exceptions

    Arguments:
        d {List[str]} -- tokenized document
        stop_word_path {List[str]} -- list of stop words
        exceptions {List[str]} -- exceptions, which might exist in the stop word list but should not be deleted for this run
                                  For example, AND, OR or NAND for boolean queries.

    Returns:
        List[str] -- tokenized document without the stop words
    """
    
    return [word for word in d if (word in exceptions) or (word not in stop_words)]

def remove_stop_words_collection(collection: OrdDict[str, List[str]], stop_word_path: str) -> OrdDict[str, List[str]]:
    """remove all stop words from corpus given a stop words file
    
    Arguments:
        collection {OrdDict[str, List[str]]} -- corpus of split articles
        stop_word_path {string} -- path of the stop words file
    
    Returns:
        {OrdDict[str, List[str]]} -- updated corpus
    """
    stp = load_stop_words(stop_word_path)
    new_corpus = {}
    for key in collection.keys():
        new_corpus[key] = remove_stop_words_from_document(collection[key], stp, [])
    return new_corpus

def get_lemmatizer() -> StemmerI:
    """generate a lemmatizer
    
    Returns:
        StemmerI -- lemmatizer
    """
    return WordNetLemmatizer()    

def lemmatize_document(document: List[str], lemmatizer: StemmerI = get_lemmatizer()) -> List[str]:
    """
    lemmatize a single sentence, document, query.
    Having a full sentence/query/document allows to use the context for a better lemmatization.
    
    Arguments:
        sentence {List[str]} -- tokenized sentence
    
    Keyword Arguments:
        lemmatizer {StemmerI} -- lemmatizer (default: {get_lemmatizer()}) - generate one if necessary
    
    Returns:
        str -- [description]
    """
   
    tags = pos_tag(document)
    return [lemmatizer.lemmatize(tag[0],get_wordnet_pos(tag[1])) for tag in tags]

def lemmatize_collection(segmented_collection: OrdDict[str, List[str]]) -> OrdDict[str, List[str]]:
    """Lemmatize all articles in corpus using pos tags
    
    Arguments:
        segmented_collection {OrdDict[str, List[str]]} -- corpus without stop words
    
    Returns:
        {OrdDict[str, List[str]]} -- lemmatized corpus
    """
    lemmatized_collection = OrderedDict()
    stemmer = WordNetLemmatizer() # initialisation d'un lemmatiseur
    for key in segmented_collection.keys():
        lemmatized_collection[key] = lemmatize_document(segmented_collection[key], stemmer)
    return lemmatized_collection

def construct_documents_mapping(collection: OrdDict[str, Any]) -> OrdDict[int, str]:
    """get the default mapping between documents paths and ids for an inverted index
    
    Arguments:
        collection {OrdDict[str, Any]} -- collection 
    
    Returns:
        List[str] -- [description]
    """    
    mapping = OrderedDict()
    for idx, key in enumerate(collection.keys()):
        mapping[idx] = key
    return mapping

@dataclass
class InvertedIndex:
    """
    Representation of an inverted index

    'self.index' is the actual inverted index, from terms to document id
    'self.mapping' is the mapping from document id to the actual document path/name
    'self.itype' describe the type of the index :
        - 1 : simple doc index, we save only the id of the documents in which the term appears
        - 2 : frequency index, we save both the id of the documents and the tf in the document
        - 3 : position index, we save, for each term, the id of the document, and all the position of the terms
    """
    itype: int
    index: Union[
        OrdDict[str, List[int]], # itype == 1
        OrdDict[str, OrdDict[int, int]], # itype == 2
        OrdDict[str, OrdDict[int, List[int]]] # itype == 3
        ]
    mapping: OrdDict[int, str]


# TODO: @Antoine-marchais : This part could be refactored with auxiliary function for lisibility.
def build_inverted_index(
    collection: OrdDict[str, str],
    stop_words_path: str,
    type_index: int = 1,
    ) -> InvertedIndex:
    """Build an inverted index from a corpus
    
    Arguments:
        collection {OrdDict[str, str]} -- corpus
    
    Keyword Arguments:
        type_index {int} -- type of index : document(1) frequency(2) position(3) (default: {1})
    
    Returns:
        {InvertedIndex} -- inverted index of given type
    """
    collection = tokenize_collection(collection)
    collection = remove_stop_words_collection(collection, stop_words_path)
    collection = lemmatize_collection(collection)
    collection = remove_stop_words_collection(collection, stop_words_path)
    mapping = OrderedDict()
    index = OrderedDict()

    doc_id = 0
    if type_index == 1:
        for document in collection:

            for term in collection[document]:
                if term in index.keys():
                    if doc_id not in index[term]:
                        index[term].append(doc_id)
                else:
                    index[term]=[doc_id]

            mapping[doc_id] = document
            doc_id += 1

    elif type_index ==2:
        for document in collection:
            for term in collection[document]:
                if term in index.keys():
                    if doc_id in index[term].keys():
                        index[term][doc_id] = index[term][doc_id] + 1
                    else:
                        index[term][doc_id]= 1
                else:
                    index[term]=OrderedDict()
                    index[term][doc_id]=1

            mapping[doc_id] = document
            doc_id += 1

    elif type_index==3:
        for document in collection:
            pos=0
            for term in collection[document]:
                if term in index.keys():
                    if doc_id in index[term].keys():
                        index[term][doc_id].append(pos)
                    else:
                        index[term][doc_id]= [pos]
                else:
                    index[term]=OrderedDict()
                    index[term][doc_id]=[pos]
                pos += 1
            
            mapping[doc_id] = document
            doc_id += 1
    
    else:
        raise Exception(f"index type '{type_index}' is not supported")
    
    return InvertedIndex(type_index, index, mapping)

def get_wordnet_pos(treebank_tag: str) -> str:
    """Convert treebank tags into wordnet POS tag"""

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


if __name__ == "__main__" :
    print("reading files")
    if os.path.exists(PATH_DATA_BIN) and not DEV_MODE:
        corpus = load_corpus_from_binary(PATH_DATA_BIN)
    else:
        corpus = create_corpus_from_files(PATH_DATA, dev=DEV_MODE, dev_iter=DEV_ITER)
        if not DEV_MODE :
            print("saving corpus as binary")
            with open(PATH_DATA_BIN,"wb") as f:
                pkl.dump(corpus, f)
    
    print("creating inverted index")
    index = build_inverted_index(corpus, PATH_STOP_WORDS, type_index=1)
    print("saving index")
    with open(PATH_INDEX,"wb") as f:
        pkl.dump(index,f)