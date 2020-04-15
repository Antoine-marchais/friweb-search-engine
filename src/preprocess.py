import os
import pickle as pkl

from collections import OrderedDict, Counter
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Dict, List, Union, Tuple

from config import PATH_DATA, DEV_MODE, DEV_ITER, PATH_INDEX, PATH_STOP_WORDS, PATH_DATA_BIN
from nltk import pos_tag
from nltk.stem.api import StemmerI
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize


def create_corpus_from_files(path: str, dev: bool =False, dev_iter: Optional[int]=None) -> Dict[str, List[str]]:
    """Read file and import tokens into a new collection
    
    Arguments:
        path {string} -- path of the collection
    
    Keyword Arguments:
        dev {bool} -- activate dev mode (default: {False})
        dev_iter {Optional[int]} -- number of iterations in dev mode (default: {None})
    
    Returns:
        Dict[str, List[str]] -- The loaded corpus
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

def load_corpus_from_binary(path: str) -> Dict[str, List[str]]:
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
    tokens = word_tokenize(document)
    return [tok.lower() for tok in tokens]

def tokenize_collection(collection: Dict[str, str]) -> Dict[str, List[str]]:
    new_collection = OrderedDict()
    for key in tqdm(collection, desc="tokenizing collection : "):
        new_collection[key] = tokenize_document(collection[key])

    return new_collection

def load_stop_words(stop_word_path: str) -> List[str]:
    with open(stop_word_path,"r") as f:
        stp = [word.lower() for word in f.read().split("\n") if word != ""]
    return stp

def remove_stop_words_from_document(d: List[str], stop_words: List[str], exceptions: List[str] = []) -> List[str]:
    """remove stop words from a list, except for tokens specified in exceptions

    Arguments:
        d {List[str]} -- [description]
        stop_word_path {str} -- [description]
        exceptions {List[str]} -- [description]

    Returns:
        List[str] -- [description]
    """
    
    return [word for word in d if (word in exceptions) or (word not in stop_words)]

def remove_stop_words_collection(collection: Dict[str, List[str]], stop_word_path: str) -> Dict[str, List[str]]:
    """remove all stop words from corpus given a stop words file
    
    Arguments:
        collection {Dict[str, List[str]]} -- corpus of split articles
        stop_word_path {string} -- path of the stop words file
    
    Returns:
        {Dict[str, List[str]]} -- updated corpus
    """
    stp = load_stop_words(stop_word_path)
    new_corpus = {}
    for key in tqdm(collection, desc="removing stop words"):
        new_corpus[key] = remove_stop_words_from_document(collection[key], stp, [])
    return new_corpus

def get_lemmatizer() -> StemmerI:
    return WordNetLemmatizer()

def lemmatize_one_token(token: str, lemmatizer: StemmerI) -> str:
    tags = pos_tag([token])
    tag = tags[0]
    return lemmatizer.lemmatize(tag[0], get_wordnet_pos(tag[1]))

def lemmatize_doc(tokens: List[str]) -> List[str]:
    stemmer = WordNetLemmatizer()
    tags = pos_tag(tokens)
    return [stemmer.lemmatize(tag[0],get_wordnet_pos(tag[1])) for tag in tags]

def lemmatize_collection(segmented_collection: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Lemmatize all articles in corpus using pos tags
    
    Arguments:
        segmented_collection {Dict[str, List[str]]} -- corpus without stop words
    
    Returns:
        {Dict[str, List[str]]} -- lemmatized corpus
    """
    lemmatized_collection = {}
    stemmer = WordNetLemmatizer() # initialisation d'un lemmatiseur
    for key in tqdm(segmented_collection, desc="lemmatizing collection"):
        tags = pos_tag(segmented_collection[key])
        lemmatized_collection[key] = [stemmer.lemmatize(tag[0],get_wordnet_pos(tag[1])) for tag in tags]
    return lemmatized_collection

def construct_documents_mapping(collection: Dict[str, List[str]]) -> List[str]:
    return collection.keys()

@dataclass
class StatCollection:
    """
    Usefull stats of a collection:

    'self.nb_docs' is the number of docs in the collection
    'self.doc_stats' is a dictionary that provides for a given article id its collection
    """
    nb_docs: int
    doc_stats: Dict[int, Dict[str, float]]

def get_stats_document(document: List[str]) -> Dict[str, float]:
    """Computes usefull stats for a document
    
    Arguments:
        document {List[str]} -- lemmatized document
    
    Returns:
        Dict[str, float] -- max frequency, mid frequency and unique count in the document
    """
    stats={}
    frequencies = Counter(document)
    stats["freq_max"] = max(frequencies.values())
    stats["moy_freq"] = sum(frequencies.values())/len(frequencies)
    stats["unique"] = len(frequencies.values())
    return stats

def get_stats_collection(processed_collection: Dict[int, List[str]]) -> StatCollection:
    """Computes usefull stats for the collection
    
    Arguments:
        processed_collection {Dict[str, List[str]]} -- collection of lemmatized articles
    
    Returns:
        StatCollection -- Statictics of the collection
    """
    stats={}
    for key in processed_collection:
        stats[key] = get_stats_document(processed_collection[key])
    return StatCollection(len(processed_collection), stats)

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
        Dict[str, List[int]], # itype == 1
        Dict[str, Dict[int, int]], # itype == 2
        Dict[str, Dict[int, List[int]]] # itype == 3
        ]
    mapping: Dict[int, str]
    stats: StatCollection

def build_inverted_index(
    collection: Dict[str, str],
    stop_words_path: str,
    type_index: int = 1,
    ) -> InvertedIndex:
    """Build an inverted index from a processed corpus
    
    Arguments:
        collection {dict} -- processed corpus
    
    Keyword Arguments:
        type_index {int} -- type of index : document(1) frequency(2) position(3) (default: {1})
    
    Returns:
        dict -- inverted index of given type
    """
    collection = tokenize_collection(collection)
    collection = remove_stop_words_collection(collection, stop_words_path)
    collection = lemmatize_collection(collection)
    collection = remove_stop_words_collection(collection, stop_words_path)
    mapping = OrderedDict()
    index = OrderedDict()

    doc_id = 0
    if type_index == 1:
        for document in tqdm(collection, desc="building index : "):

            for term in collection[document]:
                if term in index.keys():
                    if doc_id not in index[term]:
                        index[term].append(doc_id)
                else:
                    index[term]=[doc_id]

            mapping[doc_id] = document
            doc_id += 1

    elif type_index ==2:
        for document in tqdm(collection, desc="building index : "):
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
        for document in tqdm(collection, desc="building index : "):
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

    stats = get_stats_collection({doc_id:collection[mapping[doc_id]] for doc_id in mapping})
    
    return InvertedIndex(type_index, index, mapping, stats)

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
    
    index = build_inverted_index(corpus, PATH_STOP_WORDS, type_index=2)
    print("saving index")
    with open(PATH_INDEX,"wb") as f:
        pkl.dump(index,f)