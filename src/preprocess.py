import os
import pickle as pkl

from typing import Optional, Dict, List, Union, Tuple

from config import PATH_DATA, DEV_MODE, DEV_ITER, PATH_INDEX, PATH_STOP_WORDS, PATH_DATA_BIN
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

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
    corpus = {}
    for n_dir in os.listdir(path):
        dir_path = os.path.join(path, n_dir)
        index_file = 0
        list_files = os.listdir(dir_path)
        while index_file < len(list_files) and not (dev and index_file >= dev_iter):
            filename = list_files[index_file]
            index_file += 1
            file_path = os.path.join(dir_path, filename)
            with open(file_path, "r") as f:
                tokens = f.read().split()
            corpus[os.path.join(n_dir,filename)] = [token.lower() for token in tokens]
    return corpus

def load_corpus_from_binary(path):
    """Load corpus from binary file
    
    Arguments:
        path {string} -- path of the binary file to load corpus from
    
    Returns:
        [type] -- [description]
    """
    with open(path,"rb") as f:
        corpus = pkl.load(f)
    return corpus

def remove_stop_words(collection: Dict[str, List[str]], stop_word_path: str) -> Dict[str, List[str]]:
    """remove all stop words from corpus given a stop words file
    
    Arguments:
        collection {Dict[str, List[str]]} -- corpus of split articles
        stop_word_path {string} -- path of the stop words file
    
    Returns:
        {Dict[str, List[str]]} -- updated corpus
    """
    with open(stop_word_path,"r") as f:
        stp = [word.lower() for word in f.read().split("\n") if word != ""]
    new_corpus = {}
    for key in collection.keys():
        new_corpus[key] = [word for word in collection[key] if word not in stp]
    return new_corpus

def collection_lemmatize(segmented_collection: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Lemmatize all articles in corpus using pos tags
    
    Arguments:
        segmented_collection {Dict[str, List[str]]} -- corpus without stop words
    
    Returns:
        {Dict[str, List[str]]} -- lemmatized corpus
    """
    lemmatized_collection = {}
    stemmer = WordNetLemmatizer() # initialisation d'un lemmatiseur
    for key in segmented_collection.keys():
        tags = pos_tag(segmented_collection[key])
        lemmatized_collection[key] = [stemmer.lemmatize(tag[0],get_wordnet_pos(tag[1])) for tag in tags]
    return lemmatized_collection

def build_inverted_index(
    collection: Dict[str, List[str]],
    stop_words_path: str,
    type_index: int = 1,
    ) -> Union[Dict[str, str], Dict[str, Tuple[str, int]], Dict[str, Tuple[int]]]:
    """Build an inverted index from a processed corpus
    
    Arguments:
        collection {dict} -- processed corpus
    
    Keyword Arguments:
        type_index {int} -- type of index : document(1) frequency(2) position(3) (default: {1})
    
    Returns:
        dict -- inverted index of given type
    """
    collection = remove_stop_words(collection, stop_words_path)
    collection = collection_lemmatize(collection)
    index = {}
    for key in collection.keys():
        terms = set(collection[key])
        for t in terms:
            if not t in index:
                index[t] = []
            if type_index == 1:
                index[t].append(key)
            elif type_index == 2:
                index[t].append((key,collection[key].count(t)))
            elif type_index == 3:
                index[t].append((key,tuple([index for index in range(len(collection[key])) if collection[key][index] == t])))
    return index

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
            with open(PATH_DATA_BIN,"wb") as f:
                pkl.dump(corpus, f)
    
    print("creating inverted index")
    index = build_inverted_index(corpus, PATH_STOP_WORDS, type_index=1)
    print("saving index")
    with open(PATH_INDEX,"wb") as f:
        pkl.dump(index,f)