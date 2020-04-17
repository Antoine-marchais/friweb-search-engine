import pickle as pkl

from config import PATH_DATA, DEV_MODE, DEV_ITER, PATH_STOP_WORDS
from preprocess import create_corpus_from_files, build_inverted_index, save_index

index_type = 1

print("reading corpus")
corpus = create_corpus_from_files(PATH_DATA, dev=DEV_MODE, dev_iter=DEV_ITER)

print(f"build inverted index of type {index_type}")
index = build_inverted_index(corpus, PATH_STOP_WORDS, type_index=index_type)

print("saving index with pickle")
save_index(f"data/index_type{index_type}", index)