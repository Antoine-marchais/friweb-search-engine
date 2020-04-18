import argparse

from config import PATH_DATA, DEV_MODE, DEV_ITER, PATH_STOP_WORDS
from preprocess import create_corpus_from_files, build_inverted_index, save_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("index_type", type=int, help="type of the index to build")
    parser.add_argument("--pos", type=bool, default=False, help="flag to use the the pos lemmatization")
    args = parser.parse_args()

    index_type = args.index_type
    pos = args.pos

    print(f"constructing an index of type {index_type} {'with pos lemmatization' if pos else ''}")

    print("reading corpus")
    corpus = create_corpus_from_files(PATH_DATA, dev=DEV_MODE, dev_iter=DEV_ITER)

    print(f"build inverted index of type {index_type}")
    index = build_inverted_index(corpus, PATH_STOP_WORDS, type_index=index_type, pos=pos)

    print("saving index with pickle")
    save_index(f"data/index.pkl", index)
