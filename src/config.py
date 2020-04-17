import os

DEV_MODE = True
DEV_ITER = 100

cwd = os.getcwd()
PATH_DATA = os.path.join(cwd,"data","collection")
PATH_DATA_BIN = os.path.join(cwd,"data","corpus.pkl")
PATH_STOP_WORDS = os.path.join(cwd, "data", "stop_words.txt")
PATH_INDEX = os.path.join(cwd, "data", "index.pkl") if not DEV_MODE else os.path.join(cwd, "data", "dev_index.pkl")

POS = False

WEIGHT_QUERY = "tf_idf"
WEIGHT_DOCUMENT = "tf_idf_logarithmic_normalize"