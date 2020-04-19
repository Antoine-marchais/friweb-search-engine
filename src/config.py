import os

DEV_MODE = bool(os.getenv("DEV", default=False))
DEV_ITER = int(os.getenv("DEV_ITER", default=100))

cwd = os.getcwd()
PATH_DATA = os.path.join(cwd,"data","collection")
PATH_DATA_BIN = os.path.join(cwd,"data","corpus.pkl")
PATH_STOP_WORDS = os.path.join(cwd, "data", "stop_words.txt")
PATH_INDEX = os.getenv("PATH_INDEX", default=os.path.join(cwd, "data", "index_type2_pos.pkl") if not DEV_MODE else os.path.join(cwd, "data", "dev_index.pkl"))

POS = bool(os.getenv("POS", True))

WEIGHT_QUERY = os.getenv("WEIGHT_QUERY", default="tf_idf")
WEIGHT_DOCUMENT = os.getenv("WEIGHT_DOCUMENT", default="tf_idf_logarithmic_normalize")