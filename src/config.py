import os

DEV_MODE = True
DEV_ITER = 100

cwd = os.getcwd()
PATH_DATA = os.path.join(cwd,"data","collection")
PATH_STOP_WORDS = os.path.join(cwd, "data", "stop_words.txt")
PATH_INDEX = os.path.join(cwd, "data", "index.pkl") if DEV_MODE else os.path.join(cwd, "data", "dev_index.pkl")