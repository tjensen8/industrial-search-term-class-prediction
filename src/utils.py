"""
Misc. utility functions.
"""


import pickle as pkl


def save_pickle_file(obj: object, loc: str):
    try:
        with open(loc, "wb") as f:
            pkl.dump(obj, f)
            return True
    except Exception as e:
        print(e)
        return False


def load_pickle_file(loc: str):
    try:
        with open(loc, "rb") as f:
            obj = pkl.load(f)
            return obj
    except Exception as e:
        print(e)
        return False
