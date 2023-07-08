"""
Functions that facilitate the modeling process.
"""


class SearchClassificationTrainer:
    def __init__(self, chunk_size:int=100000) -> None:
        self.chunk_size = 0
        pass

    @staticmethod
    def split_array(arr:np.array, chunk_size:int)