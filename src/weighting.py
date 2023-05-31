import polars as pl


class Weighting:
    """
    Weighting of PPIN.
    """

    def __init__(self, ppin: str):
        self.ppin = ppin
