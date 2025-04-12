import numpy as np

def get_dummy_sentiment(date, ticker):
    """
    Returns a dummy sentiment score between -1 and 1.
    """
    seed = hash(f"{date}-{ticker}") % (2**32)
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1)