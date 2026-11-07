"""Small NumPy array helpers shared by simulation and rendering."""

import numpy as np


def higher_pow2(n: int | np.uint32 | np.uint64) -> int:
    """Return the smallest power of two strictly greater than n (zero gives one)."""
    return 1 << int(n).bit_length()
