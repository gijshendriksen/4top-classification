from functools import wraps
import os

import numpy as np


def cache_np(filename: str, use_cache: bool):
    def wrapper(func):
        @wraps(func)
        def with_caching(*args, **kwargs):
            if use_cache and os.path.exists(filename):
                return np.load(filename)

            data = func(*args, **kwargs)

            if use_cache:
                np.save(filename, data)

            return data
        return with_caching
    return wrapper
