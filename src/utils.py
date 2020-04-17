import functools
import logging
import time

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"starting '{func.__name__}'")
        t0 = time.perf_counter()
        res = func(*args, **kwargs)
        print(f"finished '{func.__name__}' in {time.perf_counter() - t0:3f}s.")
        return res
    return wrapper
