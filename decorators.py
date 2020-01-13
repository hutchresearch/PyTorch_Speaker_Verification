import time
from functools import wraps

def timeit(func=None, *, msg=None):
    def _timeit(f):
        @wraps(f)
        def wrapper(*args, verbose=True, **kwargs):
            if msg:
                m = '{}...'.format(msg)
            else:
                m = 'Running {}...'.format(f.__name__)

            print(m, end='\r')
            s = time.time()
            res = f(*args, **kwargs)
            t = time.time() - s
            print('{}\tDone ({:0.3f} seconds)'.format(m, t))
            return res
        return wrapper

    if func:
        if callable(func):
            return _timeit(func)
        else:
            raise TypeError("Argument func is expected to be callable")
    else:
        return _timeit
