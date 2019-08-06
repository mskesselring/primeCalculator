from numba import vectorize
import numba
import numpy as np


def convert_seconds(s):
    x = s
    mult = 1
    while s < 1:
        if mult < 1e9:
            s *= 1e3
            mult *= 1e3
        else:
            break
    if mult == 1:
        unit = "s"
    elif mult == 1e3:
        unit = "ms"
    elif mult == 1e6:
        unit = "us"
    elif mult == 1e9:
        unit = "ns"
    else:
        raise ValueError("Error converting number %f: %f, %f" % (x, s, mult))
    return s, unit


def cuda_multithread(n):
    print(min(n), end='\r', flush=True)
    a = np.array(range(int(min(n)), int(max(n)+1)), dtype='uint64')
    p = check_primes_cuda(a).tolist()
    return p


@vectorize(['uint64(uint64)'], target='cuda')
def check_primes_cuda(p):
    if p < 10:
        if p in [4, 6, 8, 9]:
            return numba.uint64(0)
        else:
            pass
    else:
        for i in range(2, (p**0.5)//1):
            if (p % i) == 0:
                return numba.uint64(0)
            else:
                pass
    return p
