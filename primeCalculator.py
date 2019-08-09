from primeChecker import PrimeChecker
import sys
import os
import json
import argparse
import math
from multiprocessing import Pool
import time
from primechecker_cuda import *
import numpy as np

filepath = os.path.dirname(os.path.abspath(__file__))
primename = os.path.join(filepath, "primes.json")
lastcheckedfile = os.path.join(filepath, "last_checked.json")
maxStep = int(2e5)
maxIncr = int(1e9)
uint64max = np.iinfo(np.uint64).max
largeInc = int(1e6)


def primeworker(arg):
    print(arg[0], arg[1], end='\r', flush=True)
    primes = set()
    c = PrimeChecker()
    for i in range(int(arg[0]), int(arg[1]) + 1):
        if c.check_prime(i):
            primes.add(i)
    return primes


def find_lowest_multiple(num, bound):
    for i in range(bound, (bound * num) + 1):
        if (i % num) == 0:
            return i
    return None


def sieve(limit):
    primes = set()
    lim_sqrt = int(np.floor(limit**0.5))
    seg = 100
    delta = seg if (seg < lim_sqrt) else lim_sqrt
    divs = list(range(delta, limit + 1, delta))
    for i in range(len(divs)):
        print(divs[i], end='\r', flush=True)
        if i == 0:
            # Normal sieve
            nums = set(range(2, divs[0]+1))
            for p in range(2, divs[0]+1):
                if p not in nums:
                    continue
                else:
                    pass
                for j in range(p ** 2, divs[0], p):
                    nums = nums.difference({j})
            primes = primes.union(nums)
        else:
            # Segmented sieve
            nums = set(range(divs[i-1], divs[i]+1))
            m = divs[i]
            for p in range(2, int(np.floor(m**0.5))+1):
                if (p not in nums) and (p > divs[i-1]):
                    continue
                else:
                    pass
                p_mult = find_lowest_multiple(p, m - delta)
                for j in range(p_mult, m, p):
                    nums = nums.difference({j})
            primes = primes.union(nums)
    print("", end='\r', flush=True)
    primes = list(primes)
    primes.sort()
    return set(primes)


def pool_func(args, func, processes):
    with Pool(processes) as p:
        r = p.imap_unordered(func, args)
        p.close()
        p.join()
        tmp = []
        for i in r:
            tmp.append(i)
    return tmp


def main(increment: int, processes: int, cuda: bool):
    if increment > largeInc:
        print(
            "WARNING: large increments may cause your computer to become" +
            " unresponsive.")
        while True:
            ans = input("Continue anyway?[Y/N]: ")
            if ans.lower() == "y":
                break
            elif ans.lower() == "n":
                return
            else:
                print("INVALID INPUT")

    # Load last number checked
    try:
        with open(lastcheckedfile, "r") as checkedfile:
            lastchecked = json.load(checkedfile)
        print("file loaded")
    except:
        lastchecked = 3
        print("unable to load file")

    print("Calculating primes from %d to %d" % (
        lastchecked, lastchecked + increment))

    start = time.perf_counter()

    print("Processes: %d (CUDA Enabled)" % processes)
    # Check if numbers will be greater than max uint64
    if (lastchecked + increment) > uint64max:
        raise ValueError(
            "ERROR: lastchecked plus increment (%d) must be less than %d" % (
                (lastchecked + increment), uint64max))
    # Loop through batches of size [maxIncr]
    primes = set()
    primebatches = []
    for i in range(lastchecked, lastchecked + increment, maxIncr):
        if (i + maxIncr) > (lastchecked + increment):
            n = lastchecked + increment
        else:
            n = i + maxIncr
        # print("%d to %d" % (i, n), end='\r', flush=True)
        args = []
        # Break up into groups of [maxStep]
        for j in range(i, n, maxStep):
            m = n if ((j + maxStep) > n) else (j + maxStep)
            tmp = np.array([j, m], dtype='uint64')
            args.append(tmp)
        if cuda:
            r = pool_func(args, cuda_multithread, processes)
        else:
            r = pool_func(args, primeworker, processes)
        for i in r:
            primebatches.append(i)

        lastchecked = n
    print("", flush=True)
    print("Combining results...")
    primes = primes.union(*primebatches)
    print("Total found:", len(primes))
    print("Total time:", time.perf_counter() - start)
    dump(primes, lastchecked)


def dump(primes, lastchecked):
    # Load history file
    print("Writing results to file...")
    try:
        with open(primename, "r") as primefile:
            data = set(json.load(primefile))
    except:
        data = set()

    # Merge history and new primes
    data = sorted(data.union(primes))

    # Write prime numbers and last checked number to file
    with open(primename, "w") as primefile:
        json.dump(data, primefile)
    with open(lastcheckedfile, "w") as checkedfile:
        json.dump(lastchecked, checkedfile)


def increment_type(x):
    try:
        x = int(x)
    except ValueError:
        x = int(float(x))

    if x > uint64max:
        raise argparse.ArgumentTypeError(
            "Maximum increment is %d" % uint64max)
    else:
        return x


if __name__ == "__main__":
    process_choices = []
    for i in range(1, os.cpu_count() + 1):
        process_choices.append(i)
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument("--increment", type=increment_type, default=1000,
                        help="Increment <= 100000000")
    parser.add_argument("--processes", type=int,
                        default=math.floor(os.cpu_count() * 0.5),
                        choices=process_choices)
    kwargs = parser.parse_args()
    sys.exit(main(kwargs.increment, kwargs.processes, kwargs.cuda))
