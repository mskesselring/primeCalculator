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
maxStep = 1000000


def primeworker(arg):
    print(arg["start"], end='\r', flush=True)
    primes = set()
    c = PrimeChecker()
    for i in range(arg["start"], arg["end"]):
        if c.check_prime(i):
            primes.add(i)
    return primes


def main(increment: int, processes: int, cuda: bool):
    start = time.perf_counter()
    # Load last number checked
    try:
        with open(lastcheckedfile, "r") as checkedfile:
            lastchecked = json.load(checkedfile)
        print("file loaded")
    except:
        lastchecked = 0
        print("unable to load file")

    print("Calculating primes from %d to %d" % (
        lastchecked, lastchecked + increment))

    if cuda:
        print("Processes: %d (CUDA Enabled)" % processes)
        # Check if numbers will be greater than max uint64
        uint64max = np.iinfo(np.uint64).max
        if (lastchecked + increment) > uint64max:
            raise ValueError(
                "ERROR: lastchecked plus increment (%d) must be less than %d" % (
                    (lastchecked + increment), uint64max))
        # Check numbers, break up into groups of [maxStep]
        if increment > maxStep:
            part = increment // maxStep
            remainder = increment % (maxStep * part)
        else:
            part = 0
            remainder = increment
        if not remainder:
            args = []
        else:
            tmp = np.array([lastchecked, lastchecked + remainder],
                           dtype='uint64')
            args = [tmp]
        primes = set()
        for i in range(part):
            try:
                last = max(args[len(args) - 1])
            except IndexError:
                last = lastchecked
            tmp = np.array([last, last + maxStep], dtype='uint64')
            args.append(tmp)
        lastchecked = int(args[len(args)-1][1])
        # Multiple cpu processes to help keep gpu fed
        with Pool(processes) as p:
            r = p.imap_unordered(cuda_multithread, args)
            p.close()
            p.join()
            print("Combining results...", flush=True)
            for i in r:
                primes = primes.union(i)
    else:
        # Multiprocessing
        print("Processes: %d" % processes)
        with Pool(processes) as p:
            if processes == 1:
                args = [{"start": lastchecked, "end": lastchecked + increment}]
            else:
                step = math.floor((lastchecked + increment) / processes)
                bounds = []
                for i in range(lastchecked, lastchecked + increment, step):
                    bounds.append(i)
                lastchecked = lastchecked + increment
                bounds[len(bounds) - 1] = lastchecked
                args = []
                for i in range(0, len(bounds) - 1):
                    args.append({"start": bounds[i], "end": bounds[i + 1]})
            r = p.imap_unordered(primeworker, args)
            p.close()
            p.join()
            primes = set()
            print("Combining results...", flush=True)
            for i in r:
                primes = primes.union(i)

    print("Found: %d" % len(primes))
    print("Elapsed time:", time.perf_counter() - start)

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

    if x > 100000000:
        raise argparse.ArgumentTypeError(
            "Maximum increment is 100 Million (100000000)")
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
