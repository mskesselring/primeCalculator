from primeChecker import PrimeChecker
import sys
import os
import json
import argparse
import math
from multiprocessing import Pool
import time

filepath = os.path.dirname(os.path.abspath(__file__))
primename = os.path.join(filepath, "primes.json")
lastcheckedfile = os.path.join(filepath, "last_checked.json")


def primeworker(arg):
    primes = set()
    c = PrimeChecker()
    for i in range(arg["start"], arg["end"]):
        if c.check_prime(i):
            primes.add(i)
    return primes


def main(increment: int, processes: int):
    start = time.time()
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

    # Multiprocessing
    print("Processes: %d" % processes)
    with Pool(processes) as p:
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
        for i in r:
            primes = primes.union(i)

    # # Look for prime numbers
    # primes = set()
    # c = PrimeChecker()
    # try:
    #     for i in range(lastchecked, lastchecked + increment + 1):
    #         lastchecked = i
    #         if c.check_prime(i):
    #             primes.add(i)
    # except KeyboardInterrupt:
    #     lastchecked -= 1

    # Load history file
    try:
        with open(primename, "r") as primefile:
            data = set(json.load(primefile))
    except:
        data = set()

    # Merge history and new primes
    print("Found: %d" % len(primes))
    data = sorted(data.union(primes))
    # for i in data:
    #     print(i)

    # Write prime numbers and last checked number to file
    with open(primename, "w") as primefile:
        json.dump(data, primefile)
    with open(lastcheckedfile, "w") as checkedfile:
        json.dump(lastchecked, checkedfile)
    print("Elapsed time:", time.time() - start)


if __name__ == "__main__":
    process_choices = []
    for i in range(1, os.cpu_count() + 1):
        process_choices.append(i)
    parser = argparse.ArgumentParser()
    parser.add_argument("--increment", type=int, default=1000)
    parser.add_argument("--processes", type=int,
                        default=math.floor(os.cpu_count() * 0.5),
                        choices=process_choices)
    args = parser.parse_args()
    sys.exit(main(args.increment, args.processes))
