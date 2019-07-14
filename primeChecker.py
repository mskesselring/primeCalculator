import math


class PrimeChecker:

    def __init__(self):
        self.factHist = None

    def fact(self, n: int):
        if self.factHist and (n >= self.factHist["n"]):
            # print("using cache")
            val = self.factHist["fact"]
            # print(self.factHist)
            for i in range(self.factHist["n"]+1, n + 1):
                # print(i)
                val *= i
        else:
            # print("using library")
            val = math.factorial(n)
        self.factHist = {"n": n, "fact": val}
        # print(self.factHist)
        return val

    def check_prime(self, p):

        if (p <= 1) or not (p & 1):
            return False
        else:
            a =self.fact(p-1) % p
            b = p - 1
            # print("p: %d" % p)
            # print("a: %d" % a)
            # print("b: %d" % b)
            return a==b
