from math import gcd, ceil
from fractions import Fraction
import numpy as np
from functools import reduce

np.set_printoptions(formatter={'all':lambda x: str(Fraction(x).limit_denominator())})

# Find least common multiplier for a list of factors
def lcm(m):
    return reduce(lambda a,b: a * b // gcd(a, b), m)

# Return Absorbing probabilities based on https://en.wikipedia.org/wiki/Absorbing_Markov_chain#Absorbing_probabilities
def answer(m):
    idx = next(i for i, v in enumerate(m) if not any(v))
    # Replace values by probabilities for non final states
    for i in range(idx):
        val = m[i]
        odds = sum(val)
        for j in range(len(m[0])):
            val[j] = Fraction(val[j], odds)
    mat = np.array(m, dtype=float)
    # Probability of transitioning from some transient state to another
    Q = mat[:idx, :idx]
    # Probability of transitioning from some transient state to some absorbing state
    R = mat[:idx, idx:]
    # Fundamental matrix
    N = np.linalg.inv((np.identity(idx) - Q))
    # Probability of being absorbed in the absorbing state j when starting from transient state i
    B = np.dot(N, R)
    # Probability of being absorbed in the absorbing state j when starting from transient state 0
    B0 = B[0]
    common_denominator = lcm([Fraction(x).limit_denominator().denominator for x in B0.tolist()])
    # Multiply by the common denominator to get normalized numerators
    normalized = np.apply_along_axis(lambda x: x * common_denominator, axis=0, arr=B0)
    # Append common denominator to list
    result = np.append(normalized, [common_denominator])
    # As values are stored as float let's get corresponding ceiling values
    return [ceil(x) for x in result]