from math import gcd, ceil
from fractions import Fraction
import numpy as np
from functools import reduce

# Useful when we wanna print a matrix
np.set_printoptions(formatter={'all':lambda x: str(Fraction(x).limit_denominator())})

# Find least common multiplier for a list of factors
def lcm(m):
    return reduce(lambda a,b: a * b // gcd(a, b), m)

# Return Absorbing probabilities based on https://en.wikipedia.org/wiki/Absorbing_Markov_chain#Absorbing_probabilities
def answer(input_list):
    # Input is a matrix n x n in standard form with non absorbing states first
    idx = next(i for i, v in enumerate(input_list) if not any(v))
    # Replace values by probabilities for non final states
    for i in range(idx):
        val = input_list[i]
        odds = sum(val)
        for j in range(len(input_list[0])):
            val[j] = Fraction(val[j], odds)
    matrix_p = np.array(input_list, dtype=float)
    # Probability of transitioning from some transient state to another
    matrix_q = matrix_p[:idx, :idx]
    # Probability of transitioning from some transient state to some absorbing state
    matrix_r = matrix_p[:idx, idx:]
    # Fundamental matrix
    matrix_n = np.linalg.inv((np.identity(idx) - matrix_q))
    # Probability of being absorbed in the absorbing state j when starting from transient state i
    matrix_b = np.dot(matrix_n, matrix_r)
    # Probability of being absorbed in the absorbing state j when starting from transient state 0
    matrix_b0 = matrix_b[0]
    common_denominator = lcm([Fraction(x).limit_denominator().denominator for x in matrix_b0.tolist()])
    # Multiply by the common denominator to get normalized numerators
    normalized_matrix = np.apply_along_axis(lambda x: x * common_denominator, axis=0, arr=matrix_b0)
    # Append common denominator to list
    # As values are stored as float let's get corresponding ceiling values
    return [ceil(x) for x in np.append(normalized_matrix, [common_denominator])]
