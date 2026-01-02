"""
Copyright 2025 @hyperbolekillsme

As shared on the Xenharmonic Alliance discord https://discord.com/channels/332357996569034752/947247604100649000/1395090206247358474
"""

import math
import heapq

prime_factors_cache = {}

def get_complexity_score(a, b):
    """
    Calculates the complexity score of a rational number a/b.
    A lower complexity score is better for simpler ratios.
    The formula is 0.9^(a+b).
    """
    return 0.9**(height(a, b))

def get_accuracy_score(j, c):
    """
    Calculates how well a cent value j matches a target cent value c.
    A higher accuracy score is better.
    The formula is 0.98^(abs(j-c)^1.5).
    """
    return 0.97**(abs(j - c)**1.3)

def ratio_to_cents(a, b):
    """
    Converts a rational number a/b to cents.
    """
    if a <= 0 or b <= 0:
        return 0
    return 1200 * math.log2(a / b)

def find_top_approximations(target_cents, limit=10):
    """
    Finds the top rational number approximations for a given cent value.

    It works by iterating through sums of numerators and denominators (s = a+b)
    and calculating a score for each coprime fraction a/b. The score is a product
    of a complexity score (favoring small a and b) and an accuracy score
    (favoring fractions whose cent value is close to the target).

    A min-heap is used to efficiently keep track of the top 'limit' results.
    The search stops when the best possible score for a given complexity
    is lower than the worst score in our current list of top approximations.

    Args:
        target_cents (float): The target cent value to approximate.
        limit (int): The number of top approximations to return.

    Returns:
        list: A list of tuples, where each tuple contains:
              ((numerator, denominator), score, cents_value).
              The list is sorted by score in descending order.
    """
    # A min-heap to store the top approximations. We store tuples of
    # (score, (a, b), cents_val), as heapq sorts by the first element.
    top_approximations_heap = []

    s = 2
    # This loop could theoretically run for a long time, but the break
    # condition based on score ensures it terminates. A safety break is
    # included just in case.
    while s < 2000: # Safety break
        # Optimization: if the best possible score for this s (which has accuracy 1)
        # is worse than the worst score we have in our top list, we can stop.
        max_possible_score_at_s = 0.95**s
        if len(top_approximations_heap) == limit and max_possible_score_at_s < top_approximations_heap[0][0]:
            break

        for a in range(1, s):
            b = s - a
            if math.gcd(a, b) == 1:
                j = ratio_to_cents(a, b)
                complexity_score = get_complexity_score(a, b)
                accuracy_score = get_accuracy_score(j, target_cents)
                total_score = complexity_score * accuracy_score

                if len(top_approximations_heap) < limit:
                    heapq.heappush(top_approximations_heap, (total_score, (a, b), j))
                elif total_score > top_approximations_heap[0][0]:
                    heapq.heapreplace(top_approximations_heap, (total_score, (a, b), j))
        s += 1

    # The heap gives us the top results, but they are in min-heap order.
    # We convert it to a list of ((a,b), score, cents) and sort descending by score.
    result = [(item[1], item[0], item[2]) for item in top_approximations_heap]
    result.sort(key=lambda x: x[1], reverse=True)

    return result

def get_prime_factors(n):
    """
    Returns a list of prime factors of a positive integer.
    """
    factors = []
    d = 2
    while d * d <= n:
        while (n % d) == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
       factors.append(n)
    return factors

def _calculate_otonality_component(n, method="multiplicativeEuler"):
    """
    Helper for otonality calculation.
    For a number n, it computes a value based on its prime factors.
    The result of this function is NOT a logarithm.
    """
    if method == "multiplicativeEuler":
        if n in prime_factors_cache:
            factors = prime_factors_cache[n]
        else:
            factors = get_prime_factors(n)
            prime_factors_cache[n] = factors

        product = 1
        for factor in factors:
            product *= (factor - 0.5)
        return product + 1

    elif method == "euler":
        return 2**((sum(get_prime_factors(n)) - len(get_prime_factors(n)) + 1) * 0.4)

    elif method == "sum":
        return float(n)

    raise ValueError(f"Unknown method: {method}")

def otonality(a, b):
    """
    Calculates the otonality of a rational number a/b.

    Otonality is calculated as log2(product(p-1 for p in prime_factors(a)) / product(q-1 for q in prime_factors(b))).
    It uses a global cache for prime factorizations for efficiency.
    """
    if a <= 0 or b <= 0:
        return 0

    numerator_val = _calculate_otonality_component(a)
    denominator_val = _calculate_otonality_component(b)

    if denominator_val <= 0:
        return float('inf')

    return math.log2(numerator_val / denominator_val)

def height(a, b, method="multiplicativeEuler"):
    """
    Calculates the height of a rational number a/b.
    The height is a measure of the interval's complexity, based on its prime factors.
    """
    return _calculate_otonality_component(a, method) + _calculate_otonality_component(b, method)

def analyze_interval(target_cents, limit=100):
    """
    Analyzes an interval in cents to determine its aggregate otonality and height.

    This is done by finding the top rational approximations for the interval,
    and then summing their otonality and height values, weighted by their
    approximation score.

    Args:
        target_cents (float): The target cent value to analyze.
        limit (int): The number of top approximations to consider for the analysis.

    Returns:
        tuple: A tuple containing (average_otonality, average_height, consonance).
    """
    approximations = find_top_approximations(target_cents, limit)

    if not approximations:
        return 0, 0, 0

    total_otonality = 0
    total_height = 0
    total_score = 0

    for rational, score, _ in approximations:
        a, b = rational
        total_otonality += otonality(a, b) * score
        total_height += height(a, b) * score
        total_score += score

    return total_otonality, total_height/total_score, total_score


if __name__ == '__main__':
    for target_cents_value in [800, 900]:
        top_10 = find_top_approximations(target_cents_value, 10)

        print(f"Top 10 rational approximations for {target_cents_value} cents:\n")
        for rational, score, cents_val in top_10:
            print(
                f"  Rational: {rational[0]:>2}/{rational[1]:<2} "
                f"| Cents: {cents_val:>6.1f} "
                f"| Score: {score:.4f} "
                f"| Otonality: {otonality(rational[0], rational[1]):>6.3f} "
                f"| Height: {height(rational[0], rational[1]):.2f}"
            )

        # Analyze the interval using a larger set of approximations for a more stable result.
        avg_otonality, avg_height, consonance = analyze_interval(target_cents_value, limit=100)
        print(f"\nAnalysis for {target_cents_value} cents (based on top 100 approximations):")
        print(f"  Otonality: {avg_otonality:.4f}")
        print(f"  Height: {avg_height:.4f}")
        print(f"  Consonance (Total Score): {consonance:.4f}")
