"""
Seat protection: Littlewood, EMSR-a, EMSR-b and the optimal DP policy.

* Classes are ordered by fare, highest first.
* y_j is the number of seats protected for classes.
* V_j(x) is the expected revenue from classes.
"""

import math
from dataclasses import dataclass
from statistics import NormalDist


@dataclass(frozen=True)
class FareClass:
    fare: float
    mean_demand: float
    sd_demand: float


def main() -> None:
    capacity = 100
    classes = [
        FareClass(fare=500, mean_demand=40, sd_demand=10),
        FareClass(fare=200, mean_demand=35, sd_demand=8),
        FareClass(fare=150, mean_demand=30, sd_demand=8),
    ]

    compare(classes, capacity)


def demand_probabilities(demand: NormalDist, capacity: int) -> list[float]:
    """
    Chance that exactly d people want to book.
    The last entry also covers demand above capacity.
    """
    probabilities = [demand.cdf(0)]
    for d in range(1, capacity):
        probabilities.append(demand.cdf(d) - demand.cdf(d - 1))
    probabilities.append(1 - demand.cdf(capacity - 1))
    return probabilities


def prob_at_least(demand: NormalDist, seats: int) -> float:
    """
    Chance that at least `seats` people want to book.
    Demand is a whole number, so at least `seats` means more than seats - 1.
    """
    return 1 - demand.cdf(seats - 1)


def littlewood_level(
    high_fare: float, demand: NormalDist, lower_fare: float, capacity: int
) -> int:
    """
    Number of seats to protect for the higher fare. Keep a seat while the
    higher fare times the chance it sells is more than the lower fare.
    """
    for seats in range(capacity, 0, -1):
        if lower_fare < high_fare * prob_at_least(demand, seats):
            return seats
    return 0


def littlewood(classes: list[FareClass], capacity: int) -> list[int]:
    # NOTE: Littlewood's rule is a two-class rule (only classes 1 and 2 used)
    high = classes[0]
    lower = classes[1]
    demand = NormalDist(high.mean_demand, high.sd_demand)
    return [littlewood_level(high.fare, demand, lower.fare, capacity)]


def emsr_a(classes: list[FareClass], capacity: int) -> list[int]:
    protection_levels: list[int] = []

    for j in range(1, len(classes)):
        higher = classes[:j]
        lower_fare = classes[j].fare

        total = 0
        for fare_class in higher:
            demand = NormalDist(fare_class.mean_demand, fare_class.sd_demand)
            total += littlewood_level(fare_class.fare, demand, lower_fare, capacity)

        # NOTE: the separate levels can add up to more than capacity
        protection_levels.append(min(total, capacity))

    return protection_levels


def emsr_b(classes: list[FareClass], capacity: int) -> list[int]:
    protection_levels: list[int] = []

    for j in range(1, len(classes)):
        higher = classes[:j]
        lower_fare = classes[j].fare

        mean = 0.0
        variance = 0.0
        revenue = 0.0
        for fare_class in higher:
            mean += fare_class.mean_demand
            variance += fare_class.sd_demand**2
            revenue += fare_class.fare * fare_class.mean_demand

        demand = NormalDist(mean, math.sqrt(variance))
        weighted_fare = revenue / mean

        level = littlewood_level(weighted_fare, demand, lower_fare, capacity)
        protection_levels.append(level)

    return protection_levels


def optimal_policy(
    classes: list[FareClass], capacity: int
) -> tuple[list[int], float]:
    revenue_table: list[list[float]] = [[0.0] * (capacity + 1)]

    for fare_class in classes:
        demand = NormalDist(fare_class.mean_demand, fare_class.sd_demand)
        demand_probs = demand_probabilities(demand, capacity)
        higher_revenue = revenue_table[-1]
        revenue: list[float] = []

        for seats_left in range(capacity + 1):
            best_at_most = [higher_revenue[seats_left]]
            for sold in range(1, seats_left + 1):
                sell = fare_class.fare * sold + higher_revenue[seats_left - sold]
                best_at_most.append(max(best_at_most[-1], sell))

            expected = 0.0
            for d, prob in enumerate(demand_probs):
                expected += prob * best_at_most[min(d, seats_left)]
            revenue.append(expected)
        revenue_table.append(revenue)

    protection_levels: list[int] = []
    for j in range(1, len(classes)):
        v_j = revenue_table[j]
        lower_fare = classes[j].fare

        level = 0
        for seats in range(capacity, 0, -1):
            if lower_fare < v_j[seats] - v_j[seats - 1]:
                level = seats
                break
        protection_levels.append(level)

    return protection_levels, revenue_table[-1][-1]


def compare(classes: list[FareClass], capacity: int) -> None:
    dp_levels, dp_revenue = optimal_policy(classes, capacity)
    results = {
        "Littlewood": littlewood(classes, capacity),
        "EMSR-a": emsr_a(classes, capacity),
        "EMSR-b": emsr_b(classes, capacity),
        "DP (optimal)": dp_levels,
    }

    for name, protection_levels in results.items():
        print(name)
        for j, level in enumerate(protection_levels, start=1):
            print(f"  y_{j} = {level}")
    print(f"  V_{len(classes)}({capacity}) = {dp_revenue:.2f}")


if __name__ == "__main__":
    main()
