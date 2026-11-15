"""
Optimal nested protection levels by dynamic programming (Section 5).
"""


import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FareClass:
    """A fare class with normally distributed demand"""

    fare: float
    mean_demand: float
    sd_demand: float

    def demand_probabilities(self, capacity: int) -> list[float]:
        def normal_cdf(value: float) -> float:
            z = (value - self.mean_demand) / (self.sd_demand * math.sqrt(2))
            return 0.5 * (1 + math.erf(z))

        probabilities = [normal_cdf(0)]
        for demand in range(1, capacity):
            probabilities.append(normal_cdf(demand) - normal_cdf(demand - 1))
        probabilities.append(1 - normal_cdf(capacity - 1))
        return probabilities


@dataclass(frozen=True)
class Policy:
    """The optimal policy found by the DP"""

    revenue_table: list[list[float]]
    protection_levels: list[int]

    @property
    def expected_revenue(self) -> float:
        return self.revenue_table[-1][-1]



def main() -> None:
    capacity = 100
    classes = [
        FareClass(fare=500, mean_demand=40, sd_demand=10),
        FareClass(fare=200, mean_demand=35, sd_demand=8),
        FareClass(fare=150, mean_demand=30, sd_demand=8),
    ]

    policy = optimal_policy(classes, capacity)
    for j, level in enumerate(policy.protection_levels, start=1):
        print(f"y_{j} = {level}")
    print(f"V_{len(classes)}({capacity}) = {policy.expected_revenue:.2f}")


def optimal_policy(classes: list[FareClass], capacity: int) -> Policy:
    revenue_table: list[list[float]] = [[0.0] * (capacity + 1)]

    for fare_class in classes:
        demand_probs = fare_class.demand_probabilities(capacity)
        higher_revenue = revenue_table[-1]
        revenue: list[float] = []

        for seats_left in range(capacity + 1):

            best_up_to = [higher_revenue[seats_left]]
            for sold in range(1, seats_left + 1):
                sell = fare_class.fare * sold + higher_revenue[seats_left - sold]
                best_up_to.append(max(best_up_to[-1], sell))

            revenue.append(sum(
                prob * best_up_to[min(demand, seats_left)]
                for demand, prob in enumerate(demand_probs)
            ))
        revenue_table.append(revenue)

    protection_levels = [
        protection_level(revenue_table[j], classes[j].fare)
        for j in range(1, len(classes))
    ]

    return Policy(revenue_table, protection_levels)


def protection_level(revenue: list[float], next_fare: float) -> int:
    for seats_left in range(len(revenue) - 1, 0, -1):
        if next_fare < revenue[seats_left] - revenue[seats_left - 1]:
            return seats_left
    return 0


if __name__ == "__main__":
    main()
