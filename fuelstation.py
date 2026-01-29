import heapq
from typing import List, Dict, Optional, Tuple


# =========================================================
# STATE REPRESENTATION
# =========================================================
class FuelState:
    """
    Represents the current fuel quantities in two tanks:
    - tank_x → fuel in Tank X
    - tank_y → fuel in Tank Y
    """

    def __init__(self, x: int, y: int):
        self.tank_x = x
        self.tank_y = y

    def __eq__(self, other):
        """States are equal if both tank levels match"""
        return (
            isinstance(other, FuelState) and
            self.tank_x == other.tank_x and
            self.tank_y == other.tank_y
        )

    def __hash__(self):
        """
        Allows FuelState to be used as a dictionary key
        (required for A* bookkeeping)
        """
        return hash((self.tank_x, self.tank_y))

    def __repr__(self):
        """Readable output for printing solution paths"""
        return f"({self.tank_x}L , {self.tank_y}L)"


# =========================================================
# A* SEARCH STORAGE
# =========================================================

# Stores the parent of each visited state
parent_state: Dict[FuelState, Optional[FuelState]] = {}

# Stores the cheapest known cost to reach each state
cost_so_far: Dict[FuelState, int] = {}


# =========================================================
# HEURISTIC FUNCTION
# Remaining fuel mismatch
# =========================================================
def fuel_mismatch(state: FuelState, target_octane: int) -> int:
    """
    Heuristic for A*:
    Estimates how close the current state is to the target.
    We measure the minimum difference between either tank
    and the desired octane level.
    """
    return min(
        abs(state.tank_x - target_octane),
        abs(state.tank_y - target_octane)
    )


# =========================================================
# POSSIBLE FUEL OPERATIONS (STATE TRANSITIONS)
# =========================================================
def generate_transitions(
    state: FuelState,
    cap_x: int,
    cap_y: int
) -> List[FuelState]:
    """
    Generates all valid next states from the current state
    using allowed fuel operations.
    """

    next_states = []

    # 1️⃣ Load Tank X completely
    next_states.append(FuelState(cap_x, state.tank_y))

    # 2️⃣ Load Tank Y completely
    next_states.append(FuelState(state.tank_x, cap_y))

    # 3️⃣ Drain Tank X
    next_states.append(FuelState(0, state.tank_y))

    # 4️⃣ Drain Tank Y
    next_states.append(FuelState(state.tank_x, 0))

    # 5️⃣ Transfer fuel from X → Y
    move_xy = min(state.tank_x, cap_y - state.tank_y)
    next_states.append(
        FuelState(
            state.tank_x - move_xy,
            state.tank_y + move_xy
        )
    )

    # 6️⃣ Transfer fuel from Y → X
    move_yx = min(state.tank_y, cap_x - state.tank_x)
    next_states.append(
        FuelState(
            state.tank_x + move_yx,
            state.tank_y - move_yx
        )
    )

    return next_states


# =========================================================
# PATH RECONSTRUCTION
# =========================================================
def reconstruct_solution(goal: FuelState) -> List[FuelState]:
    """
    Reconstructs the solution path from start → goal
    using parent_state mapping.
    """
    path = []

    while goal is not None:
        path.append(goal)
        goal = parent_state[goal]

    # Reverse path (constructed from goal → start)
    path.reverse()
    return path


# =========================================================
# A* SEARCH: FUEL BLENDING OPTIMIZATION
# =========================================================
def optimize_blending(
    capacity_x: int,
    capacity_y: int,
    target_octane: int
) -> Optional[List[FuelState]]:
    """
    Uses A* search to find the optimal sequence of
    fuel blending operations.
    """

    # Priority queue stores (priority, state)
    open_set: List[Tuple[int, FuelState]] = []

    start = FuelState(0, 0)

    # Reset A* storage (important if function is reused)
    parent_state.clear()
    cost_so_far.clear()

    # Initialize A*
    heapq.heappush(open_set, (0, start))
    parent_state[start] = None
    cost_so_far[start] = 0

    while open_set:
        _, current = heapq.heappop(open_set)

        # 🎯 Goal check
        if (
            current.tank_x == target_octane or
            current.tank_y == target_octane
        ):
            return reconstruct_solution(current)

        # Explore all possible transitions
        for next_state in generate_transitions(
            current, capacity_x, capacity_y
        ):
            new_cost = cost_so_far[current] + 1

            # If state is new or we found a cheaper path
            if (
                next_state not in cost_so_far or
                new_cost < cost_so_far[next_state]
            ):
                cost_so_far[next_state] = new_cost
                parent_state[next_state] = current

                priority = new_cost + fuel_mismatch(
                    next_state, target_octane
                )

                heapq.heappush(open_set, (priority, next_state))

    # No solution found
    return None


# =========================================================
# GCD UTILITY FUNCTION
# =========================================================
def gcd(a: int, b: int) -> int:
    """
    Computes the Greatest Common Divisor using
    the Euclidean algorithm.
    """
    while b != 0:
        a, b = b, a % b
    return a


# =========================================================
# MAIN DRIVER
# =========================================================
def main():
    print("\n⛽ Automated Fuel Blending Optimization System")

    capacity_x = int(input("Enter capacity of Fuel Tank X: "))
    capacity_y = int(input("Enter capacity of Fuel Tank Y: "))
    target_octane = int(input("Enter required octane level: "))

    # Feasibility check 1: capacity constraint
    if target_octane > max(capacity_x, capacity_y):
        print("\n❌ Target octane level exceeds system limits.")
        return

    # Feasibility check 2: mathematical solvability
    if target_octane % gcd(capacity_x, capacity_y) != 0:
        print("\n❌ No feasible blending configuration exists.")
        return

    solution = optimize_blending(
        capacity_x, capacity_y, target_octane
    )

    if solution is None:
        print("\n❌ Optimization failed.")
    else:
        print("\n✅ Optimized Fuel States (Tank X , Tank Y):")
        for state in solution:
            print(state)


if __name__ == "__main__":
    main()
