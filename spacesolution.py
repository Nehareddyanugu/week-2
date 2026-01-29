import heapq
from typing import List, Tuple, Dict, Optional


class FluidState:
    """
    Represents the amount of water in the two containers:
    - container_x → water in container X
    - container_y → water in container Y
    """

    def __init__(self, x: int, y: int):
        self.container_x = x
        self.container_y = y

    def __eq__(self, other):
        return (
            isinstance(other, FluidState) and
            self.container_x == other.container_x and
            self.container_y == other.container_y
        )

    def __hash__(self):
        # Required so FluidState can be used as dictionary keys
        return hash((self.container_x, self.container_y))

    def __repr__(self):
        return f"({self.container_x}L , {self.container_y}L)"


# -----------------------------------
# A* SEARCH SUPPORT STRUCTURES
# -----------------------------------

# Stores the parent of each state (for path reconstruction)
parent_map: Dict[FluidState, Optional[FluidState]] = {}

# Stores the cost from the start to each state
cost_so_far: Dict[FluidState, int] = {}


def heuristic(state: FluidState, target: int) -> int:
    """
    Heuristic function for A*:
    Estimates how close we are to the target amount.
    We take the minimum distance from either container.
    """
    return min(
        abs(state.container_x - target),
        abs(state.container_y - target)
    )


def generate_next_states(
    state: FluidState,
    cap_x: int,
    cap_y: int
) -> List[FluidState]:
    """
    Generates all possible valid next states from the current state.
    """

    states = []

    # 1️⃣ Fill container X completely
    states.append(FluidState(cap_x, state.container_y))

    # 2️⃣ Fill container Y completely
    states.append(FluidState(state.container_x, cap_y))

    # 3️⃣ Empty container X
    states.append(FluidState(0, state.container_y))

    # 4️⃣ Empty container Y
    states.append(FluidState(state.container_x, 0))

    # 5️⃣ Transfer from X → Y
    move_xy = min(state.container_x, cap_y - state.container_y)
    states.append(
        FluidState(
            state.container_x - move_xy,
            state.container_y + move_xy
        )
    )

    # 6️⃣ Transfer from Y → X
    move_yx = min(state.container_y, cap_x - state.container_x)
    states.append(
        FluidState(
            state.container_x + move_yx,
            state.container_y - move_yx
        )
    )

    return states


def reconstruct_path(goal: FluidState) -> List[FluidState]:
    """
    Reconstructs the path from the start state to the goal state
    using the parent_map.
    """
    path = []
    while goal is not None:
        path.append(goal)
        goal = parent_map[goal]

    # Reverse because we built it from goal → start
    path.reverse()
    return path


def calibrate_water(
    capacity_x: int,
    capacity_y: int,
    target: int
) -> Optional[List[FluidState]]:
    """
    Performs A* search to find a sequence of steps
    that results in exactly `target` liters in either container.
    """

    # Min-heap priority queue: (priority, state)
    priority_queue: List[Tuple[int, FluidState]] = []

    start = FluidState(0, 0)

    # Initialize structures
    heapq.heappush(priority_queue, (0, start))
    parent_map.clear()
    cost_so_far.clear()

    parent_map[start] = None
    cost_so_far[start] = 0

    while priority_queue:
        _, current = heapq.heappop(priority_queue)

        # 🎯 Goal check
        if (
            current.container_x == target or
            current.container_y == target
        ):
            return reconstruct_path(current)

        # Explore neighboring states
        for next_state in generate_next_states(
            current, capacity_x, capacity_y
        ):
            new_cost = cost_so_far[current] + 1

            # If new state is unvisited or cheaper path is found
            if (
                next_state not in cost_so_far or
                new_cost < cost_so_far[next_state]
            ):
                cost_so_far[next_state] = new_cost
                parent_map[next_state] = current

                # Priority = cost so far + heuristic estimate
                priority = new_cost + heuristic(next_state, target)
                heapq.heappush(
                    priority_queue,
                    (priority, next_state)
                )

    # No solution found
    return None


# -----------------------------------
# MAIN PROGRAM
# -----------------------------------

def main():
    print("\n🚀 Space Mission Water Calibration 🚀")

    capacity_x = int(input("Enter capacity of Container X: "))
    capacity_y = int(input("Enter capacity of Container Y: "))
    target = int(input("Enter required calibration amount: "))

    # Quick feasibility check
    if target > max(capacity_x, capacity_y):
        print("\n❌ Calibration impossible: target exceeds container limits.")
        return

    solution = calibrate_water(capacity_x, capacity_y, target)

    if solution is None:
        print("\n❌ No valid calibration sequence found.")
    else:
        print("\n✅ Calibration Steps (X , Y):")
        for step in solution:
            print(step)


if __name__ == "__main__":
    main()
