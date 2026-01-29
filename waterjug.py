import heapq
import math

def water_jug_a_star(capA, capB, goal):
    """
    Solves the Water Jug problem using A* Search
    Returns the solution path if possible, else None
    """

    # Start state (both jugs empty)
    start = (0, 0)

    # Priority Queue → (f = g + h, state)
    pq = []
    heapq.heappush(pq, (0, start))

    parent = {start: None}   # To reconstruct path
    cost = {start: 0}        # g(n): cost to reach each state

    while pq:
        _, (x, y) = heapq.heappop(pq)

        # Goal test
        if x == goal or y == goal:
            path = []
            curr = (x, y)
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            return path[::-1]

        # Generate next possible states
        next_states = [
            (capA, y),                        # Fill Jug A
            (x, capB),                        # Fill Jug B
            (0, y),                           # Empty Jug A
            (x, 0),                           # Empty Jug B
            (x - min(x, capB - y), y + min(x, capB - y)),  # Pour A → B
            (x + min(y, capA - x), y - min(y, capA - x))   # Pour B → A
        ]

        for nx, ny in next_states:
            new_cost = cost[(x, y)] + 1

            if (nx, ny) not in cost or new_cost < cost[(nx, ny)]:
                cost[(nx, ny)] = new_cost

                # Heuristic: distance from goal
                h = min(abs(nx - goal), abs(ny - goal))
                f = new_cost + h

                heapq.heappush(pq, (f, (nx, ny)))
                parent[(nx, ny)] = (x, y)

    return None


# -------------------------------
# USER INPUT SECTION
# -------------------------------

print("\n--- Water Jug Problem using A* Search ---")

capA = int(input("Enter capacity of Jug A: "))
capB = int(input("Enter capacity of Jug B: "))
goal = int(input("Enter target amount: "))

# -------------------------------
# INPUT VALIDATION
# -------------------------------

if goal > max(capA, capB):
    print("\n❌ Goal cannot be greater than both jug capacities.")
    exit()

if goal % math.gcd(capA, capB) != 0:
    print("\n❌ No solution exists (goal not divisible by GCD of capacities).")
    exit()

# -------------------------------
# SOLVING
# -------------------------------

solution = water_jug_a_star(capA, capB, goal)

if solution:
    print("\n✅ Solution Path (Jug A, Jug B):")
    for step in solution:
        print(step)
else:
    print("\n❌ No solution found.")
