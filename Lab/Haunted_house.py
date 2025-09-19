import heapq
import math

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def diagonal(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

def greedy_bfs(grid, start, goal, heuristic, moves):
    rows, cols = len(grid), len(grid[0])
    visited = set()
    pq = [(heuristic(start, goal), start)]
    parent = {start: None}

    while pq:
        _, current = heapq.heappop(pq)
        if current == goal:
            return reconstruct_path(parent, goal), visited

        if current in visited:
            continue
        visited.add(current)

        for dx, dy in moves:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != 1:
                neighbor = (nx, ny)
                if neighbor not in visited:
                    parent[neighbor] = current
                    heapq.heappush(pq, (heuristic(neighbor, goal), neighbor))

    return None, visited

def astar(grid, start, goal, heuristic, moves):
    rows, cols = len(grid), len(grid[0])
    open_set = [(heuristic(start, goal), 0, start)]
    parent = {start: None}
    g_score = {start: 0}
    visited = set()

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(parent, goal), visited

        if current in visited:
            continue
        visited.add(current)

        for dx, dy in moves:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != 1:
                neighbor = (nx, ny)

            
                step_cost = 5 if grid[nx][ny] == 2 else 1
                tentative_g = g_score[current] + step_cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    parent[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

    return None, visited

def reconstruct_path(parent, goal):
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    return path[::-1]

def print_grid(grid, path, start, goal):
    display = [row[:] for row in grid]
    if path:
        for (x, y) in path:
            if (x, y) != start and (x, y) != goal:
                display[x][y] = '*'
    print("\nPath Visualization:")
    for row in display:
        print(" ".join(str(c) for c in row))

def show_result(name, path, visited, grid, start, goal):
    print(f"\n{name}:")
    if path is None:
        print("No path found")
    else:
        print("Path:", path)
        print("Visited nodes:", len(visited))
        print_grid(grid, path, start, goal)

if __name__ == "__main__":
                          # 0 = open, 1 = wall, 2 = ghost zone
    grid = [
        [0, 0, 0, 1, 0],  
        [1, 1, 2, 1, 0],  # ghost zone at (1,2)
        [0, 0, 2, 1, 0],  # ghost zone at (2,2)
        [1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ]

    start = (0, 0)
    goal = (1, 2)

    print("Start:", start, " Goal:", goal)

    moves = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]

    path, visited = greedy_bfs(grid, start, goal, manhattan, moves)
    show_result("Greedy BFS (Manhattan)", path, visited, grid, start, goal)

    path, visited = astar(grid, start, goal, manhattan, moves)
    show_result("A* (Manhattan)", path, visited, grid, start, goal)

    path, visited = astar(grid, start, goal, euclidean, moves)
    show_result("A* (Euclidean)", path, visited, grid, start, goal)

    path, visited = astar(grid, start, goal, diagonal, moves)
    show_result("A* (Diagonal)", path, visited, grid, start, goal)
