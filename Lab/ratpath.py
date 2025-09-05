from collections import deque
import heapq

graph = {
    "A": [("B", 1), ("C", 4)],
    "B": [("A", 1), ("D", 2), ("E", 5)],
    "C": [("A", 4), ("F", 3)],
    "D": [("B", 2), ("G", 6)],
    "E": [("B", 5), ("G", 1)],
    "F": [("C", 3), ("G", 2)],
    "G": [("D", 6), ("E", 1), ("F", 2)]
}

def dfs(start, goal):
    stack = [(start, [start], 0)]
    visited = set()
    while stack:
        node, path, cost = stack.pop()
        if node == goal:
            print("Path:", path)
            print("Total cost:", cost)
            return
        if node not in visited:
            visited.add(node)
            for neigh, c in graph.get(node, []):
                if neigh not in visited:
                    stack.append((neigh, path + [neigh], cost + c))
    print("No path found.")

def bfs(start, goal):
    queue = deque([(start, [start], 0)])
    visited = set()
    while queue:
        node, path, cost = queue.popleft()
        if node == goal:
            print("Path:", path)
            print("Total cost:", cost)
            return
        if node not in visited:
            visited.add(node)
            for neigh, c in graph.get(node, []):
                if neigh not in visited:
                    queue.append((neigh, path + [neigh], cost + c))
    print("No path found.")

def ucs(start, goal):
    pq = [(0, start, [start])]
    visited = set()
    while pq:
        cost, node, path = heapq.heappop(pq)
        if node == goal:
            print("Path:", path)
            print("Total cost:", cost)
            return
        if node not in visited:
            visited.add(node)
            for neigh, c in graph.get(node, []):
                if neigh not in visited:
                    heapq.heappush(pq, (cost + c, neigh, path + [neigh]))
    print("No path found.")

start = input("Enter starting node: ").upper()
goal = input("Enter target node: ").upper()
algo = input("Choose algorithm (DFS / BFS / UCS): ").upper()

if algo == "DFS":
    dfs(start, goal)
elif algo == "BFS":
    bfs(start, goal)
elif algo == "UCS":
    ucs(start, goal)
else:
    print("Invalid algorithm choice!")
