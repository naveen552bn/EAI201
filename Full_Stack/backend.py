from flask import Flask, jsonify, request, render_template
from collections import deque
from queue import PriorityQueue
import math

app = Flask(__name__)

# ---------------------------
# CAMPUS GRAPH DATA
# ---------------------------

# Convert edges list to adjacency list format
edges = [
    ("Main Gate", "ID Gate", 170),
    ("ID Gate", "Library", 70),
    ("Library", "Lawn Area", 100),
    ("Library", "Vendi", 30),
    ("Lawn Area", "ACB2", 60),
    ("ACB2", "Food court", 200),
    ("Food court", "Hostel", 150),
    ("Hostel", "Sports", 450),
    ("Lawn Area", "Vendi", 65),
    ("Vendi", "Cafe", 20),
]

# Build adjacency list from edges
campus_graph = {}
for edge in edges:
    node1, node2, weight = edge
    
    # Add node1 -> node2 connection
    if node1 not in campus_graph:
        campus_graph[node1] = []
    campus_graph[node1].append((node2, weight))
    
    # Add node2 -> node1 connection (bidirectional)
    if node2 not in campus_graph:
        campus_graph[node2] = []
    campus_graph[node2].append((node1, weight))

# Campus coordinates (latitude, longitude) - approximate values
campus_coordinates = {
    'Main Gate': (13.2200, 77.7539),
    'ID Gate': (13.2218, 77.7550),
    'Library': (13.221971, 77.75558),
    'Lawn Area': (13.222775, 77.755576),
    'Vendi': (13.222264, 77.755126),
    'ACB2': (13.22336, 77.755963),
    'Food court': (13.224758, 77.75725),
    'Hostel': (13.224195, 77.758613),
    'Sports': (13.22887, 77.7572),
    'Cafe': (13.22242, 77.755158),
}

# Building information
building_info = {
    'Main Gate': "Main Entry Gate to the campus",
    'ID Gate': "Security checkpoint for campus entry",
    'Library': "Central library with study areas",
    'Lawn Area': "Open lawn area for recreational activities",
    'Vendi': "Vending area and snack center",
    'ACB2': "Academic Block 2",
    'Food court': "Food court with multiple dining options",
    'Hostel': "Student hostel building",
    'Sports': "Sports and recreational area",
    'Cafe': "Main cafeteria for students and staff"
}

# Walking speed in meters per second
WALKING_SPEED_MPS = 1.4

# ---------------------------
# SEARCH ALGORITHMS
# ---------------------------

def bfs(start, target):
    """Breadth-First Search algorithm using weighted graph"""
    if start not in campus_graph or target not in campus_graph:
        return False, [], 0
    
    visited = set()
    queue = deque([(start, [start])])
    nodes_explored = 0
    
    while queue:
        nodes_explored += 1
        node, path = queue.popleft()
        
        if node == target:
            return True, path, nodes_explored
        
        if node not in visited:
            visited.add(node)
            for neighbor, _ in campus_graph.get(node, []):  # Ignore weight for BFS
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
    
    return False, [], nodes_explored

def dfs(start, target):
    """Depth-First Search algorithm using weighted graph"""
    if start not in campus_graph or target not in campus_graph:
        return False, [], 0
    
    visited = set()
    stack = [(start, [start])]
    nodes_explored = 0
    
    while stack:
        nodes_explored += 1
        node, path = stack.pop()
        
        if node == target:
            return True, path, nodes_explored
        
        if node not in visited:
            visited.add(node)
            # Add neighbors in reverse order for DFS
            neighbors = [neighbor for neighbor, _ in campus_graph.get(node, [])]
            for neighbor in reversed(neighbors):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))
    
    return False, [], nodes_explored

def ucs(start, target):
    """Uniform Cost Search algorithm"""
    if start not in campus_graph or target not in campus_graph:
        return False, [], 0, 0
    
    visited = set()
    priority_queue = PriorityQueue()
    priority_queue.put((0, start, [start]))  # (cost, node, path)
    nodes_explored = 0
    
    while not priority_queue.empty():
        nodes_explored += 1
        cost, node, path = priority_queue.get()
        
        if node == target:
            return True, path, nodes_explored, cost
        
        if node not in visited:
            visited.add(node)
            for neighbor, edge_cost in campus_graph.get(node, []):
                if neighbor not in visited:
                    new_cost = cost + edge_cost
                    priority_queue.put((new_cost, neighbor, path + [neighbor]))
    
    return False, [], nodes_explored, 0

def heuristic(node, target):
    """Heuristic function for A* (Euclidean distance based on coordinates)"""
    if node not in campus_coordinates or target not in campus_coordinates:
        return 0
    
    # Calculate approximate distance using coordinates
    lat1, lon1 = campus_coordinates[node]
    lat2, lon2 = campus_coordinates[target]
    
    # Simple distance approximation (not actual geodesic distance)
    return math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2) * 100000  # Scale factor

def a_star(start, target):
    """A* Search algorithm"""
    if start not in campus_graph or target not in campus_graph:
        return False, [], 0, 0
    
    visited = set()
    priority_queue = PriorityQueue()
    # (f_cost, g_cost, node, path)
    initial_h = heuristic(start, target)
    priority_queue.put((initial_h, 0, start, [start]))
    nodes_explored = 0
    
    while not priority_queue.empty():
        nodes_explored += 1
        f_cost, g_cost, node, path = priority_queue.get()
        
        if node == target:
            return True, path, nodes_explored, g_cost
        
        if node not in visited:
            visited.add(node)
            for neighbor, edge_cost in campus_graph.get(node, []):
                if neighbor not in visited:
                    new_g_cost = g_cost + edge_cost
                    new_f_cost = new_g_cost + heuristic(neighbor, target)
                    priority_queue.put((new_f_cost, new_g_cost, neighbor, path + [neighbor]))
    
    return False, [], nodes_explored, 0

def calculate_time(distance):
    """Calculate walking time in minutes"""
    return round((distance / WALKING_SPEED_MPS) / 60, 2)

def get_distance_between_nodes(node1, node2):
    """Get the distance between two connected nodes"""
    for neighbor, distance in campus_graph.get(node1, []):
        if neighbor == node2:
            return distance
    # Check reverse direction
    for neighbor, distance in campus_graph.get(node2, []):
        if neighbor == node1:
            return distance
    return 0  # Nodes not connected

def calculate_path_distance(path):
    """Calculate total distance for a path"""
    if len(path) < 2:
        return 0
    
    total_distance = 0
    for i in range(len(path) - 1):
        current_node = path[i]
        next_node = path[i + 1]
        distance = get_distance_between_nodes(current_node, next_node)
        total_distance += distance
    
    return total_distance

def generate_directions(path):
    """Generate step-by-step directions for the path"""
    directions = []
    
    if len(path) < 2:
        return directions
    
    for i in range(len(path) - 1):
        current = path[i]
        next_node = path[i + 1]
        distance = get_distance_between_nodes(current, next_node)
        
        direction_text = f"From {current} to {next_node} - Walk {distance} meters"
        
        # Add some contextual information based on locations
        if "Gate" in current and "ID Gate" in next_node:
            direction_text += " through the security checkpoint"
        elif "Cafe" in current or "Food court" in current:
            direction_text += " near dining areas"
        elif "Library" in current or "ACB2" in current:
            direction_text += " through academic areas"
        elif "Hostel" in current:
            direction_text += " in hostel zone"
        elif "Sports" in current:
            direction_text += " in sports area"
        
        directions.append(direction_text)
    
    return directions

# ---------------------------
# FLASK ROUTES
# ---------------------------

@app.route("/")
def index():
    """Render the main page with all locations"""
    locations = list(campus_coordinates.keys())
    return render_template("index.html", locations=locations)

@app.route("/api/buildings")
def get_buildings():
    """API endpoint to get all building information"""
    buildings = []
    for name, coords in campus_coordinates.items():
        buildings.append({
            "name": name,
            "coords": coords,
            "info": building_info.get(name, "No information available")
        })
    return jsonify(buildings)

@app.route("/find_path", methods=["POST"])
def find_path():
    """API endpoint for pathfinding"""
    try:
        data = request.get_json()
        start = data.get("start")
        goal = data.get("goal")
        algorithm = data.get("algorithm", "bfs").lower()
        
        if not start or not goal:
            return jsonify({"success": False, "error": "Start and goal locations are required"})
        
        if start not in campus_coordinates or goal not in campus_coordinates:
            return jsonify({"success": False, "error": "Invalid start or goal location"})
        
        # Execute the selected algorithm
        if algorithm == "bfs":
            found, path, nodes_explored = bfs(start, goal)
            distance = calculate_path_distance(path)
        elif algorithm == "dfs":
            found, path, nodes_explored = dfs(start, goal)
            distance = calculate_path_distance(path)
        elif algorithm == "ucs":
            found, path, nodes_explored, distance = ucs(start, goal)
        elif algorithm == "a_star":
            found, path, nodes_explored, distance = a_star(start, goal)
        else:
            return jsonify({"success": False, "error": "Invalid algorithm"})
        
        if not found or not path:
            return jsonify({"success": False, "error": "No path found between the selected locations"})
        
        # Generate directions
        directions = generate_directions(path)
        
        # Get coordinates for the path
        path_coords = [campus_coordinates[node] for node in path if node in campus_coordinates]
        
        return jsonify({
            "success": True,
            "path": path,
            "path_coords": path_coords,
            "distance": distance,
            "time": calculate_time(distance),
            "nodes_explored": nodes_explored,
            "directions": directions,
            "algorithm": algorithm
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": f"An error occurred: {str(e)}"})

@app.route("/api/algorithms")
def get_algorithms():
    """API endpoint to get available algorithms"""
    return jsonify({
        "algorithms": [
            {"id": "bfs", "name": "Breadth-First Search", "description": "Finds shortest path in number of steps"},
            {"id": "dfs", "name": "Depth-First Search", "description": "Explores deeply first, may not find shortest path"},
            {"id": "ucs", "name": "Uniform Cost Search", "description": "Finds shortest path by actual distance"},
            {"id": "a_star", "name": "A* Search", "description": "Optimized search with distance heuristics"}
        ]
    })

if __name__ == "__main__":
    app.run(debug=True)
