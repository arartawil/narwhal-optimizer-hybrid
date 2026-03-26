"""
Vehicle Routing Problem (VRP) Implementation
Minimize total travel distance/time for delivery fleet
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

class VehicleRoutingProblem:
    """
    Capacitated Vehicle Routing Problem (CVRP)
    
    Objective: Minimize total distance traveled by all vehicles
    Constraints:
        - Each customer visited exactly once
        - Vehicle capacity not exceeded
        - All routes start and end at depot
    """
    
    def __init__(self, n_customers=20, n_vehicles=3, vehicle_capacity=100, seed=42):
        """
        Initialize VRP instance
        
        Parameters:
        -----------
        n_customers : int
            Number of customers (excluding depot)
        n_vehicles : int
            Number of available vehicles
        vehicle_capacity : float
            Maximum capacity per vehicle
        seed : int
            Random seed for reproducibility
        """
        np.random.seed(seed)
        self.n_customers = n_customers
        self.n_vehicles = n_vehicles
        self.vehicle_capacity = vehicle_capacity
        
        # Generate customer locations (depot at origin)
        self.depot = np.array([50, 50])
        self.customers = np.random.uniform(0, 100, (n_customers, 2))
        
        # Generate customer demands (10-30 units)
        self.demands = np.random.randint(10, 31, n_customers)
        
        # Calculate distance matrix
        self.distance_matrix = self._calculate_distances()
        
    def _calculate_distances(self):
        """Calculate Euclidean distance matrix"""
        n = self.n_customers + 1  # +1 for depot
        dist = np.zeros((n, n))
        
        # Distances from depot
        for i in range(self.n_customers):
            dist[0, i+1] = np.linalg.norm(self.depot - self.customers[i])
            dist[i+1, 0] = dist[0, i+1]
        
        # Distances between customers
        for i in range(self.n_customers):
            for j in range(i+1, self.n_customers):
                dist[i+1, j+1] = np.linalg.norm(self.customers[i] - self.customers[j])
                dist[j+1, i+1] = dist[i+1, j+1]
        
        return dist
    
    def decode_solution(self, x):
        """
        Decode continuous solution vector to routes
        
        Parameters:
        -----------
        x : array-like, shape (n_customers,)
            Continuous values [0,1] representing customer priorities
            
        Returns:
        --------
        routes : list of lists
            Vehicle routes (customer indices)
        """
        # Sort customers by priority values
        customer_order = np.argsort(x)
        
        routes = []
        current_route = []
        current_load = 0
        
        for customer_idx in customer_order:
            customer_demand = self.demands[customer_idx]
            
            # Check if adding customer exceeds capacity
            if current_load + customer_demand <= self.vehicle_capacity:
                current_route.append(customer_idx)
                current_load += customer_demand
            else:
                # Start new route
                if current_route:
                    routes.append(current_route)
                current_route = [customer_idx]
                current_load = customer_demand
        
        # Add last route
        if current_route:
            routes.append(current_route)
        
        return routes
    
    def calculate_route_distance(self, route):
        """Calculate total distance for a single route"""
        if not route:
            return 0
        
        distance = 0
        # Depot to first customer
        distance += self.distance_matrix[0, route[0] + 1]
        
        # Between customers
        for i in range(len(route) - 1):
            distance += self.distance_matrix[route[i] + 1, route[i+1] + 1]
        
        # Last customer to depot
        distance += self.distance_matrix[route[-1] + 1, 0]
        
        return distance
    
    def evaluate(self, x):
        """
        Evaluate VRP solution
        
        Returns total distance with penalties for constraint violations
        """
        routes = self.decode_solution(x)
        
        total_distance = sum(self.calculate_route_distance(route) for route in routes)
        
        # Penalty for exceeding vehicle limit
        penalty = 0
        if len(routes) > self.n_vehicles:
            penalty += 1e6 * (len(routes) - self.n_vehicles)**2
        
        # Penalty for unvisited customers
        visited = set()
        for route in routes:
            visited.update(route)
        
        unvisited = self.n_customers - len(visited)
        if unvisited > 0:
            penalty += 1e6 * unvisited**2
        
        return total_distance + penalty
    
    def visualize_solution(self, x, title="VRP Solution", save_path=None):
        """Visualize routes"""
        routes = self.decode_solution(x)
        
        plt.figure(figsize=(10, 10))
        
        # Plot depot
        plt.scatter(self.depot[0], self.depot[1], c='red', s=200, marker='s', 
                   label='Depot', zorder=3, edgecolors='black', linewidth=2)
        
        # Plot customers
        plt.scatter(self.customers[:, 0], self.customers[:, 1], c='blue', s=100, 
                   label='Customers', zorder=2, edgecolors='black', linewidth=1)
        
        # Plot routes with different colors
        colors = plt.cm.rainbow(np.linspace(0, 1, len(routes)))
        
        for idx, (route, color) in enumerate(zip(routes, colors)):
            route_coords = [self.depot]
            for customer_idx in route:
                route_coords.append(self.customers[customer_idx])
            route_coords.append(self.depot)
            
            route_coords = np.array(route_coords)
            plt.plot(route_coords[:, 0], route_coords[:, 1], 
                    c=color, linewidth=2, alpha=0.7, label=f'Vehicle {idx+1}')
            
            # Calculate route load
            route_load = sum(self.demands[c] for c in route)
            
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def get_solution_info(self, x):
        """Get detailed solution information"""
        routes = self.decode_solution(x)
        total_distance = sum(self.calculate_route_distance(route) for route in routes)
        
        info = {
            'total_distance': total_distance,
            'n_routes': len(routes),
            'routes': routes,
            'route_loads': [sum(self.demands[c] for c in route) for route in routes],
            'route_distances': [self.calculate_route_distance(route) for route in routes]
        }
        
        return info


def create_vrp_objective(vrp_instance):
    """Create objective function for optimization"""
    def objective(x):
        # Ensure x is in [0, 1] range
        x = np.clip(x, 0, 1)
        return vrp_instance.evaluate(x)
    
    return objective


# Example benchmark VRP instances
def get_benchmark_vrp(problem_name='small'):
    """
    Get benchmark VRP instances
    
    Available problems:
    - 'small': 20 customers, 3 vehicles (easy)
    - 'medium': 50 customers, 5 vehicles (moderate)
    - 'large': 100 customers, 10 vehicles (hard)
    """
    
    configs = {
        'small': {'n_customers': 20, 'n_vehicles': 3, 'vehicle_capacity': 100, 'seed': 42},
        'medium': {'n_customers': 50, 'n_vehicles': 5, 'vehicle_capacity': 120, 'seed': 42},
        'large': {'n_customers': 100, 'n_vehicles': 10, 'vehicle_capacity': 150, 'seed': 42}
    }
    
    if problem_name not in configs:
        raise ValueError(f"Unknown problem: {problem_name}. Choose from {list(configs.keys())}")
    
    return VehicleRoutingProblem(**configs[problem_name])


if __name__ == "__main__":
    # Test VRP implementation
    print("=" * 70)
    print("Vehicle Routing Problem (VRP) Test")
    print("=" * 70)
    
    vrp = get_benchmark_vrp('small')
    
    print(f"\nProblem Configuration:")
    print(f"  Customers: {vrp.n_customers}")
    print(f"  Vehicles: {vrp.n_vehicles}")
    print(f"  Vehicle Capacity: {vrp.vehicle_capacity}")
    print(f"  Total Demand: {np.sum(vrp.demands)}")
    
    # Test random solution
    x_random = np.random.rand(vrp.n_customers)
    fitness_random = vrp.evaluate(x_random)
    info_random = vrp.get_solution_info(x_random)
    
    print(f"\nRandom Solution:")
    print(f"  Total Distance: {info_random['total_distance']:.2f}")
    print(f"  Number of Routes: {info_random['n_routes']}")
    print(f"  Fitness (with penalties): {fitness_random:.2f}")
    
    for i, (route, load, dist) in enumerate(zip(info_random['routes'], 
                                                  info_random['route_loads'],
                                                  info_random['route_distances'])):
        print(f"  Route {i+1}: {len(route)} customers, Load: {load}/{vrp.vehicle_capacity}, Distance: {dist:.2f}")
    
    # Visualize
    vrp.visualize_solution(x_random, title="Random VRP Solution")
    
    print("\n" + "=" * 70)
    print("VRP Test Completed!")
    print("=" * 70)
