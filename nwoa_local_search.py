"""
Hybrid Narwhal Optimization Algorithm with Local Search (NWOA-LS)

This hybrid algorithm combines the global exploration of NWOA with
local search refinement for improved exploitation. This is a memetic approach
that applies local search to the best solutions periodically.

Available Local Search Strategies:
1. Nelder-Mead Simplex (default) - Robust derivative-free optimization
2. Hill Climbing - Simple gradient-free local improvement
3. Pattern Search - Coordinate descent-based method
4. Random Local Search - Stochastic local exploration

Author: Hybrid Implementation
Date: 2026
"""

import numpy as np
from typing import Callable, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import minimize


class NWOA_LocalSearch:
    """
    Hybrid Narwhal Optimization Algorithm with Local Search (NWOA-LS)
    
    This memetic algorithm combines NWOA's global search with local refinement.
    Local search is applied periodically to exploit promising regions.
    
    Parameters
    ----------
    objective_function : callable
        The objective function to minimize.
    dim : int
        Dimension of the search space.
    lb : float or array-like
        Lower bound(s) of the search space.
    ub : float or array-like
        Upper bound(s) of the search space.
    n_agents : int, optional (default=30)
        Number of narwhal agents (population size).
    max_iter : int, optional (default=500)
        Maximum number of iterations.
    local_search_method : str, optional (default='nelder-mead')
        Local search strategy: 'nelder-mead', 'hill-climbing', 
        'pattern-search', or 'random'
    ls_frequency : int, optional (default=10)
        Apply local search every N iterations.
    ls_candidates : int, optional (default=3)
        Number of best solutions to refine with local search.
    ls_max_evals : int, optional (default=50)
        Maximum function evaluations for each local search.
    A, k, omega, delta, lambda_decay : float
        NWOA-specific parameters (see base NWOA documentation).
    """
    
    def __init__(
        self,
        objective_function: Callable,
        dim: int,
        lb: float,
        ub: float,
        n_agents: int = 30,
        max_iter: int = 500,
        local_search_method: str = 'nelder-mead',
        ls_frequency: int = 10,
        ls_candidates: int = 3,
        ls_max_evals: int = 50,
        A: float = 1.0,
        k: float = 2 * np.pi,
        omega: float = 2 * np.pi,
        delta: float = 0.01,
        lambda_decay: float = 0.001
    ):
        """Initialize the Hybrid NWOA-LS."""
        self.objective_function = objective_function
        self.dim = dim
        self.lb = np.array([lb] * dim) if isinstance(lb, (int, float)) else np.array(lb)
        self.ub = np.array([ub] * dim) if isinstance(ub, (int, float)) else np.array(ub)
        self.n_agents = n_agents
        self.max_iter = max_iter
        
        # Local search parameters
        self.local_search_method = local_search_method.lower()
        self.ls_frequency = ls_frequency
        self.ls_candidates = min(ls_candidates, n_agents)
        self.ls_max_evals = ls_max_evals
        
        # NWOA parameters
        self.A = A
        self.k = k
        self.omega = omega
        self.delta = delta
        self.lambda_decay = lambda_decay
        
        # Initialize population and tracking
        self.agents = None
        self.fitness = None
        self.best_agent = None
        self.best_fitness = np.inf
        self.convergence_curve = []
        self.prey_energy = 1.0
        self.function_evals = 0
        
    def initialize_population(self):
        """Initialize the narwhal population randomly within bounds."""
        self.agents = np.random.uniform(
            self.lb, self.ub, (self.n_agents, self.dim)
        )
        self.fitness = np.array([self.objective_function(agent) for agent in self.agents])
        self.function_evals += self.n_agents
        
        # Find best agent
        best_idx = np.argmin(self.fitness)
        self.best_agent = self.agents[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]
        
    def bound_solution(self, solution: np.ndarray) -> np.ndarray:
        """Ensure solution stays within bounds."""
        return np.clip(solution, self.lb, self.ub)
    
    def echolocation_phase(self, agent: np.ndarray, t: int) -> np.ndarray:
        """
        Phase 1: Echolocation-based exploration.
        Uses wave-based search inspired by narwhal sonar.
        """
        r = np.random.rand()
        wave_strength = self.A * np.exp(-self.delta * t)
        
        # Wave-based position update
        phase = self.k * np.linalg.norm(agent - self.best_agent) - self.omega * t
        wave_effect = wave_strength * np.sin(phase)
        
        # Random walk component
        random_walk = np.random.randn(self.dim)
        
        new_position = agent + wave_effect * (self.best_agent - agent) + \
                      r * random_walk * (self.ub - self.lb) / 10
        
        return self.bound_solution(new_position)
    
    def sonar_communication(self, agent: np.ndarray, t: int) -> np.ndarray:
        """
        Phase 2: Sonar-based communication and position sharing.
        Agents share information about promising regions.
        """
        # Select random agents for communication
        idx1, idx2 = np.random.choice(self.n_agents, 2, replace=False)
        agent1, agent2 = self.agents[idx1], self.agents[idx2]
        
        # Communication strength decreases with iteration
        comm_strength = 1 - t / self.max_iter
        
        # Position update based on neighbors
        new_position = agent + comm_strength * (agent1 - agent2) * np.random.rand()
        
        return self.bound_solution(new_position)
    
    def tusk_stunning(self, agent: np.ndarray, t: int) -> np.ndarray:
        """
        Phase 3: Tusk-based stunning (exploitation).
        Focused search near the best solution.
        """
        # Stunning effectiveness increases with iteration
        stun_power = t / self.max_iter
        
        # Focused attack on prey (best solution)
        levy_step = self._levy_flight(self.dim)
        new_position = self.best_agent + stun_power * levy_step * \
                      (self.best_agent - agent) * self.prey_energy
        
        return self.bound_solution(new_position)
    
    def _levy_flight(self, dim: int, beta: float = 1.5) -> np.ndarray:
        """Generate Lévy flight step for improved exploration."""
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                   (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.randn(dim) * sigma_u
        v = np.random.randn(dim)
        step = u / np.abs(v)**(1 / beta)
        return step
    
    def update_prey_energy(self, t: int):
        """Update prey energy over time."""
        self.prey_energy = np.exp(-self.lambda_decay * t)
    
    # ============= LOCAL SEARCH METHODS =============
    
    def nelder_mead_search(self, start_point: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Nelder-Mead simplex local search.
        Robust derivative-free optimization method.
        """
        def bounded_objective(x):
            x_bounded = self.bound_solution(x)
            return self.objective_function(x_bounded)
        
        try:
            result = minimize(
                bounded_objective,
                start_point,
                method='Nelder-Mead',
                options={'maxfev': self.ls_max_evals, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            self.function_evals += result.nfev
            return self.bound_solution(result.x), result.fun
        except:
            return start_point, self.objective_function(start_point)
    
    def hill_climbing_search(self, start_point: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Hill climbing local search.
        Simple iterative improvement with random perturbations.
        """
        current = start_point.copy()
        current_fitness = self.objective_function(current)
        self.function_evals += 1
        
        step_size = 0.1 * (self.ub - self.lb)
        
        for _ in range(self.ls_max_evals):
            # Generate neighbor
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            neighbor = self.bound_solution(current + perturbation)
            neighbor_fitness = self.objective_function(neighbor)
            self.function_evals += 1
            
            # Accept if better
            if neighbor_fitness < current_fitness:
                current = neighbor
                current_fitness = neighbor_fitness
                step_size *= 1.1  # Increase step size on success
            else:
                step_size *= 0.9  # Decrease step size on failure
            
            # Minimum step size
            step_size = np.maximum(step_size, 1e-10 * (self.ub - self.lb))
        
        return current, current_fitness
    
    def pattern_search(self, start_point: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Pattern search (coordinate descent).
        Systematic exploration along coordinate directions.
        """
        current = start_point.copy()
        current_fitness = self.objective_function(current)
        self.function_evals += 1
        
        step_size = 0.1 * (self.ub - self.lb)
        evals_used = 1
        
        while evals_used < self.ls_max_evals:
            improved = False
            
            for dim_idx in range(self.dim):
                if evals_used >= self.ls_max_evals:
                    break
                
                # Try positive direction
                trial = current.copy()
                trial[dim_idx] = min(trial[dim_idx] + step_size[dim_idx], self.ub[dim_idx])
                trial_fitness = self.objective_function(trial)
                self.function_evals += 1
                evals_used += 1
                
                if trial_fitness < current_fitness:
                    current = trial
                    current_fitness = trial_fitness
                    improved = True
                    continue
                
                # Try negative direction
                if evals_used >= self.ls_max_evals:
                    break
                    
                trial = current.copy()
                trial[dim_idx] = max(trial[dim_idx] - step_size[dim_idx], self.lb[dim_idx])
                trial_fitness = self.objective_function(trial)
                self.function_evals += 1
                evals_used += 1
                
                if trial_fitness < current_fitness:
                    current = trial
                    current_fitness = trial_fitness
                    improved = True
            
            # Adjust step size
            if improved:
                step_size *= 1.2
            else:
                step_size *= 0.5
            
            # Check minimum step size
            if np.all(step_size < 1e-10 * (self.ub - self.lb)):
                break
        
        return current, current_fitness
    
    def random_local_search(self, start_point: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Random local search.
        Explores nearby region with decreasing radius.
        """
        current = start_point.copy()
        current_fitness = self.objective_function(current)
        self.function_evals += 1
        
        for i in range(self.ls_max_evals - 1):
            # Decreasing search radius
            radius = (1 - i / self.ls_max_evals) * 0.2 * (self.ub - self.lb)
            
            # Random perturbation
            perturbation = np.random.uniform(-radius, radius, self.dim)
            candidate = self.bound_solution(current + perturbation)
            candidate_fitness = self.objective_function(candidate)
            self.function_evals += 1
            
            # Accept if better
            if candidate_fitness < current_fitness:
                current = candidate
                current_fitness = candidate_fitness
        
        return current, current_fitness
    
    def apply_local_search(self, solution: np.ndarray) -> Tuple[np.ndarray, float]:
        """Apply the selected local search method."""
        if self.local_search_method == 'nelder-mead':
            return self.nelder_mead_search(solution)
        elif self.local_search_method == 'hill-climbing':
            return self.hill_climbing_search(solution)
        elif self.local_search_method == 'pattern-search':
            return self.pattern_search(solution)
        elif self.local_search_method == 'random':
            return self.random_local_search(solution)
        else:
            raise ValueError(f"Unknown local search method: {self.local_search_method}")
    
    # ============= MAIN OPTIMIZATION =============
    
    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """
        Execute the hybrid NWOA-LS optimization.
        
        Returns
        -------
        best_agent : np.ndarray
            Best solution found.
        best_fitness : float
            Fitness of the best solution.
        convergence_curve : list
            Best fitness at each iteration.
        """
        # Initialize
        self.initialize_population()
        self.convergence_curve = [self.best_fitness]
        
        # Main optimization loop
        for t in range(self.max_iter):
            # Update prey energy
            self.update_prey_energy(t)
            
            # Update each agent
            for i in range(self.n_agents):
                # Select hunting strategy based on prey energy
                if self.prey_energy > 0.5:
                    # Exploration: Echolocation
                    new_agent = self.echolocation_phase(self.agents[i], t)
                elif 0.2 < self.prey_energy <= 0.5:
                    # Transition: Sonar communication
                    new_agent = self.sonar_communication(self.agents[i], t)
                else:
                    # Exploitation: Tusk stunning
                    new_agent = self.tusk_stunning(self.agents[i], t)
                
                # Evaluate new position
                new_fitness = self.objective_function(new_agent)
                self.function_evals += 1
                
                # Update agent if improved
                if new_fitness < self.fitness[i]:
                    self.agents[i] = new_agent
                    self.fitness[i] = new_fitness
                    
                    # Update global best
                    if new_fitness < self.best_fitness:
                        self.best_agent = new_agent.copy()
                        self.best_fitness = new_fitness
            
            # Apply local search periodically
            if (t + 1) % self.ls_frequency == 0:
                # Get indices of best candidates
                sorted_indices = np.argsort(self.fitness)
                best_indices = sorted_indices[:self.ls_candidates]
                
                # Apply local search to best candidates
                for idx in best_indices:
                    refined_solution, refined_fitness = self.apply_local_search(
                        self.agents[idx]
                    )
                    
                    # Update if improved
                    if refined_fitness < self.fitness[idx]:
                        self.agents[idx] = refined_solution
                        self.fitness[idx] = refined_fitness
                        
                        # Update global best
                        if refined_fitness < self.best_fitness:
                            self.best_agent = refined_solution.copy()
                            self.best_fitness = refined_fitness
            
            # Store convergence
            self.convergence_curve.append(self.best_fitness)
            
            # Optional: Print progress
            if (t + 1) % 50 == 0:
                print(f"Iteration {t+1}/{self.max_iter}, Best Fitness: {self.best_fitness:.6e}")
        
        return self.best_agent, self.best_fitness, self.convergence_curve


# ============= BENCHMARK FUNCTIONS FOR TESTING =============

def sphere(x):
    """Sphere function: f(x*) = 0 at x* = (0,...,0)"""
    return np.sum(x**2)

def rosenbrock(x):
    """Rosenbrock function: f(x*) = 0 at x* = (1,...,1)"""
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rastrigin(x):
    """Rastrigin function: f(x*) = 0 at x* = (0,...,0)"""
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


if __name__ == "__main__":
    print("=" * 70)
    print("NWOA-LocalSearch Hybrid Algorithm - Quick Test")
    print("=" * 70)
    
    # Test parameters
    dim = 10
    lb, ub = -100, 100
    n_agents = 30
    max_iter = 200
    
    # Test different local search methods
    methods = ['nelder-mead', 'hill-climbing', 'pattern-search', 'random']
    test_functions = [
        (sphere, "Sphere"),
        (rastrigin, "Rastrigin")
    ]
    
    for func, name in test_functions:
        print(f"\n{name} Function (Dim={dim}):")
        print("-" * 70)
        
        for method in methods:
            optimizer = NWOA_LocalSearch(
                objective_function=func,
                dim=dim,
                lb=lb,
                ub=ub,
                n_agents=n_agents,
                max_iter=max_iter,
                local_search_method=method,
                ls_frequency=10,
                ls_candidates=3,
                ls_max_evals=50
            )
            
            best_solution, best_fitness, convergence = optimizer.optimize()
            
            print(f"  {method:15s}: Best Fitness = {best_fitness:.6e} "
                  f"(FEs: {optimizer.function_evals})")
    
    print("\n" + "=" * 70)
    print("Test completed!")
