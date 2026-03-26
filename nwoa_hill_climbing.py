"""
NWOA-Hill Climbing Hybrid Algorithm

Combines Narwhal Optimization with Hill Climbing local search.
Hill Climbing performs iterative improvement through random perturbations.

Advantages:
- Simple and fast
- Works well with multimodal functions
- Good exploitation of local regions
- Low memory requirements

Reference:
- Russell, S., & Norvig, P. (2010). Artificial Intelligence: A Modern Approach.
"""

import numpy as np
import math
from typing import Callable, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class NWOA_HillClimbing:
    """
    Hybrid NWOA with Hill Climbing local search.
    
    Parameters
    ----------
    objective_function : callable
        Function to minimize
    dim : int
        Problem dimension
    lb, ub : float or array-like
        Search space bounds
    n_agents : int, default=30
        Population size
    max_iter : int, default=500
        Maximum iterations
    ls_frequency : int, default=10
        Apply local search every N iterations
    ls_candidates : int, default=3
        Number of top solutions to refine
    ls_max_evals : int, default=50
        Max evaluations per local search
    """
    
    def __init__(self, objective_function: Callable, dim: int, lb: float, ub: float,
                 n_agents: int = 30, max_iter: int = 500, ls_frequency: int = 10,
                 ls_candidates: int = 3, ls_max_evals: int = 50,
                 A: float = 1.0, k: float = 2*np.pi, omega: float = 2*np.pi,
                 delta: float = 0.01, lambda_decay: float = 0.001):
        
        self.objective_function = objective_function
        self.dim = dim
        self.lb = np.array([lb] * dim) if isinstance(lb, (int, float)) else np.array(lb)
        self.ub = np.array([ub] * dim) if isinstance(ub, (int, float)) else np.array(ub)
        self.n_agents = n_agents
        self.max_iter = max_iter
        self.ls_frequency = ls_frequency
        self.ls_candidates = min(ls_candidates, n_agents)
        self.ls_max_evals = ls_max_evals
        
        # NWOA parameters
        self.A = A
        self.k = k
        self.omega = omega
        self.delta = delta
        self.lambda_decay = lambda_decay
        
        self.agents = None
        self.fitness = None
        self.best_agent = None
        self.best_fitness = np.inf
        self.convergence_curve = []
        self.prey_energy = 1.0
        
    def initialize_population(self):
        """Initialize population randomly."""
        self.agents = np.random.uniform(self.lb, self.ub, (self.n_agents, self.dim))
        self.fitness = np.array([self.objective_function(agent) for agent in self.agents])
        best_idx = np.argmin(self.fitness)
        self.best_agent = self.agents[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]
        
    def bound_solution(self, solution: np.ndarray) -> np.ndarray:
        """Clip solution to bounds."""
        return np.clip(solution, self.lb, self.ub)
    
    def echolocation_phase(self, agent: np.ndarray, t: int) -> np.ndarray:
        """NWOA echolocation exploration."""
        wave_strength = self.A * np.exp(-self.delta * t)
        phase = self.k * np.linalg.norm(agent - self.best_agent) - self.omega * t
        wave_effect = wave_strength * np.sin(phase)
        random_walk = np.random.randn(self.dim)
        new_position = agent + wave_effect * (self.best_agent - agent) + \
                      np.random.rand() * random_walk * (self.ub - self.lb) / 10
        return self.bound_solution(new_position)
    
    def sonar_communication(self, agent: np.ndarray, t: int) -> np.ndarray:
        """NWOA sonar communication."""
        idx1, idx2 = np.random.choice(self.n_agents, 2, replace=False)
        agent1, agent2 = self.agents[idx1], self.agents[idx2]
        comm_strength = 1 - t / self.max_iter
        new_position = agent + comm_strength * (agent1 - agent2) * np.random.rand()
        return self.bound_solution(new_position)
    
    def tusk_stunning(self, agent: np.ndarray, t: int) -> np.ndarray:
        """NWOA tusk stunning exploitation."""
        stun_power = t / self.max_iter
        levy_step = self._levy_flight(self.dim)
        new_position = self.best_agent + stun_power * levy_step * \
                      (self.best_agent - agent) * self.prey_energy
        return self.bound_solution(new_position)
    
    def _levy_flight(self, dim: int, beta: float = 1.5) -> np.ndarray:
        """Generate Lévy flight step."""
        sigma_u = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                   (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.randn(dim) * sigma_u
        v = np.random.randn(dim)
        step = u / np.abs(v)**(1 / beta)
        return step
    
    def update_prey_energy(self, t: int):
        """Update prey energy."""
        self.prey_energy = np.exp(-self.lambda_decay * t)
    
    def hill_climbing_search(self, start_point: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Hill Climbing local search.
        Iteratively accepts better solutions through random perturbations.
        """
        current = start_point.copy()
        current_fitness = self.objective_function(current)
        
        step_size = 0.1 * (self.ub - self.lb)
        
        for _ in range(self.ls_max_evals - 1):
            # Generate neighbor with random perturbation
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            neighbor = self.bound_solution(current + perturbation)
            neighbor_fitness = self.objective_function(neighbor)
            
            # Accept if better (greedy)
            if neighbor_fitness < current_fitness:
                current = neighbor
                current_fitness = neighbor_fitness
                step_size *= 1.1  # Increase step on success
            else:
                step_size *= 0.9  # Decrease step on failure
            
            # Minimum step size threshold
            step_size = np.maximum(step_size, 1e-10 * (self.ub - self.lb))
        
        return current, current_fitness
    
    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """Run NWOA-Hill Climbing optimization."""
        self.initialize_population()
        self.convergence_curve = [self.best_fitness]
        
        for t in range(self.max_iter):
            self.update_prey_energy(t)
            
            # Update each agent with NWOA strategies
            for i in range(self.n_agents):
                if self.prey_energy > 0.5:
                    new_agent = self.echolocation_phase(self.agents[i], t)
                elif 0.2 < self.prey_energy <= 0.5:
                    new_agent = self.sonar_communication(self.agents[i], t)
                else:
                    new_agent = self.tusk_stunning(self.agents[i], t)
                
                new_fitness = self.objective_function(new_agent)
                
                if new_fitness < self.fitness[i]:
                    self.agents[i] = new_agent
                    self.fitness[i] = new_fitness
                    
                    if new_fitness < self.best_fitness:
                        self.best_agent = new_agent.copy()
                        self.best_fitness = new_fitness
            
            # Apply Hill Climbing local search periodically
            if (t + 1) % self.ls_frequency == 0:
                sorted_indices = np.argsort(self.fitness)
                best_indices = sorted_indices[:self.ls_candidates]
                
                for idx in best_indices:
                    refined_solution, refined_fitness = self.hill_climbing_search(self.agents[idx])
                    
                    if refined_fitness < self.fitness[idx]:
                        self.agents[idx] = refined_solution
                        self.fitness[idx] = refined_fitness
                        
                        if refined_fitness < self.best_fitness:
                            self.best_agent = refined_solution.copy()
                            self.best_fitness = refined_fitness
            
            self.convergence_curve.append(self.best_fitness)
        
        return self.best_agent, self.best_fitness, self.convergence_curve
