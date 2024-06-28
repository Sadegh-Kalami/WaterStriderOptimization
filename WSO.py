import numpy as np
from scipy.signal import freqz

class WaterStriderOptimization:
    def __init__(self, pop_size, dim, max_iter, inertia_weight, cognitive_coeff, social_coeff, bounds, omega_pass, omega_stop, alpha, tau, W):
        self.pop_size = pop_size
        self.dim = dim
        self.max_iter = max_iter
        self.inertia_weight = inertia_weight
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.bounds = bounds
        self.omega_pass = omega_pass
        self.omega_stop = omega_stop
        self.alpha = alpha
        self.tau = tau
        self.W = W
        
        # Initialize population and velocities
        self.population = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))
        self.velocities = np.random.uniform(-1, 1, (pop_size, dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(pop_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def fitness(self, coeffs):
        # Example fitness function: minimize the error between desired and actual frequency response
        desired = np.ones(128)
        _, response = freqz(coeffs, worN=128)
        error = np.mean((np.abs(response) - desired) ** 2)
        return error
    
    def update_velocity(self, i):
        inertia = self.inertia_weight * self.velocities[i]
        cognitive = self.cognitive_coeff * np.random.uniform(0, 1, self.dim) * (self.personal_best_positions[i] - self.population[i])
        social = self.social_coeff * np.random.uniform(0, 1, self.dim) * (self.global_best_position - self.population[i])
        self.velocities[i] = inertia + cognitive + social
    
    def update_position(self, i):
        self.population[i] += self.velocities[i]
        # Boundary check
        self.population[i] = np.clip(self.population[i], self.bounds[0], self.bounds[1])
    
    def optimize(self):
        for iteration in range(self.max_iter):
            for i in range(self.pop_size):
                fitness = self.fitness(self.population[i])
                if fitness < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness
                    self.personal_best_positions[i] = self.population[i]
                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best_position = self.population[i]
            
            for i in range(self.pop_size):
                self.update_velocity(i)
                self.update_position(i)
            
            print(f"Iteration {iteration+1}/{self.max_iter}, Global Best Score: {self.global_best_score}")
        
        return self.global_best_position

# Example usage
pop_size = 30
dim = 20  # Number of coefficients in the FIR filter
max_iter = 10
inertia_weight = 0.5
cognitive_coeff = 1.5
social_coeff = 1.5
bounds = (-1, 1)
omega_pass = 0.3
omega_stop = 0.4
alpha = 0
tau = 10
W = lambda omega: 1  # Uniform weighting function as an example

wso = WaterStriderOptimization(pop_size, dim, max_iter, inertia_weight, cognitive_coeff, social_coeff, bounds, omega_pass, omega_stop, alpha, tau, W)
best_coeffs = wso.optimize()

print("Best FIR filter coefficients found:", best_coeffs)
