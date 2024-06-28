import numpy as np
from scipy.signal import freqz

class WaterStriderOptimization:
    def __init__(self, pop_size, dim, max_iter, inertia_weight, cognitive_coeff, social_coeff, bounds, omega_pass, omega_stop, alpha, tau, W, delta_pass, delta_stop):
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
        self.delta_pass = delta_pass
        self.delta_stop = delta_stop
        
        # Initialize population and velocities
        self.population = self.initialize_positions()
        self.velocities = np.random.uniform(-1, 1, (pop_size, dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(pop_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def initialize_positions(self):
        upper = self.bounds[1]
        lower = self.bounds[0]
        R = np.random.uniform(0, 1, (self.pop_size, self.dim))
        return upper + R * (upper - lower)

    def fitness(self, coeffs):
        num_points = 128
        num_pass = int(self.omega_pass * num_points)
        num_stop = num_points - num_pass
        desired_response = np.concatenate([
            np.ones(num_pass),
            np.zeros(num_stop)
        ])
        
        # Actual frequency response
        _, actual_response = freqz(coeffs, worN=num_points)
        
        # Ensure actual_response has the same length as desired_response
        actual_response = np.abs(actual_response)
        
        # Calculate the error function
        error = self.W(np.linspace(0, 1, num_points)) * (actual_response - desired_response)
        
        # Parks-McClellan error function
        error_pass = np.maximum(0, np.abs(error[:num_pass]) - self.delta_pass)
        error_stop = np.maximum(0, np.abs(error[num_pass:]) - self.delta_stop)
        error_function = np.max(error_pass) + np.max(error_stop)
        
        return error_function
    
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
max_iter = 100
inertia_weight = 0.5
cognitive_coeff = 1.5
social_coeff = 1.5
bounds = (-1, 1)
omega_pass = 0.3
omega_stop = 0.4
alpha = 0
tau = 10
W = lambda omega: 1  # Uniform weighting function as an example
delta_pass = 0.01  # Example pass-band ripple
delta_stop = 0.01  # Example stop-band ripple

wso = WaterStriderOptimization(pop_size, dim, max_iter, inertia_weight, cognitive_coeff, social_coeff, bounds, omega_pass, omega_stop, alpha, tau, W, delta_pass, delta_stop)
best_coeffs = wso.optimize()

print("Best FIR filter coefficients found:", best_coeffs)
