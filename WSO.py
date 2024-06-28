import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

class WaterStriderOptimization:
    def __init__(self, pop_size, dim, max_iter, inertia_weight, cognitive_coeff, social_coeff, bounds, omega_pass, omega_stop, alpha, tau, W, delta_pass, delta_stop, nte, ar):
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
        self.nte = nte  # Number of territories
        self.ar = ar  # Attraction response probability
        
        # Initialize population and velocities
        self.population = self.initialize_positions()
        self.velocities = np.random.uniform(-1, 1, (pop_size, dim//2))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(pop_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def initialize_positions(self):
        upper = self.bounds[1]
        lower = self.bounds[0]
        R = np.random.uniform(0, 1, (self.pop_size, self.dim//2))
        positions = lower + R * (upper - lower)
        return np.hstack([positions, positions[:, ::-1]])

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
        cognitive = self.cognitive_coeff * np.random.uniform(0, 1, self.dim//2) * (self.personal_best_positions[i][:self.dim//2] - self.population[i][:self.dim//2])
        social = self.social_coeff * np.random.uniform(0, 1, self.dim//2) * (self.global_best_position[:self.dim//2] - self.population[i][:self.dim//2])
        self.velocities[i] = inertia + cognitive + social
    
    def update_position(self, i):
        self.population[i][:self.dim//2] += self.velocities[i]
        self.population[i][:self.dim//2] = np.clip(self.population[i][:self.dim//2], self.bounds[0], self.bounds[1])
        self.population[i][self.dim//2:] = self.population[i][:self.dim//2][::-1]
    
    def establish_territories(self):
        territories = []
        territory_size = self.pop_size // self.nte
        for i in range(self.nte):
            start = i * territory_size
            end = (i + 1) * territory_size if i != self.nte - 1 else self.pop_size
            territory = self.population[start:end]
            territories.append(territory)
        return territories
    
    def mating_process(self, territory):
        best_fitness = np.inf
        keystone = None
        for i in range(len(territory)):
            fitness = self.fitness(territory[i])
            if fitness < best_fitness:
                best_fitness = fitness
                keystone = territory[i]
        
        for i in range(len(territory)):
            if np.random.rand() < self.ar:  # Attraction response
                Q = np.random.uniform(-1, 1, self.dim//2)
                territory[i][:self.dim//2] += Q * np.random.uniform(0, 1)
            else:  # Repulsion response
                Q = np.random.uniform(-1, 1, self.dim//2)
                territory[i][:self.dim//2] += Q * (1 + np.random.uniform(0, 1))
            
            territory[i][:self.dim//2] = np.clip(territory[i][:self.dim//2], self.bounds[0], self.bounds[1])  # Boundary check
            territory[i][self.dim//2:] = territory[i][:self.dim//2][::-1]
        return territory

    def feeding_process(self, territory):
        new_territory = []
        for strider in territory:
            current_fitness = self.fitness(strider)
            if self.global_best_position is not None:
                if current_fitness > self.global_best_score:  # Moves towards food
                    new_position = strider[:self.dim//2] + 2 * np.random.uniform(0, 1) * (self.global_best_position[:self.dim//2] - strider[:self.dim//2])
                else:  # Moves towards optimal foraging-habitat females
                    new_position = strider[:self.dim//2] + 2 * np.random.uniform(0, 1) * (self.global_best_position[:self.dim//2] - strider[:self.dim//2]) * (1 + np.random.uniform(0, 1))
                new_position = np.clip(new_position, self.bounds[0], self.bounds[1])  # Boundary check
                new_strider = np.hstack([new_position, new_position[::-1]])
                new_territory.append(new_strider)
            else:
                new_territory.append(strider)
        return new_territory

    def death_and_succession(self, territory):
        min_position = np.min(territory, axis=0)[:self.dim//2]
        max_position = np.max(territory, axis=0)[:self.dim//2]
        new_territory = []
        for strider in territory:
            current_fitness = self.fitness(strider)
            if current_fitness > self.global_best_score:  # Considered dead
                R = np.random.uniform(0, 1, self.dim//2)
                new_position = min_position + 2 * R * (max_position - min_position)
                new_position = np.clip(new_position, self.bounds[0], self.bounds[1])  # Boundary check
                new_strider = np.hstack([new_position, new_position[::-1]])
                new_territory.append(new_strider)
            else:
                new_territory.append(strider)
        return new_territory

    def optimize(self):
        for iteration in range(self.max_iter):
            territories = self.establish_territories()
            for idx, territory in enumerate(territories):
                territory = self.mating_process(territory)
                territory = self.feeding_process(territory)
                territory = self.death_and_succession(territory)
                for i in range(len(territory)):
                    fitness = self.fitness(territory[i])
                    if fitness < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = fitness
                        self.personal_best_positions[i] = territory[i]
                    if fitness < self.global_best_score:
                        self.global_best_score = fitness
                        self.global_best_position = territory[i]
            
            print(f"Iteration {iteration+1}/{self.max_iter}, Global Best Score: {self.global_best_score}")
            if iteration > 50 and abs(self.global_best_score - previous_global_best_score) < 1e-6:
                print("Convergence achieved.")
                break
            previous_global_best_score = self.global_best_score
        
        return self.global_best_position

    def plot_frequency_response(self, coeffs):
        """
        Plots the frequency response of the FIR filter defined by the coefficients.
        """
        # Compute the frequency response
        w, h = freqz(coeffs, worN=8000)
        # Normalize frequency to π rad/sample and calculate magnitude
        w = w / np.pi
        h_mag = np.abs(h)

        # Create the plot
        plt.figure(figsize=(10, 5))
        plt.plot(w, h_mag, label='Frequency Response')
        plt.title('Frequency Response of FIR Filter')
        plt.xlabel('Normalized Frequency (×π rad/sample)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.axvline(self.omega_pass, color='r', linestyle='--', label='Passband Edge')
        plt.axvline(self.omega_stop, color='g', linestyle='--', label='Stopband Edge')
        plt.legend()
        plt.show()

# Example usage
pop_size = 50
dim = 21  # Number of coefficients in the FIR filter
max_iter = 100
inertia_weight = 0.5
cognitive_coeff = 1.5
social_coeff = 1.5
bounds = (-1, 1)
omega_pass = 0.44
omega_stop = 0.55
alpha = 0
tau = 10
W = lambda omega: 1  # Uniform weighting function as an example
delta_pass = 0.01  # Example pass-band ripple
delta_stop = 0.01  # Example stop-band ripple
nte = 5  # Number of territories
ar = 0.7  # Attraction response probability

wso = WaterStriderOptimization(pop_size, dim, max_iter, inertia_weight, cognitive_coeff, social_coeff, bounds, omega_pass, omega_stop, alpha, tau, W, delta_pass, delta_stop, nte, ar)
best_coeffs = wso.optimize()

print("Best FIR filter coefficients found:", best_coeffs)

# Plot the low-pass filter frequency response of the best filter coefficients
wso.plot_frequency_response(best_coeffs)
