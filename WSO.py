import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

class WaterStriderOptimization:
    def __init__(self, pop_size, dim, max_iter, inertia_weight, cognitive_coeff, social_coeff, bounds, omega_pass, omega_stop, alpha, tau, W, delta_pass, delta_stop, nte, ar, filter_type='lowpass', initial_population=None):
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
        self.nte = nte
        self.ar = ar
        self.filter_type = filter_type  # Add filter type parameter
        self.convergence_count = 0
        self.previous_global_best_score = np.inf

        self.population = self.initialize_positions(initial_population)
        self.velocities = np.random.uniform(-1, 1, (pop_size, (dim + 1) // 2))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(pop_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def initialize_positions(self, initial_population):
        if initial_population is not None:
            population = np.tile(initial_population, (self.pop_size, 1))
        else:
            upper = self.bounds[1]
            lower = self.bounds[0]
            R = np.random.uniform(0, 1, (self.pop_size, (self.dim + 1) // 2))
            positions = lower + R * (upper - lower)
            if self.dim % 2 == 0:
                population = np.hstack([positions, positions[:, ::-1]])
            else:
                population = np.hstack([positions[:, :-1], positions[:, ::-1]])
        return population

    def fitness(self, coeffs):
        num_points = 128
        num_pass = int(self.omega_pass * num_points)
        num_stop = int(self.omega_stop * num_points)
        desired_response = np.zeros(num_points)

        if self.filter_type == 'lowpass':
            desired_response[:num_pass] = 1
        elif self.filter_type == 'highpass':
            desired_response[num_stop:] = 1
        elif self.filter_type == 'bandpass':
            desired_response[num_pass:num_stop] = 1
        elif self.filter_type == 'bandstop':
            desired_response[:num_pass] = 1
            desired_response[num_stop:] = 1

        _, actual_response = freqz(coeffs, worN=num_points)
        actual_response = np.abs(actual_response)
        
        error = self.W(np.linspace(0, 1, num_points)) * (actual_response - desired_response)
        
        error_pass = np.maximum(0, np.abs(error[:num_pass]) - self.delta_pass)
        error_stop = np.maximum(0, np.abs(error[num_stop:]) - self.delta_stop)
        error_function = np.max(error_pass) + np.max(error_stop)
        
        return error_function
    
    def update_velocity(self, i):
        inertia = self.inertia_weight * self.velocities[i]
        cognitive = self.cognitive_coeff * np.random.uniform(0, 1, (self.dim + 1) // 2) * (self.personal_best_positions[i][: (self.dim + 1) // 2] - self.population[i][: (self.dim + 1) // 2])
        social = self.social_coeff * np.random.uniform(0, 1, (self.dim + 1) // 2) * (self.global_best_position[: (self.dim + 1) // 2] - self.population[i][: (self.dim + 1) // 2])
        self.velocities[i] = inertia + cognitive + social
    
    def update_position(self, i):
        self.population[i][: (self.dim + 1) // 2] += self.velocities[i]
        self.population[i][: (self.dim + 1) // 2] = np.clip(self.population[i][: (self.dim + 1) // 2], self.bounds[0], self.bounds[1])
        if self.dim % 2 == 0:
            self.population[i][(self.dim + 1) // 2:] = self.population[i][: (self.dim + 1) // 2][::-1]
        else:
            self.population[i][(self.dim + 1) // 2:] = self.population[i][: (self.dim + 1) // 2 - 1][::-1]
    
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
            Q = np.random.uniform(-1, 1, (self.dim + 1) // 2)
            if np.random.rand() < self.ar:  # Attraction response
                territory[i][: (self.dim + 1) // 2] += Q * np.random.uniform(0, 1)
            else:  # Repulsion response
                territory[i][: (self.dim + 1) // 2] += Q * (1 + np.random.uniform(0, 1))
            
            territory[i][: (self.dim + 1) // 2] = np.clip(territory[i][: (self.dim + 1) // 2], self.bounds[0], self.bounds[1])
            if self.dim % 2 == 0:
                territory[i][(self.dim + 1) // 2:] = territory[i][: (self.dim + 1) // 2][::-1]
            else:
                territory[i][(self.dim + 1) // 2:] = territory[i][: (self.dim + 1) // 2 - 1][::-1]
        return territory

    def feeding_process(self, territory):
        new_territory = []
        for strider in territory:
            current_fitness = self.fitness(strider)
            if self.global_best_position is not None:
                if current_fitness > self.global_best_score:  # Moves towards food
                    new_position = strider[: (self.dim + 1) // 2] + 2 * np.random.uniform(0, 1) * (self.global_best_position[: (self.dim + 1) // 2] - strider[: (self.dim + 1) // 2])
                else:  # Moves towards optimal foraging-habitat females
                    new_position = strider[: (self.dim + 1) // 2] + 2 * np.random.uniform(0, 1) * (self.global_best_position[: (self.dim + 1) // 2] - strider[: (self.dim + 1) // 2]) * (1 + np.random.uniform(0, 1))
                new_position = np.clip(new_position, self.bounds[0], self.bounds[1])
                if self.dim % 2 == 0:
                    new_strider = np.hstack([new_position, new_position[::-1]])
                else:
                    new_strider = np.hstack([new_position[:-1], new_position[::-1]])
                new_territory.append(new_strider)
            else:
                new_territory.append(strider)
        return new_territory

    def death_and_succession(self, territory):
        min_position = np.min(territory, axis=0)[: (self.dim + 1) // 2]
        max_position = np.max(territory, axis=0)[: (self.dim + 1) // 2]
        new_territory = []
        for strider in territory:
            current_fitness = self.fitness(strider)
            if current_fitness > self.global_best_score:  # Considered dead
                R = np.random.uniform(0, 1, (self.dim + 1) // 2)
                new_position = min_position + 2 * R * (max_position - min_position)
                new_position = np.clip(new_position, self.bounds[0], self.bounds[1])
                if self.dim % 2 == 0:
                    new_strider = np.hstack([new_position, new_position[::-1]])
                else:
                    new_strider = np.hstack([new_position[:-1], new_position[::-1]])
                new_territory.append(new_strider)
            else:
                new_territory.append(strider)
        return new_territory

    def optimize(self):
        self.convergence_count = 0
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

            if abs(self.global_best_score - self.previous_global_best_score) < 1e-6:
                self.convergence_count += 1
            else:
                self.convergence_count = 0

            if self.convergence_count >= 5:
                print("Convergence achieved over 5 consecutive iterations.")
                break

            self.previous_global_best_score = self.global_best_score
                
        return self.global_best_position

    def plot_frequency_response(self, coeffs, ax=None):
        w, h = freqz(coeffs, worN=8000)
        w = w / np.pi
        h_mag = np.abs(h)

        if ax is None:
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
        else:
            ax.plot(w, h_mag, label='Frequency Response')
            ax.set_title('Frequency Response of FIR Filter')
            ax.set_xlabel('Normalized Frequency (×π rad/sample)')
            ax.set_ylabel('Magnitude')
            ax.grid(True)
            ax.axvline(self.omega_pass, color='r', linestyle='--', label='Passband Edge')
            ax.axvline(self.omega_stop, color='g', linestyle='--', label='Stopband Edge')
            ax.legend()

    def run_multiple(self, n_runs):
        fig, axes = plt.subplots(n_runs, 1, figsize=(10, 5 * n_runs))
        if n_runs == 1:
            axes = [axes]
        for i in range(n_runs):
            self.population = self.initialize_positions(initial_population=None)
            self.personal_best_positions = np.copy(self.population)
            self.personal_best_scores = np.full(self.pop_size, np.inf)
            self.global_best_position = None
            self.global_best_score = np.inf
            self.previous_global_best_score = np.inf
            best_coeffs = self.optimize()
            self.plot_frequency_response(best_coeffs, ax=axes[i])
        plt.tight_layout()
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
# #lowpass
# omega_pass = 0.44
# omega_stop = 0.55
# band+pass+stop
# omega_pass = 0.35
# omega_stop = 0.65
# #highpass
omega_pass = 0.4
omega_stop = 0.51

alpha = 0
tau = 10
W = lambda omega: 1  # Uniform weighting function as an example
delta_pass = 0.01  # Example pass-band ripple
delta_stop = 0.01  # Example stop-band ripple
nte = 5  # Number of territories
ar = 0.7  # Attraction response probability

# Initial Ideal Filter Coef provided as N1
Nlowpass = [0.0388639813481993, 0.00260088729777498, -0.0302244308396995, -0.0180939407775842, 0.0356920644947972, 
      0.0393786067312622, -0.0450132314893481, -0.0923298471847741, 0.0471816860342088, 0.311919757687732, 
      0.448804967338731, 0.311919757687732, 0.0471816860342088, -0.0923298471847741, -0.0450132314893481, 
      0.0393786067312622, 0.0356920644947972, -0.0180939407775842, -0.0302244308396995, 0.00260088729777498, 
      0.0388639813481993]

Nbandpass =  [0.198121838804337,-8.71790667529623e-05,-0.117713638701085,-0.000362304145142637,-0.0741235739610571,-0.000798171220967827,0.107100347504947,-0.00119313175650152,-0.306278112103464,-0.00139881848691635,0.269551140201775,-0.00139881848691635,-0.306278112103464,-0.00119313175650152,0.107100347504947,-0.000798171220967827,-0.0741235739610571,-0.000362304145142637,-0.117713638701085,-8.71790667529623e-05,0.198121838804337]
Nhighpass = [0.186317890363039,0.0464340730363551,-0.0852225118852898,0.0542746298935874,0.0151745959446690,-0.0126856439884851,-0.0798211193342827,0.119499473611657,0.0292566901356156,-0.289826805167671,0.421121186400957,-0.289826805167671,0.0292566901356156,0.119499473611657,-0.0798211193342827,-0.0126856439884851,0.0151745959446690,0.0542746298935874,-0.0852225118852898,0.0464340730363551,0.186317890363039]

Nbandstop = [-0.198121838804334,8.71790667314333e-05,0.117713638701116,0.000362304145089246,0.0741235739611247,0.000798171220874148,-0.107100347504838,0.00119313175637083,0.306278112103605,0.00139881848676322,0.730448859798378,0.00139881848676322,0.306278112103605,0.00119313175637083,-0.107100347504838,0.000798171220874148,0.0741235739611247,0.000362304145089246,0.117713638701116,8.71790667314333e-05,-0.198121838804334]
# Convert N1 to a numpy array and initialize WSO with it
N1 = np.array(Nhighpass)
wso = WaterStriderOptimization(pop_size, dim, max_iter, inertia_weight, cognitive_coeff, social_coeff, bounds, omega_pass, omega_stop, alpha, tau, W, delta_pass, delta_stop, nte, ar, filter_type='highpass', initial_population=N1)
wso.run_multiple(n_runs=10)
