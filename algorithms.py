import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy_stable
from matplotlib.animation import FuncAnimation

class CityData:
    def __init__(self):
        self.city_names = [
            'Tunis', 'Sfax', 'Sousse', 'Kairouan', 'Bizerte', 
            'Gabès', 'Ariana', 'Gafsa', 'La Marsa', 'Menzel Bourguiba',
            'Monastir', 'Mahdia', 'Tozeur', 'Kebili', 'Zarzis',
            'Nabeul', 'Beja', 'Jendouba', 'Siliana', 'Tataouine',
            'Ben Arous', 'Manouba', 'Moknine', 'Metlaoui', 'Douz'
        ]
        self.cities = np.array([
            [10.1815, 36.8065],  # Tunis
            [10.7603, 34.7406],  # Sfax
            [10.6346, 35.8245],  # Sousse
            [10.1017, 35.6781],  # Kairouan
            [9.8739, 37.2744],   # Bizerte
            [10.0982, 33.8815],  # Gabès
            [10.1934, 36.8601],  # Ariana
            [8.7842, 34.4250],   # Gafsa
            [10.3300, 36.8782],  # La Marsa
            [9.7844, 37.1537],   # Menzel Bourguiba
            [10.8113, 35.7771],  # Monastir
            [11.0457, 35.5047],  # Mahdia
            [8.1335, 33.9197],   # Tozeur
            [8.7114, 33.7044],   # Kebili
            [11.1122, 33.5032],  # Zarzis
            [10.7333, 36.4510],  # Nabeul
            [9.1833, 36.7333],   # Beja
            [8.7833, 36.5000],   # Jendouba
            [9.3700, 36.0833],   # Siliana
            [10.4500, 32.9333],  # Tataouine
            [10.2320, 36.7538],  # Ben Arous (placeholder)
            [10.1012, 36.8081],  # Manouba (placeholder)
            [10.9900, 35.2500],  # Moknine (placeholder)
            [8.4000, 34.3200],   # Metlaoui (placeholder)
            [9.0167, 33.4500]    # Douz (placeholder)
        ])
        # Compute symmetric distance matrix using Euclidean distance and scale to km
        n = len(self.cities)
        self.distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = self.cities[i][0] - self.cities[j][0]
                    dy = self.cities[i][1] - self.cities[j][1]
                    self.distance_matrix[i, j] = np.sqrt(dx**2 + dy**2) * 111  # Approximate km

class FlowerPollinationAlgorithm:
    def __init__(self, city_data, 
                population_size=20, 
                n_iterations=100, 
                switch_prob=0.8, 
                gamma=0.1, 
                lambda_=1.5):
        self.city_data = city_data
        self.population_size = population_size
        self.n_iterations = n_iterations
        self.switch_prob = switch_prob
        self.gamma = gamma
        self.lambda_ = lambda_
        self.population = np.random.rand(population_size, len(city_data.city_names))
        self.best_solution = self.population[0].copy()
        self.best_permutation = self.vector_to_permutation(self.best_solution)
        self.best_distance = self.total_distance(self.best_permutation)
        self.history = []

    def vector_to_permutation(self, vector):
        return np.argsort(vector)

    def total_distance(self, permutation):
        distance = 0
        for i in range(len(permutation) - 1):
            distance += self.city_data.distance_matrix[permutation[i], permutation[i + 1]]
        distance += self.city_data.distance_matrix[permutation[-1], permutation[0]]  # Return to start
        return distance

    def optimize(self):
        for _ in range(self.n_iterations):
            for i in range(self.population_size):
                if np.random.rand() < self.switch_prob:
                    step = levy_stable.rvs(alpha=self.lambda_, beta=0, loc=0, scale=self.gamma, size=len(self.city_data.city_names))
                    new_solution = self.population[i] + step * (self.best_solution - self.population[i])
                else:
                    epsilon = np.random.rand()
                    j, k = np.random.choice(self.population_size, 2, replace=False)
                    new_solution = self.population[i] + epsilon * (self.population[j] - self.population[k])
                
                new_solution = np.clip(new_solution, 0, 1)
                new_permutation = self.vector_to_permutation(new_solution)
                new_distance = self.total_distance(new_permutation)

                if new_distance < self.best_distance:
                    self.best_solution = new_solution.copy()
                    self.best_permutation = new_permutation
                    self.best_distance = new_distance
            
            self.history.append(self.best_permutation.copy())

    def animate(self):
        fig, ax = plt.subplots(figsize=(8, 6))

        def update(frame):
            ax.clear()
            perm = self.history[frame]
            route = np.append(perm, perm[0])
            ax.scatter(self.city_data.cities[:, 0], self.city_data.cities[:, 1], c='red', s=100, zorder=5, label='Cities')
            for i, (x, y) in enumerate(self.city_data.cities):
                ax.text(x, y, f' {self.city_data.city_names[i]} ({np.where(perm == i)[0][0] + 1})', fontsize=8, ha='right', zorder=10)
            ax.plot(self.city_data.cities[route, 0], self.city_data.cities[route, 1], 'b-', lw=2, alpha=0.7, label='Route')
            ax.set_title(f'Iteration: {frame + 1}\nDistance: {self.total_distance(perm):.0f} km', fontsize=12)  # Fixed format specifier
            ax.set_xlabel('Longitude (°E)', fontsize=10)
            ax.set_ylabel('Latitude (°N)', fontsize=10)
            ax.set_xlim(np.min(self.city_data.cities[:, 0]) - 0.5, np.max(self.city_data.cities[:, 0]) + 0.5)
            ax.set_ylim(np.min(self.city_data.cities[:, 1]) - 0.5, np.max(self.city_data.cities[:, 1]) + 0.5)
            ax.set_aspect('equal')
            ax.legend(loc='upper right', fontsize=9)


class SimulatedAnnealing:
    def __init__(self, city_data, 
                initial_temp=1000, 
                cooling_rate=0.9, 
                min_temp=10, 
                max_iterations=100):  
        
        self.city_data = city_data
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iterations = max_iterations
        self.current_solution = np.random.permutation(len(city_data.city_names))
        self.current_distance = self.total_distance(self.current_solution)
        self.best_solution = self.current_solution.copy()
        self.best_distance = self.current_distance
        self.history = []

    def total_distance(self, permutation):
        distance = 0
        for i in range(len(permutation) - 1):
            distance += self.city_data.distance_matrix[permutation[i], permutation[i + 1]]
        distance += self.city_data.distance_matrix[permutation[-1], permutation[0]]  # Return to start
        return distance

    def swap_cities(self, solution):
        new_solution = solution.copy()
        i, j = np.random.choice(len(solution), 2, replace=False)
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        return new_solution

    def optimize(self, max_iterations=None):
        temp = self.initial_temp
        iteration = 0
        max_iter = max_iterations if max_iterations is not None else self.max_iterations
        while temp > self.min_temp and iteration < max_iter:
            new_solution = self.swap_cities(self.current_solution)
            new_distance = self.total_distance(new_solution)
            delta = new_distance - self.current_distance

            if delta < 0 or np.random.rand() < np.exp(-delta / temp):
                self.current_solution = new_solution
                self.current_distance = new_distance

                if new_distance < self.best_distance:
                    self.best_solution = new_solution
                    self.best_distance = new_distance

            self.history.append(self.best_solution.copy())
            temp *= self.cooling_rate
            iteration += 1

    def visualize(self):
        best_route = np.append(self.best_solution, self.best_solution[0])
        plt.figure(figsize=(8, 6))
        plt.scatter(self.city_data.cities[:, 0], self.city_data.cities[:, 1], c='red', s=100, zorder=5, label='Cities')
        for i, (x, y) in enumerate(self.city_data.cities):
            plt.text(x, y, f' {self.city_data.city_names[i]} ({np.where(self.best_solution == i)[0][0] + 1})', fontsize=8, ha='right', zorder=10)
        plt.plot(self.city_data.cities[best_route, 0], self.city_data.cities[best_route, 1], 'b-', lw=2, alpha=0.7, label='Route')
        plt.title(f'Simulated Annealing Solution\nDistance: {self.best_distance:.0f} km', fontsize=12)
        plt.xlabel('Longitude (°E)', fontsize=10)
        plt.ylabel('Latitude (°N)', fontsize=10)
        plt.xlim(np.min(self.city_data.cities[:, 0]) - 0.5, np.max(self.city_data.cities[:, 0]) + 0.5)
        plt.ylim(np.min(self.city_data.cities[:, 1]) - 0.5, np.max(self.city_data.cities[:, 1]) + 0.5)
        plt.gca().set_aspect('equal')
        plt.legend(loc='upper right', fontsize=9)
        plt.tight_layout()
        plt.show()

class GeneticAlgorithm:
    def __init__(self, city_data, population_size=50, n_generations=100, mutation_rate=0.1):
        self.city_data = city_data
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.population = [np.random.permutation(len(city_data.city_names)) for _ in range(population_size)]
        self.best_solution = self.population[0]
        self.best_distance = self.total_distance(self.best_solution)
        self.history = []

    def total_distance(self, permutation):
        # Ensure the route starts and ends with Tunis (index 0)
        permutation = [0] + [city for city in permutation if city != 0] + [0]
        distance = 0
        for i in range(len(permutation) - 1):
            distance += self.city_data.distance_matrix[permutation[i], permutation[i + 1]]
        return distance

    def crossover(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(np.random.choice(range(size), 2, replace=False))
        child = [-1] * size
        child[start:end] = parent1[start:end]
        pointer = 0
        for gene in parent2:
            if gene not in child:
                while child[pointer] != -1:
                    pointer += 1
                child[pointer] = gene
        return np.array(child)

    def mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            i, j = np.random.choice(len(individual), 2, replace=False)
            individual[i], individual[j] = individual[j], individual[i]

    def optimize(self):
        for generation in range(self.n_generations):
            fitness = np.array([1 / self.total_distance(ind) for ind in self.population])
            probabilities = fitness / fitness.sum()
            new_population = []
            population_array = np.array(self.population)  # Convert population to a NumPy array

            for _ in range(self.population_size):
                parents_indices = np.random.choice(len(population_array), size=2, p=probabilities, replace=False)
                parent1, parent2 = population_array[parents_indices[0]], population_array[parents_indices[1]]
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)

            self.population = new_population

            # Update the best solution
            for individual in self.population:
                distance = self.total_distance(individual)
                if distance < self.best_distance:
                    self.best_solution = individual
                    self.best_distance = distance

            self.history.append(self.best_solution.copy())

class ComparisonVisualization:
    def __init__(self, city_data, fpa, sa):
        self.city_data = city_data
        self.fpa = fpa
        self.sa = sa
        self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 8))
        self.fpa_frames = len(fpa.history)
        self.sa_frames = len(sa.history)
        self.max_frames = max(self.fpa_frames, self.sa_frames)

    def update(self, frame):
        for ax in self.axes:
            ax.clear()

        # Update FPA plot
        if frame < self.fpa_frames:
            perm = self.fpa.history[frame]
            route = np.append(perm, perm[0])
            ax = self.axes[0]
            ax.scatter(self.city_data.cities[:, 0], self.city_data.cities[:, 1], c='red', s=100, zorder=5, label='Cities')
            for i, (x, y) in enumerate(self.city_data.cities):
                ax.text(x, y, f' {self.city_data.city_names[i]} ({np.where(perm == i)[0][0] + 1})', fontsize=8, ha='right', zorder=10)
            ax.plot(self.city_data.cities[route, 0], self.city_data.cities[route, 1], 'b-', lw=2, alpha=0.7, label='Route')
            ax.set_title(f'FPA Iteration: {frame + 1}\nDistance: {self.fpa.total_distance(perm):.0f} km', fontsize=12)  # Fixed format specifier
            ax.set_xlabel('Longitude (°E)', fontsize=10)
            ax.set_ylabel('Latitude (°N)', fontsize=10)
            ax.set_xlim(np.min(self.city_data.cities[:, 0]) - 0.5, np.max(self.city_data.cities[:, 0]) + 0.5)
            ax.set_ylim(np.min(self.city_data.cities[:, 1]) - 0.5, np.max(self.city_data.cities[:, 1]) + 0.5)
            ax.set_aspect('equal')
            ax.legend(loc='upper right', fontsize=9)

        # Update SA plot
        if frame < self.sa_frames:
            perm = self.sa.history[frame]
            route = np.append(perm, perm[0])
            ax = self.axes[1]
            ax.scatter(self.city_data.cities[:, 0], self.city_data.cities[:, 1], c='red', s=100, zorder=5, label='Cities')
            for i, (x, y) in enumerate(self.city_data.cities):
                ax.text(x, y, f' {self.city_data.city_names[i]} ({np.where(perm == i)[0][0] + 1})', fontsize=8, ha='right', zorder=10)
            ax.plot(self.city_data.cities[route, 0], self.city_data.cities[route, 1], 'b-', lw=2, alpha=0.7, label='Route')
            ax.set_title(f'SA Iteration: {frame + 1}\nDistance: {self.sa.total_distance(perm):.0f} km', fontsize=12)  # Fixed format specifier
            ax.set_xlabel('Longitude (°E)', fontsize=10)
            ax.set_ylabel('Latitude (°N)', fontsize=10)
            ax.set_xlim(np.min(self.city_data.cities[:, 0]) - 0.5, np.max(self.city_data.cities[:, 0]) + 0.5)
            ax.set_ylim(np.min(self.city_data.cities[:, 1]) - 0.5, np.max(self.city_data.cities[:, 1]) + 0.5)
            ax.set_aspect('equal')
            ax.legend(loc='upper right', fontsize=9)

    def animate(self):
        ani = FuncAnimation(self.fig, self.update, frames=self.max_frames, repeat=False)
        # Removed plt.show() to avoid warning in Streamlit
        ani.save('comparison_animation.gif', writer='pillow', fps=5)

# Main execution
city_data = CityData()

# Solve using Flower Pollination Algorithm
fpa = FlowerPollinationAlgorithm(city_data)
fpa.optimize()

# Solve using Simulated Annealing
sa = SimulatedAnnealing(city_data)
sa.optimize()

# Visualize both algorithms side by side
comparison = ComparisonVisualization(city_data, fpa, sa)
comparison.animate()