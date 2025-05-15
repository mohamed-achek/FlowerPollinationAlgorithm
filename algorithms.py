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
            'Nabeul', 'Beja', 'Jendouba', 'Siliana', 'Tataouine'
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
            [10.4500, 32.9333]   # Tataouine
        ])
        self.dist_matrix = np.array([
            # Tunis, Sfax, Sousse, Kairouan, Bizerte, Gabès, Ariana, Gafsa, La Marsa, Menzel Bourguiba, Monastir, Mahdia, Tozeur, Kebili, Zarzis, Nabeul, Beja, Jendouba, Siliana, Tataouine
            [0, 270, 140, 160, 70, 400, 10, 350, 15, 80, 170, 200, 450, 500, 520, 60, 100, 150, 180, 600],  # Tunis
            [270, 0, 130, 120, 300, 120, 280, 230, 285, 310, 140, 100, 320, 370, 390, 250, 320, 350, 300, 200],  # Sfax
            [140, 130, 0, 60, 210, 270, 150, 300, 155, 220, 50, 90, 360, 410, 430, 100, 200, 250, 220, 350],  # Sousse
            [160, 120, 60, 0, 230, 250, 170, 280, 175, 240, 70, 110, 340, 390, 410, 120, 220, 270, 240, 330],  # Kairouan
            [70, 300, 210, 230, 0, 470, 80, 420, 85, 50, 240, 270, 500, 550, 570, 90, 50, 100, 150, 650],  # Bizerte
            [400, 120, 270, 250, 470, 0, 410, 150, 415, 440, 290, 250, 200, 150, 170, 350, 400, 450, 400, 100],  # Gabès
            [10, 280, 150, 170, 80, 410, 0, 360, 5, 90, 180, 210, 460, 510, 530, 70, 110, 160, 190, 610],  # Ariana
            [350, 230, 300, 280, 420, 150, 360, 0, 365, 390, 340, 300, 50, 100, 120, 320, 370, 420, 370, 150],  # Gafsa
            [15, 285, 155, 175, 85, 415, 5, 365, 0, 95, 185, 215, 465, 515, 535, 75, 115, 165, 195, 615],  # La Marsa
            [80, 310, 220, 240, 50, 440, 90, 390, 95, 0, 250, 280, 510, 560, 580, 100, 60, 110, 160, 660],  # Menzel Bourguiba
            [170, 140, 50, 70, 240, 290, 180, 340, 185, 250, 0, 40, 380, 430, 450, 150, 250, 300, 270, 380],  # Monastir
            [200, 100, 90, 110, 270, 250, 210, 300, 215, 280, 40, 0, 360, 410, 430, 180, 280, 330, 300, 360],  # Mahdia
            [450, 320, 360, 340, 500, 200, 460, 50, 465, 510, 380, 360, 0, 50, 70, 420, 470, 520, 470, 200],  # Tozeur
            [500, 370, 410, 390, 550, 150, 510, 100, 515, 560, 430, 410, 50, 0, 40, 470, 520, 570, 520, 150],  # Kebili
            [520, 390, 430, 410, 570, 170, 530, 120, 535, 580, 450, 430, 70, 40, 0, 490, 540, 590, 540, 170],  # Zarzis
            [60, 250, 100, 120, 90, 350, 70, 320, 75, 100, 150, 180, 420, 470, 490, 0, 120, 170, 140, 550],  # Nabeul
            [100, 320, 200, 220, 50, 400, 110, 370, 115, 60, 250, 280, 470, 520, 540, 120, 0, 80, 100, 600],  # Beja
            [150, 350, 250, 270, 100, 450, 160, 420, 165, 110, 300, 330, 520, 570, 590, 170, 80, 0, 120, 650],  # Jendouba
            [180, 300, 220, 240, 150, 400, 190, 370, 195, 160, 270, 300, 470, 520, 540, 140, 100, 120, 0, 600],  # Siliana
            [600, 200, 350, 330, 650, 100, 610, 150, 615, 660, 380, 360, 200, 150, 170, 550, 600, 650, 600, 0],  # Tataouine
        ])

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
            distance += self.city_data.dist_matrix[permutation[i], permutation[i + 1]]
        distance += self.city_data.dist_matrix[permutation[-1], permutation[0]]  # Return to start
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
                min_temp=10):  
        
        self.city_data = city_data
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.current_solution = np.random.permutation(len(city_data.city_names))
        self.current_distance = self.total_distance(self.current_solution)
        self.best_solution = self.current_solution.copy()
        self.best_distance = self.current_distance
        self.history = []

    def total_distance(self, permutation):
        distance = 0
        for i in range(len(permutation) - 1):
            distance += self.city_data.dist_matrix[permutation[i], permutation[i + 1]]
        distance += self.city_data.dist_matrix[permutation[-1], permutation[0]]  # Return to start
        return distance

    def swap_cities(self, solution):
        new_solution = solution.copy()
        i, j = np.random.choice(len(solution), 2, replace=False)
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        return new_solution

    def optimize(self):
        temp = self.initial_temp
        iteration = 0
        while temp > self.min_temp and iteration < 100:  # Limit iterations to 100
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
            iteration += 1  # Increment iteration count

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
        self.fitness = [self.total_distance(ind) for ind in self.population]
        self.best_solution = self.population[np.argmin(self.fitness)].copy()
        self.best_distance = min(self.fitness)
        self.history = []

    def total_distance(self, permutation):
        distance = 0
        for i in range(len(permutation) - 1):
            distance += self.city_data.dist_matrix[permutation[i], permutation[i + 1]]
        distance += self.city_data.dist_matrix[permutation[-1], permutation[0]]
        return distance

    def select_parents(self):
        idx = np.argsort(self.fitness)
        return [self.population[idx[0]], self.population[idx[1]]]

    def crossover(self, parent1, parent2):
        size = len(parent1)
        a, b = sorted(np.random.choice(range(size), 2, replace=False))
        child = [-1] * size
        child[a:b] = parent1[a:b]
        fill = [item for item in parent2 if item not in child]
        idx = 0
        for i in range(size):
            if child[i] == -1:
                child[i] = fill[idx]
                idx += 1
        return np.array(child)

    def mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            i, j = np.random.choice(len(individual), 2, replace=False)
            individual[i], individual[j] = individual[j], individual[i]
        return individual

    def optimize(self):
        for _ in range(self.n_generations):
            new_population = []
            for _ in range(self.population_size):
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            self.population = new_population
            self.fitness = [self.total_distance(ind) for ind in self.population]
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_distance:
                self.best_solution = self.population[best_idx].copy()
                self.best_distance = self.fitness[best_idx]
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