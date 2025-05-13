import numpy as np
from scipy.stats import levy_stable
from visualizations import ComparisonVisualization

class CityData:
    def __init__(self):
        self.city_names = [
            'Tunis', 'Sfax', 'Sousse', 'Kairouan', 'Bizerte', 
            'Gabès', 'Ariana', 'Gafsa', 'Monastir', 'Mahdia',
            'Kebili', 'Zarzis', 'Nabeul', 'Beja', 'Siliana',
            'Tataouine', 'Douz', 'Medenine', 'Kasserine', 'Ben Guerdane',
            'El Kef', 'Sidi Bouzid', 'Zaghouan', 'Hammamet', 'Djerba'
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
            [10.8113, 35.7771],  # Monastir
            [11.0457, 35.5047],  # Mahdia
            [8.7114, 33.7044],   # Kebili
            [11.1122, 33.5032],  # Zarzis
            [10.7333, 36.4510],  # Nabeul
            [9.1833, 36.7333],   # Beja
            [9.3700, 36.0833],   # Siliana
            [10.4500, 32.9333],  # Tataouine
            [9.0143, 33.4572],   # Douz
            [10.5025, 33.3540],  # Medenine
            [8.8365, 35.1676],   # Kasserine
            [11.2167, 33.1378],  # Ben Guerdane
            [8.7089, 36.1742],   # El Kef
            [9.4978, 35.0382],   # Sidi Bouzid
            [10.1471, 36.4020],  # Zaghouan
            [10.6111, 36.4000],  # Hammamet
            [10.8651, 33.8076]   # Djerba
        ])
        self.dist_matrix = np.array([
            # Tunis, Sfax, Sousse, Kairouan, Bizerte, Gabès, Ariana, Gafsa, Monastir, Mahdia, Kebili, Zarzis, Nabeul, Beja, Siliana, Tataouine, Douz, Medenine, Kasserine, Ben Guerdane, El Kef, Sidi Bouzid, Zaghouan, Hammamet, Djerba
            [0, 270, 140, 160, 70, 400, 10, 350, 170, 200, 450, 520, 60, 100, 180, 600, 500, 550, 300, 650, 120, 250, 90, 80, 550],  # Tunis
            [270, 0, 130, 120, 300, 120, 280, 230, 140, 100, 320, 390, 250, 320, 300, 200, 180, 220, 150, 250, 350, 150, 200, 180, 300],  # Sfax
            [140, 130, 0, 60, 210, 270, 150, 300, 50, 90, 360, 430, 100, 200, 220, 350, 300, 350, 200, 400, 250, 180, 120, 80, 400],  # Sousse
            [160, 120, 60, 0, 230, 250, 170, 280, 70, 110, 340, 410, 120, 220, 240, 330, 280, 330, 180, 380, 270, 150, 140, 100, 380],  # Kairouan
            [70, 300, 210, 230, 0, 470, 80, 420, 240, 270, 500, 570, 90, 50, 150, 650, 550, 600, 400, 700, 100, 300, 120, 100, 600],  # Bizerte
            [400, 120, 270, 250, 470, 0, 410, 290, 290, 250, 200, 170, 350, 400, 400, 100, 80, 50, 200, 70, 450, 200, 300, 350, 50],  # Gabès
            [10, 280, 150, 170, 80, 410, 0, 360, 180, 210, 460, 530, 70, 110, 190, 610, 510, 560, 310, 660, 130, 260, 100, 90, 560],  # Ariana
            [350, 230, 300, 280, 420, 360, 360, 0, 380, 360, 0, 70, 420, 470, 470, 200, 150, 100, 250, 120, 450, 250, 350, 400, 100],  # Gafsa
            [170, 140, 50, 70, 240, 290, 180, 380, 0, 40, 380, 450, 150, 250, 270, 380, 330, 380, 230, 430, 280, 200, 160, 120, 430],  # Monastir
            [200, 100, 90, 110, 270, 250, 210, 360, 40, 0, 360, 430, 180, 280, 300, 360, 310, 360, 210, 410, 310, 230, 190, 150, 410],  # Mahdia
            [450, 320, 360, 340, 500, 200, 460, 0, 380, 360, 0, 70, 420, 470, 470, 200, 150, 100, 250, 120, 450, 250, 350, 400, 100],  # Kebili
            [520, 390, 430, 410, 570, 170, 530, 70, 450, 430, 70, 0, 490, 540, 540, 170, 120, 70, 320, 90, 520, 320, 420, 470, 70],  # Zarzis
            [60, 250, 100, 120, 90, 350, 70, 420, 150, 180, 420, 490, 0, 120, 140, 550, 450, 500, 250, 600, 110, 240, 80, 70, 500],  # Nabeul
            [100, 320, 200, 220, 50, 400, 110, 470, 250, 280, 470, 540, 120, 0, 100, 600, 500, 550, 300, 650, 100, 300, 120, 100, 550],  # Beja
            [180, 300, 220, 240, 150, 400, 190, 470, 270, 300, 470, 540, 140, 100, 0, 600, 500, 550, 300, 650, 180, 280, 140, 120, 550],  # Siliana
            [600, 200, 350, 330, 650, 100, 610, 200, 380, 360, 200, 170, 550, 600, 600, 0, 50, 100, 250, 70, 600, 300, 400, 450, 50],  # Tataouine
            [500, 180, 300, 280, 550, 80, 510, 150, 330, 310, 150, 120, 450, 500, 500, 50, 0, 50, 200, 70, 500, 250, 350, 400, 50],  # Douz
            [550, 220, 350, 330, 600, 50, 560, 100, 380, 360, 100, 70, 500, 550, 550, 100, 50, 0, 250, 70, 550, 300, 400, 450, 50],  # Medenine
            [300, 150, 200, 180, 400, 200, 310, 250, 230, 210, 250, 320, 250, 300, 300, 250, 200, 250, 0, 300, 300, 100, 200, 250, 250],  # Kasserine
            [650, 250, 400, 380, 700, 70, 660, 120, 430, 410, 120, 90, 600, 650, 650, 70, 70, 70, 300, 0, 650, 350, 450, 500, 70],  # Ben Guerdane
            [120, 350, 250, 270, 100, 450, 130, 450, 280, 310, 450, 520, 110, 100, 180, 600, 500, 550, 300, 650, 0, 250, 100, 90, 550],  # El Kef
            [250, 150, 180, 150, 300, 200, 260, 250, 200, 230, 250, 320, 240, 300, 280, 300, 250, 300, 100, 350, 250, 0, 150, 200, 300],  # Sidi Bouzid
            [90, 200, 120, 140, 120, 300, 100, 350, 160, 190, 350, 420, 80, 120, 140, 400, 350, 400, 200, 450, 100, 150, 0, 50, 400],  # Zaghouan
            [80, 180, 80, 100, 100, 350, 90, 400, 120, 150, 400, 470, 70, 100, 120, 450, 400, 450, 250, 500, 90, 200, 50, 0, 450],  # Hammamet
            [550, 300, 400, 380, 600, 50, 560, 100, 430, 410, 100, 70, 500, 550, 550, 50, 50, 50, 250, 70, 550, 300, 400, 450, 0],  # Djerba
        ])

class FlowerPollinationAlgorithm:
    def __init__(self, city_data, 
                population_size=30, 
                n_iterations=100, 
                switch_prob=0.8, 
                gamma=0.1, 
                lambda_=1.5):
        self.city_data = city_data
        self.population_size = population_size
        self.n_iterations = n_iterations
        self.initial_switch_prob = switch_prob  # Store the initial switch probability
        self.switch_prob = switch_prob  # Dynamic switch probability
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
        # Ensure the route starts and ends with Tunis (index 0)
        if permutation[0] != 0 or permutation[-1] != 0:
            permutation = [0] + [city for city in permutation if city != 0] + [0]
        distance = 0
        for i in range(len(permutation) - 1):
            distance += self.city_data.dist_matrix[permutation[i], permutation[i + 1]]
        return distance

    def optimize(self):
        for iteration in range(self.n_iterations):
            # Dynamically adjust switch_prob to balance exploration and exploitation
            self.switch_prob = self.initial_switch_prob * (1 - iteration / self.n_iterations)

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
            
            # Ensure history is updated even if no improvement occurs
            self.history.append(self.best_permutation.copy())

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
        self.n_iterations = 0  # Initialize iteration counter

    def total_distance(self, permutation):
        # Ensure the route starts and ends with Tunis (index 0)
        if permutation[0] != 0 or permutation[-1] != 0:
            permutation = [0] + [city for city in permutation if city != 0] + [0]
        distance = 0
        for i in range(len(permutation) - 1):
            distance += self.city_data.dist_matrix[permutation[i], permutation[i + 1]]
        return distance

    def swap_cities(self, solution):
        new_solution = solution.copy()
        i, j = np.random.choice(len(solution), 2, replace=False)
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        return new_solution

    def optimize(self, min_iterations=10):
        temp = self.initial_temp
        iteration = 0
        while (temp > self.min_temp or iteration < min_iterations) and iteration < 100:  # Ensure at least min_iterations
            new_solution = self.swap_cities(self.current_solution)
            new_distance = self.total_distance(new_solution)
            delta = new_distance - self.current_distance

            if delta < 0 or np.random.rand() < np.exp(-delta / temp):
                self.current_solution = new_solution
                self.current_distance = new_distance

                if new_distance < self.best_distance:
                    self.best_solution = new_solution
                    self.best_distance = new_distance

            # Ensure history is updated even if no improvement occurs
            self.history.append(self.best_solution.copy())
            temp *= self.cooling_rate
            iteration += 1  # Increment iteration count
            self.n_iterations += 1  # Track total iterations

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
        if permutation[0] != 0 or permutation[-1] != 0:
            permutation = [0] + [city for city in permutation if city != 0] + [0]
        distance = 0
        for i in range(len(permutation) - 1):
            distance += self.city_data.dist_matrix[permutation[i], permutation[i + 1]]
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

# Main execution
city_data = CityData()

# Solve using Flower Pollination Algorithm
fpa = FlowerPollinationAlgorithm(city_data)
fpa.optimize()

# Solve using Simulated Annealing
sa = SimulatedAnnealing(city_data)
sa.optimize()



comparison = ComparisonVisualization(city_data, fpa, sa)# Visualize both algorithms side by sidecomparison.animate()