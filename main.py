import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy_stable
from matplotlib.animation import FuncAnimation
from algorithms import CityData  # Import CityData from algorithms.py
from algorithms import FlowerPollinationAlgorithm, SimulatedAnnealing  # Import algorithms from algorithms.py
from algorithms import GeneticAlgorithm  # Import GeneticAlgorithm from algorithms.py

class ComparisonVisualization:
    def __init__(self, city_data, fpa, sa, ga):
        self.city_data = city_data
        self.fpa = fpa
        self.sa = sa
        self.ga = ga
        self.fig, self.axes = plt.subplots(1, 3, figsize=(24, 8))  # Add a third subplot for GA
        self.fpa_frames = len(fpa.history)
        self.sa_frames = len(sa.history)
        self.ga_frames = len(ga.history)
        self.max_frames = max(self.fpa_frames, self.sa_frames, self.ga_frames)

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

        # Update GA plot
        if frame < self.ga_frames:
            perm = self.ga.history[frame]
            route = np.append(perm, perm[0])
            ax = self.axes[2]
            ax.scatter(self.city_data.cities[:, 0], self.city_data.cities[:, 1], c='red', s=100, zorder=5, label='Cities')
            for i, (x, y) in enumerate(self.city_data.cities):
                ax.text(x, y, f' {self.city_data.city_names[i]} ({np.where(perm == i)[0][0] + 1})', fontsize=8, ha='right', zorder=10)
            ax.plot(self.city_data.cities[route, 0], self.city_data.cities[route, 1], 'b-', lw=2, alpha=0.7, label='Route')
            ax.set_title(f'GA Generation: {frame + 1}\nDistance: {self.ga.total_distance(perm):.0f} km', fontsize=12)
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
fpa = FlowerPollinationAlgorithm(city_data)  # Imported from algorithms.py
fpa.optimize()  # Runs the FPA algorithm for 100 iterations (default value)

# Solve using Simulated Annealing
sa = SimulatedAnnealing(city_data)  # Imported from algorithms.py
sa.optimize()  # Runs the SA algorithm for 100 iterations (default value)

# Solve using Genetic Algorithm
ga = GeneticAlgorithm(city_data)
ga.optimize()

# Visualize all three algorithms side by side
comparison = ComparisonVisualization(city_data, fpa, sa, ga)
comparison.animate()