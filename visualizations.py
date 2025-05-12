import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

def compute_cumulative_distances(city_data, route):
    """
    Compute cumulative distances for a given route.
    """
    cumulative_distances = [0]
    for i in range(1, len(route)):
        prev_city = city_data.cities[route[i - 1]]
        curr_city = city_data.cities[route[i]]
        distance = np.linalg.norm(curr_city - prev_city)
        cumulative_distances.append(cumulative_distances[-1] + distance)
    return cumulative_distances

def plot_convergence(history, algorithm_name):
    """
    Plot the convergence of the algorithm over iterations or generations.
    """
    distances = [1 / fitness if fitness > 0 else float('inf') for fitness in np.array(history).flatten()]
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(distances) + 1), distances, marker='o', linestyle='-', color='blue', label='Best Distance')
    plt.title(f"Convergence Plot - {algorithm_name}", fontsize=14)
    plt.xlabel("Iteration/Generation", fontsize=12)
    plt.ylabel("Best Distance (km)", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

def plot_convergence_over_runs(best_distances, algorithm_name):
    """
    Plot the convergence of the best distance over multiple runs.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(best_distances) + 1), best_distances, marker='o', linestyle='-', color='blue', label='Best Distance')
    plt.title(f"Convergence of Best Distance Over Runs - {algorithm_name}", fontsize=14)
    plt.xlabel("Run", fontsize=12)
    plt.ylabel("Best Distance (km)", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

def compare_execution_times(fpa_time, sa_time, ga_time):
    """
    Compare the execution times of the algorithms using a bar chart.
    """
    algorithms = ["FPA", "SA", "GA"]
    times = [fpa_time, sa_time, ga_time]

    plt.figure(figsize=(8, 6))
    plt.bar(algorithms, times, color=['blue', 'red', 'green'])
    plt.title("Comparison of Execution Times", fontsize=14)
    plt.xlabel("Algorithm", fontsize=12)
    plt.ylabel("Execution Time (seconds)", fontsize=12)
    plt.tight_layout()
    return plt.gcf()

class ComparisonVisualization:
    def __init__(self, city_data, fpa, sa):
        self.city_data = city_data
        self.fpa = fpa
        self.sa = sa
        self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 8))
        self.max_frames = max(len(fpa.history), len(sa.history))

    def update(self, frame):
        self._update_plot(self.axes[0], self.fpa, frame, "FPA")
        self._update_plot(self.axes[1], self.sa, frame, "SA")

    def _update_plot(self, ax, algorithm, frame, label):
        if frame < len(algorithm.history):
            perm = algorithm.history[frame]
            route = np.append(perm, perm[0])
            ax.clear()
            ax.scatter(self.city_data.cities[:, 0], self.city_data.cities[:, 1], c='red', s=100, zorder=5, label='Cities')
            ax.plot(self.city_data.cities[route, 0], self.city_data.cities[route, 1], 'b-', lw=2, alpha=0.7, label='Route')
            ax.set_title(f"{label} Iteration: {frame + 1}\nDistance: {algorithm.total_distance(perm):.0f} km", fontsize=12)
            ax.set_xlabel('Longitude (°E)', fontsize=10)
            ax.set_ylabel('Latitude (°N)', fontsize=10)
            ax.set_aspect('equal')
            ax.legend(loc='upper right', fontsize=9)

    def animate(self):
        if self.max_frames == 0:
            print("No frames to animate. Ensure the algorithms have been optimized.")
            return
        ani = FuncAnimation(self.fig, self.update, frames=self.max_frames, repeat=False)
        ani.save('comparison_animation.gif', writer='pillow', fps=5)