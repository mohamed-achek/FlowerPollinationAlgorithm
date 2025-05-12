# Traveling Salesman Problem - Optimization Algorithms

This project demonstrates the application of optimization algorithms to solve the **Traveling Salesman Problem (TSP)** for Tunisian cities. The algorithms used are:
- **Flower Pollination Algorithm (FPA)**
- **Simulated Annealing (SA)**
- **Genetic Algorithm (GA)**

The project provides a Streamlit-based interactive dashboard to visualize the optimization process and compare the results of the algorithms.

---

## Features

- **Dynamic Visualization**: Real-time updates of the optimization process using Matplotlib plots.
- **Comparison Tab**: Static maps (using Folium) to compare the final results of all algorithms.
- **Performance Metrics**: Displays the best distance, execution time, and average performance for each algorithm.

---

## Algorithms

### 1. Flower Pollination Algorithm (FPA)
- Inspired by the pollination process of flowering plants.
- Combines global and local search strategies to find the optimal route.

### 2. Simulated Annealing (SA)
- A probabilistic optimization algorithm inspired by the annealing process in metallurgy.
- Gradually reduces the probability of accepting worse solutions to converge to an optimal solution.

### 3. Genetic Algorithm (GA)
- A population-based optimization algorithm inspired by natural selection.
- Uses crossover and mutation to evolve solutions over generations.

---

## Example Output

### Run Algorithms Tab
- Real-time Matplotlib plot showing the optimization process for the selected algorithm.
- Animated Folium map displaying the final route.

### Compare Results Tab
- Static Folium maps comparing the final routes of FPA, SA, and GA.
- Performance metrics table showing the best distance, average distance, and execution time for each algorithm.
- Convergence plots illustrating the best distances over multiple runs.

---

## Dependencies

The project requires the following Python libraries:
- `streamlit`
- `matplotlib`
- `numpy`
- `scipy`
- `folium`
- `streamlit-folium`

Install the dependencies using:
```bash
pip install -r requirements.txt
```

---

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd FPA
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Open the provided URL in your browser to interact with the dashboard.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

- **Streamlit**: For building interactive dashboards.
- **Folium**: For map visualizations.
- **Matplotlib**: For dynamic plotting.
- **Scipy**: For implementing the Levy flight distribution in FPA.
