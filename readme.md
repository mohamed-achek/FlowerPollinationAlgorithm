# Traveling Salesman Problem - Optimization Algorithms

This project demonstrates the application of optimization algorithms to solve the **Traveling Salesman Problem (TSP)** for Tunisian cities. The algorithms used are:
- **Flower Pollination Algorithm (FPA)**
- **Simulated Annealing (SA)**

The project provides a Streamlit-based interactive dashboard to visualize the optimization process and compare the results of the two algorithms.

---

## Features

- **Dynamic Visualization**: Real-time updates of the optimization process using Matplotlib plots.
- **Comparison Tab**: Static maps (using Folium) to compare the final results of both algorithms.
- **Performance Metrics**: Displays the best distance, execution time, and stability (standard deviation) for each algorithm.

---


## Algorithms

### 1. Flower Pollination Algorithm (FPA)
- Inspired by the pollination process of flowering plants.
- Combines global and local search strategies to find the optimal route.

### 2. Simulated Annealing (SA)
- A probabilistic optimization algorithm inspired by the annealing process in metallurgy.
- Gradually reduces the probability of accepting worse solutions to converge to an optimal solution.

---

## Example Output

### Run Algorithms Tab
- Real-time Matplotlib plot showing the optimization process.

### Compare Results Tab
- Static Folium maps comparing the final routes of FPA and SA.

---

## Dependencies

- `streamlit`
- `matplotlib`
- `numpy`
- `folium`
- `streamlit-folium`


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

- **Streamlit**: For building interactive dashboards.
- **Folium**: For map visualizations.
- **Matplotlib**: For dynamic plotting.
