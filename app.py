import streamlit as st
import matplotlib.pyplot as plt
import time
import numpy as np
import folium
from streamlit_folium import st_folium  # Replace folium_static with st_folium
from main import CityData, FlowerPollinationAlgorithm, SimulatedAnnealing

# Streamlit app title
st.title("Traveling Salesman Problem - Tunisian Cities")

# Load city data
city_data = CityData()

# Initialize session state for storing results
if "fpa_result" not in st.session_state:
    st.session_state.fpa_result = None
if "sa_result" not in st.session_state:
    st.session_state.sa_result = None
if "fpa_times" not in st.session_state:
    st.session_state.fpa_times = []
if "sa_times" not in st.session_state:
    st.session_state.sa_times = []
if "fpa_distances" not in st.session_state:
    st.session_state.fpa_distances = []
if "sa_distances" not in st.session_state:
    st.session_state.sa_distances = []

# Initialize session state for storing last plots
if "fpa_last_plot" not in st.session_state:
    st.session_state.fpa_last_plot = None
if "sa_last_plot" not in st.session_state:
    st.session_state.sa_last_plot = None

# Helper function to plot routes using Matplotlib
def plot_route(city_data, route, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    route = list(route) + [route[0]]  # Complete the loop
    ax.scatter(city_data.cities[:, 0], city_data.cities[:, 1], c='red', s=100, zorder=5, label='Cities')
    for i, (x, y) in enumerate(city_data.cities):
        ax.text(x, y, f' {city_data.city_names[i]}', fontsize=8, ha='right', zorder=10)
    ax.plot(city_data.cities[route, 0], city_data.cities[route, 1], 'b-', lw=2, alpha=0.7, label='Route')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# Helper function to create a folium map
def create_map(city_data, route):
    m = folium.Map(location=[city_data.cities[:, 1].mean(), city_data.cities[:, 0].mean()], zoom_start=7)
    for i, (lon, lat) in enumerate(city_data.cities):
        folium.Marker([lat, lon], popup=city_data.city_names[i]).add_to(m)
    for i in range(len(route) - 1):
        folium.PolyLine(
            [[city_data.cities[route[i], 1], city_data.cities[route[i], 0]],
             [city_data.cities[route[i + 1], 1], city_data.cities[route[i + 1], 0]]],
            color="blue", weight=2.5, opacity=1
        ).add_to(m)
    folium.PolyLine(
        [[city_data.cities[route[-1], 1], city_data.cities[route[-1], 0]],
         [city_data.cities[route[0], 1], city_data.cities[route[0], 0]]],
        color="blue", weight=2.5, opacity=1
    ).add_to(m)
    return m

# Create tabs for running algorithms and comparing results
tab1, tab2 = st.tabs(["Run Algorithms", "Compare Results"])

with tab1:
    # Sidebar for algorithm selection
    st.sidebar.title("Select Algorithm")
    algorithm = st.sidebar.radio("Choose an algorithm to run:", ("Flower Pollination Algorithm", "Simulated Annealing"))

    # Display the last plot if available
    if algorithm == "Flower Pollination Algorithm" and st.session_state.fpa_last_plot:
        st.write("Last Run - Flower Pollination Algorithm")
        st.pyplot(st.session_state.fpa_last_plot)
    elif algorithm == "Simulated Annealing" and st.session_state.sa_last_plot:
        st.write("Last Run - Simulated Annealing")
        st.pyplot(st.session_state.sa_last_plot)

    # Button to run the selected algorithm
    if st.sidebar.button("Run Algorithm"):
        if algorithm == "Flower Pollination Algorithm":
            st.write("Running Flower Pollination Algorithm...")
            fpa = FlowerPollinationAlgorithm(city_data)
            progress_placeholder = st.empty()
            plot_placeholder = st.empty()  # Placeholder for the plot

            start_time = time.time()
            for iteration in range(fpa.n_iterations):
                fpa.optimize()  # Perform one iteration of optimization
                progress_placeholder.write(f"Iteration: {iteration + 1}/{fpa.n_iterations}, Best Distance: {fpa.best_distance:.2f} km")

                # Update the Matplotlib plot dynamically
                fig, ax = plt.subplots(figsize=(8, 6))
                current_route = list(fpa.best_permutation) + [fpa.best_permutation[0]]
                ax.scatter(city_data.cities[:, 0], city_data.cities[:, 1], c='red', s=100, zorder=5, label='Cities')
                for i, (x, y) in enumerate(city_data.cities):
                    ax.text(x, y, f' {city_data.city_names[i]}', fontsize=8, ha='right', zorder=10)
                ax.plot(city_data.cities[current_route, 0], city_data.cities[current_route, 1], 'b-', lw=2, alpha=0.7, label='Route')
                ax.set_title(f"FPA Iteration {iteration + 1}", fontsize=12)
                ax.set_xlabel("Longitude (°E)")
                ax.set_ylabel("Latitude (°N)")
                ax.legend()
                plot_placeholder.pyplot(fig)
                plt.close(fig)

            execution_time = time.time() - start_time
            # Save FPA result to session state
            st.session_state.fpa_result = {
                "best_distance": fpa.best_distance,
                "best_route": list(fpa.best_permutation),
                "execution_time": execution_time
            }
            st.session_state.fpa_times.append(execution_time)
            st.session_state.fpa_distances.append(fpa.best_distance)

            # Save the last plot to session state
            fig, ax = plt.subplots(figsize=(8, 6))
            current_route = list(fpa.best_permutation) + [fpa.best_permutation[0]]
            ax.scatter(city_data.cities[:, 0], city_data.cities[:, 1], c='red', s=100, zorder=5, label='Cities')
            for i, (x, y) in enumerate(city_data.cities):
                ax.text(x, y, f' {city_data.city_names[i]}', fontsize=8, ha='right', zorder=10)
            ax.plot(city_data.cities[current_route, 0], city_data.cities[current_route, 1], 'b-', lw=2, alpha=0.7, label='Route')
            ax.set_title("Final FPA Solution", fontsize=12)
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
            ax.legend()
            st.session_state.fpa_last_plot = fig  # Store the plot
            st.pyplot(fig)
            plt.close(fig)

        elif algorithm == "Simulated Annealing":
            st.write("Running Simulated Annealing...")
            sa = SimulatedAnnealing(city_data)
            progress_placeholder = st.empty()
            plot_placeholder = st.empty()  # Placeholder for the plot

            start_time = time.time()
            for iteration in range(100):  # Simulated Annealing is limited to 100 iterations
                sa.optimize()  # Perform one iteration of optimization
                progress_placeholder.write(f"Iteration: {iteration + 1}/100, Best Distance: {sa.best_distance:.2f} km")

                # Update the Matplotlib plot dynamically
                fig, ax = plt.subplots(figsize=(8, 6))
                current_route = list(sa.best_solution) + [sa.best_solution[0]]
                ax.scatter(city_data.cities[:, 0], city_data.cities[:, 1], c='red', s=100, zorder=5, label='Cities')
                for i, (x, y) in enumerate(city_data.cities):
                    ax.text(x, y, f' {city_data.city_names[i]}', fontsize=8, ha='right', zorder=10)
                ax.plot(city_data.cities[current_route, 0], city_data.cities[current_route, 1], 'b-', lw=2, alpha=0.7, label='Route')
                ax.set_title(f"SA Iteration {iteration + 1}", fontsize=12)
                ax.set_xlabel("Longitude (°E)")
                ax.set_ylabel("Latitude (°N)")
                ax.legend()
                plot_placeholder.pyplot(fig)
                plt.close(fig)

            execution_time = time.time() - start_time
            # Save SA result to session state
            st.session_state.sa_result = {
                "best_distance": sa.best_distance,
                "best_route": list(sa.best_solution),
                "execution_time": execution_time
            }
            st.session_state.sa_times.append(execution_time)
            st.session_state.sa_distances.append(sa.best_distance)

            # Save the last plot to session state
            fig, ax = plt.subplots(figsize=(8, 6))
            current_route = list(sa.best_solution) + [sa.best_solution[0]]
            ax.scatter(city_data.cities[:, 0], city_data.cities[:, 1], c='red', s=100, zorder=5, label='Cities')
            for i, (x, y) in enumerate(city_data.cities):
                ax.text(x, y, f' {city_data.city_names[i]}', fontsize=8, ha='right', zorder=10)
            ax.plot(city_data.cities[current_route, 0], city_data.cities[current_route, 1], 'b-', lw=2, alpha=0.7, label='Route')
            ax.set_title("Final SA Solution", fontsize=12)
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
            ax.legend()
            st.session_state.sa_last_plot = fig  # Store the plot
            st.pyplot(fig)
            plt.close(fig)

with tab2:
    st.write("Comparing Results of Flower Pollination Algorithm and Simulated Annealing...")

    # Check if results are available
    if st.session_state.fpa_result and st.session_state.sa_result:
        # Display results side by side
        col1, col2 = st.columns(2)

        with col1:
            st.write("Flower Pollination Algorithm")
            st.write(f"Best Distance: {st.session_state.fpa_result['best_distance']:.2f} km")
            st.write(f"Execution Time: {st.session_state.fpa_result['execution_time']:.2f} seconds")
            st.write(f"Stability (Std Dev): {np.std(st.session_state.fpa_distances):.2f} km")  # Fixed format specifier
            fpa_map = create_map(city_data, st.session_state.fpa_result["best_route"])
            st_folium(fpa_map, width=700, height=500)  # Replace folium_static with st_folium

        with col2:
            st.write("Simulated Annealing")
            st.write(f"Best Distance: {st.session_state.sa_result['best_distance']:.2f} km")
            st.write(f"Execution Time: {st.session_state.sa_result['execution_time']:.2f} seconds")
            st.write(f"Stability (Std Dev): {np.std(st.session_state.sa_distances):.2f} km")  # Fixed format specifier
            sa_map = create_map(city_data, st.session_state.sa_result["best_route"])
            st_folium(sa_map, width=700, height=500)  # Replace folium_static with st_folium
    else:
        st.write("Please run both algorithms in the 'Run Algorithms' tab to compare results.")

# Close Matplotlib figures to avoid memory warnings
plt.close("all")