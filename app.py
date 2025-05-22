import streamlit as st
import matplotlib.pyplot as plt
import time
import numpy as np
import folium
from streamlit_folium import st_folium
from algorithms import CityData, FlowerPollinationAlgorithm, SimulatedAnnealing, GeneticAlgorithm
from folium.plugins import AntPath
from visualizations import compare_execution_times

# Streamlit app title
st.title("Traveling Salesman Problem - Tunisian Cities")

# Load city data
city_data = CityData()

# Initialize session state for storing results
if "fpa_result" not in st.session_state:
    st.session_state.fpa_result = None
if "sa_result" not in st.session_state:
    st.session_state.sa_result = None
if "ga_result" not in st.session_state:
    st.session_state.ga_result = None  # Initialize GA result

if "fpa_times" not in st.session_state:
    st.session_state.fpa_times = []
if "sa_times" not in st.session_state:
    st.session_state.sa_times = []
if "ga_times" not in st.session_state:
    st.session_state.ga_times = []  # Initialize GA times

if "fpa_distances" not in st.session_state:
    st.session_state.fpa_distances = []
if "sa_distances" not in st.session_state:
    st.session_state.sa_distances = []
if "ga_distances" not in st.session_state:
    st.session_state.ga_distances = []  # Initialize GA distances

if "fpa_histories" not in st.session_state:
    st.session_state.fpa_histories = []  # Initialize FPA histories
if "sa_histories" not in st.session_state:
    st.session_state.sa_histories = []  # Initialize SA histories
if "ga_histories" not in st.session_state:
    st.session_state.ga_histories = []  # Initialize GA histories

# Initialize session state for storing last plots
if "fpa_last_plot" not in st.session_state:
    st.session_state.fpa_last_plot = None
if "sa_last_plot" not in st.session_state:
    st.session_state.sa_last_plot = None
if "ga_last_plot" not in st.session_state:
    st.session_state.ga_last_plot = None  # Initialize GA last plot

# Helper function to plot routes using Matplotlib
def plot_route(city_data, route, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    route = list(route) + [route[0]]  # Complete the loop
    ax.scatter(city_data.cities[:, 0], city_data.cities[:, 1], c='red', s=100, zorder=5, label='Cities')
    
    
    
    for i, (x, y) in enumerate(city_data.cities):
        order = route.index(i) + 1 if i in route else None
        ax.text(x, y, f' {city_data.city_names[i]} ({order})', fontsize=8, ha='right', zorder=10)
    ax.plot(city_data.cities[route, 0], city_data.cities[route, 1], 'b-', lw=2, alpha=0.7, label='Route')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# Enhanced helper function to create a folium map with animated route and arrows
def create_map(city_data, route):
    m = folium.Map(location=[city_data.cities[:, 1].mean(), city_data.cities[:, 0].mean()], zoom_start=7)

    # Add city markers with order numbers
    for i, (lon, lat) in enumerate(city_data.cities):
        order = route.index(i) + 1 if i in route else None
        folium.Marker(
            [lat, lon], 
            popup=f"{city_data.city_names[i]} ({order})" if order else city_data.city_names[i]
        ).add_to(m)

    # Prepare the route coordinates
    route_coords = [[city_data.cities[city, 1], city_data.cities[city, 0]] for city in route]
    route_coords.append(route_coords[0])  # Close the loop by returning to the starting city

    # Add animated route using AntPath
    AntPath(
        locations=route_coords,
        color="blue",
        pulse_color="white",
        weight=6,
        delay=300,  # ~1 second per segment
        dash_array=[60, 20],
    ).add_to(m)

    # Add arrows to indicate direction
    for i in range(len(route_coords) - 1):
        start = route_coords[i]
        end = route_coords[i + 1]
        midpoint = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2]  # Calculate midpoint

        # Calculate the angle of the arrowhead
        angle = np.arctan2(end[0] - start[0], end[1] - start[1]) * (180 / np.pi)

        folium.PolyLine(
            [start, end],
            color="blue",
            weight=2,
            opacity=0.8
        ).add_to(m)

        # Add an arrowhead marker at the midpoint with the correct rotation
        folium.RegularPolygonMarker(
            location=midpoint,
            fill_color="blue",
            number_of_sides=3,
            radius=10,
            rotation=angle + 90  # Adjust rotation to align with the line
        ).add_to(m)

    return m

# Remove the Sensitivity Analysis tab
tab1, = st.tabs(["Run Algorithms"])

with tab1:
    # Sidebar for algorithm selection
    st.sidebar.title("Select Algorithm")
    algorithm = st.sidebar.radio("Choose an algorithm to run:", ("Flower Pollination Algorithm", "Simulated Annealing", "Genetic Algorithm"))

    # Display the last run if available
    if algorithm == "Flower Pollination Algorithm" and st.session_state.fpa_result:
        st.write("Last Run - Flower Pollination Algorithm")
        if st.session_state.fpa_last_plot:
            st.pyplot(st.session_state.fpa_last_plot)  # Pass the figure explicitly
        # Display Folium map for the last run
        fpa_map = create_map(city_data, st.session_state.fpa_result["best_route"])
        st_folium(fpa_map, key="fpa_last_map", width=700, height=500)

    elif algorithm == "Simulated Annealing" and st.session_state.sa_result:
        st.write("Last Run - Simulated Annealing")
        if st.session_state.sa_last_plot:
            st.pyplot(st.session_state.sa_last_plot)  # Pass the figure explicitly
        # Display Folium map for the last run
        sa_map = create_map(city_data, st.session_state.sa_result["best_route"])
        st_folium(sa_map, key="sa_last_map", width=700, height=500)

    elif algorithm == "Genetic Algorithm" and st.session_state.get("ga_result"):
        st.write("Last Run - Genetic Algorithm")
        if st.session_state.get("ga_last_plot"):
            st.pyplot(st.session_state.ga_last_plot)
        # Display Folium map for the last run
        ga_map = create_map(city_data, st.session_state["ga_result"]["best_route"])
        st_folium(ga_map, key="ga_last_map", width=700, height=500)

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
                plot_placeholder.pyplot(fig)  # Pass the figure explicitly
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

            # Save the last Matplotlib plot
            st.session_state.fpa_last_plot = fig

            # Display the final result on a Folium map
            st.write("Final Route for Flower Pollination Algorithm (Folium Map)")
            fpa_map = create_map(city_data, st.session_state.fpa_result["best_route"])
            st_folium(fpa_map, key="fpa_final_map", width=700, height=500)

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
                plot_placeholder.pyplot(fig)  # Pass the figure explicitly
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

            # Save the last Matplotlib plot
            st.session_state.sa_last_plot = fig

            # Display the final result on a Folium map
            st.write("Final Route for Simulated Annealing (Folium Map)")
            sa_map = create_map(city_data, st.session_state.sa_result["best_route"])
            st_folium(sa_map, key="sa_final_map", width=700, height=500)

        elif algorithm == "Genetic Algorithm":
            st.write("Running Genetic Algorithm...")
            ga = GeneticAlgorithm(city_data)
            progress_placeholder = st.empty()
            plot_placeholder = st.empty()

            start_time = time.time()
            for generation in range(ga.n_generations):
                ga.optimize()
                progress_placeholder.write(f"Generation: {generation + 1}/{ga.n_generations}, Best Distance: {ga.best_distance:.2f} km")

                # Update the Matplotlib plot dynamically
                fig, ax = plt.subplots(figsize=(8, 6))
                current_route = list(ga.best_solution) + [ga.best_solution[0]]
                ax.scatter(city_data.cities[:, 0], city_data.cities[:, 1], c='red', s=100, zorder=5, label='Cities')
                for i, (x, y) in enumerate(city_data.cities):
                    ax.text(x, y, f' {city_data.city_names[i]}', fontsize=8, ha='right', zorder=10)
                ax.plot(city_data.cities[current_route, 0], city_data.cities[current_route, 1], 'b-', lw=2, alpha=0.7, label='Route')
                ax.set_title(f"GA Generation {generation + 1}", fontsize=12)
                ax.set_xlabel("Longitude (°E)")
                ax.set_ylabel("Latitude (°N)")
                ax.legend()
                plot_placeholder.pyplot(fig)
                plt.close(fig)

            execution_time = time.time() - start_time
            st.session_state["ga_result"] = {
                "best_distance": ga.best_distance,
                "best_route": list(ga.best_solution),
                "execution_time": execution_time
            }

            # Save the last Matplotlib plot
            st.session_state["ga_last_plot"] = fig

            # Display the final result on a Folium map
            st.write("Final Route for Genetic Algorithm (Folium Map)")
            ga_map = create_map(city_data, st.session_state["ga_result"]["best_route"])
            st_folium(ga_map, key="ga_final_map", width=700, height=500)