import networkx as nx
import random
import pandas as pd
import matplotlib.pyplot as plt
import collections


def describe_network_directed(G: nx.DiGraph):
    # Calculate approximate average shortest path lengths
    G_undirected = G.to_undirected()

    largest_cc = max(nx.connected_components(G_undirected), key=len)
    G_lcc = G_undirected.subgraph(largest_cc)

    sampled_nodes = random.sample(list(G_lcc.nodes()), min(500, len(G_lcc)))

    all_distances = [
        dist
        for node in sampled_nodes
        for dist in nx.single_source_shortest_path_length(G_lcc, node).values()
        if dist > 0
    ]

    approx_aspl = sum(all_distances) / len(all_distances)


    max_comp = len(largest_cc)

    # To get a dictionary of node: in_degree
    in_degrees = dict(G.in_degree())

    # Average in deg
    avg_in = sum(in_degrees.values()) / len(G)

    df = pd.DataFrame({"Nodes": [G.number_of_nodes()],
                       "Edges": [G.number_of_edges()],
                       "Average In/Out Degree": [avg_in],
                       "Weakly Connected Components": [nx.number_weakly_connected_components(G)],
                       "Average Clustering Coefficient": [nx.average_clustering(G)],
                       "Largest Connected Component": [max_comp],
                       "Average Shortest Path": [approx_aspl]
                       })
    return df


def describe_network_undirected(G: nx.Graph):
    
    # Get Largest Connected Component
    largest_cc = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(largest_cc)

    # Calculate approximate average shortest path lengths
    sampled_nodes = random.sample(list(G_lcc.nodes()), min(500, len(G_lcc)))
    all_distances = [
        dist
        for node in sampled_nodes
        for dist in nx.single_source_shortest_path_length(G_lcc, node).values()
        if dist > 0
    ]
    approx_aspl = sum(all_distances) / len(all_distances) if all_distances else 0

    # Average degree
    avg_deg = sum(dict(G.degree()).values()) / len(G)

    df = pd.DataFrame({
        "Nodes": [G.number_of_nodes()],
        "Edges": [G.number_of_edges()],
        "Average Degree": [avg_deg],
        "Connected Components": [nx.number_connected_components(G)],
        "Average Clustering Coefficient": [nx.average_clustering(G)],
        "Largest Connected Component": [len(largest_cc)],
        "Average Shortest Path": [approx_aspl]
    })
    return df


def plot_degree_distribution(graphs, labels):
    plt.figure(figsize=(10, 6))
    
    for G_plot, label in zip(graphs, labels):
        # Get degrees
        degrees = [d for n, d in G_plot.degree()]
        
        # Count frequencies
        degree_counts = collections.Counter(degrees)
        x, y = zip(*sorted(degree_counts.items()))
        
        # Plot on log-log scale with markers and a connecting line
        plt.loglog(x, y, marker='o', markersize=4, linestyle='-', linewidth=1, label=label, alpha=0.7)

    plt.title("Degree Distribution (Log-Log Scale)")
    plt.xlabel("Degree (k)")
    plt.ylabel("Frequency P(k)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()