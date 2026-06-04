import networkx as nx
import random
import pandas as pd
import matplotlib.pyplot as plt
import collections


def describe_network_directed(G: nx.DiGraph):
    """Calculate and return key network metrics for a directed graph."""
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
    """Calculate and return key network metrics for an undirected graph."""
    
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
    """Plot the degree distribution of one or more graphs on a log-log scale."""
    plt.figure(figsize=(10, 6))
    
    for G_plot, label in zip(graphs, labels):
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


def get_top_papers_by_centrality(titleabs: pd.DataFrame, ranking: dict, centrality_name: str, node_mapping: pd.DataFrame, top=20):
    """Get the top papers sorted by their centrality scores."""
    ranks = list(ranking.items())
    df = pd.DataFrame(ranks, columns=["node idx", f"{centrality_name}"])
    df = df.sort_values(by=f"{centrality_name}", ascending=False)
    df = df.merge(node_mapping, on="node idx")

    top_paper_ids = df.head(top)["paper id"].astype(int).tolist()

    existing_ids = [pid for pid in top_paper_ids if pid in titleabs.index]
    top_papers = titleabs.loc[existing_ids]

    return top_papers
    
def evaluate_communities(G, communities: list[set], node_category_mapping: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    modularity = nx.community.modularity(G, communities)
    communities_categories = []
    communities_sizes_sum = 0
    for community in communities:
        communities_sizes_sum += len(community)
        categories = {}
        for node in community:
            category = node_category_mapping.loc[node]["arxiv category"]
            if category in categories:
                categories[category] += 1
            else:
                categories[category] = 1
        max_count = 0
        max_category = ""
        for category, count in categories.items():
            if count > max_count:
                max_category = category
                max_count = count
        
        
        percentage_for_max_category = max_count / len(community)
        communities_categories.append({"max category": max_category,
                                        "percentage": percentage_for_max_category})

    df = pd.DataFrame(communities_categories)
    df = df.sort_values(by="percentage", ascending=False)
    avg_community_size = communities_sizes_sum / len(communities)
    return (modularity, avg_community_size, df)
    
