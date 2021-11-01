
import numpy as np
import networkx as nx
import logging
logging.basicConfig(level=logging.INFO)

def k_shortest(n, probs, k, log_probs=True):
    """
    Shortest path finder with a number of nodes and probability list. This is for the list of sentences.

    :param n: number of nodes (=sentences)
    :param probs: a list of probabilities, in format ((n1, prev_topic), (n2, topic), {"weight":p})
    :param log_probs: if probabilities are in log
    :param k:
    :return:
    """

    # we need a probability table for each n, prev_topic to each n, topic. but we will prune long segments
    #[((n1, prev_topic), (n2, topic), {"weight": p}]
    # nodes will be tuples of the form (n, prev_topic)
    # so the returned list will also have the topics
    start = (0, -1)
    end = (n + 1, -1)  # this means that the index is not included! (like in segment_score())

    id2node = list(set([start] + [tuple(n) for n1, n2, _ in probs for n in [n1, n2]] + [end]))
    node2id = {n: i for i, n in enumerate(id2node)}

    logging.info("Probs:" + str(len(probs)))
    _probs = [(node2id[tuple(n1)], node2id[tuple(n2)], {"weight": -w["weight"]}) for n1, n2, w in probs]  # we want the shortest path of the minus probabilities
    logging.info("_Probs:" + str(len(_probs)))
    # reduce to edges only within the range

    _G = nx.DiGraph()
    _G.add_edges_from(_probs)
    # logging.info("predeccessors: ")
    # logging.info(list(_G.predecessors(node2id[end])))
    # new_nodes = [n for n in nx.dfs_preorder_nodes(_G, source=node2id[start], depth_limit=k)]
    new_nodes = [n for n in nx.bfs_tree(_G, source=node2id[start], depth_limit=k).nodes]
    logging.info("_Nodes:" + str(len([n for n in new_nodes])))
    to_remove = [e for e in _G.edges() if e[0] not in new_nodes or e[1] not in new_nodes]
    logging.info("_to remove:" + str(len(to_remove)))
    _G.remove_edges_from(to_remove)

    # new_probs = [e for e in _G.edges(data=True) if e[0] in new_nodes and e[1] in new_nodes]
    # # new_probs = [e + ({"weight": _G.edges[e]['weight']},) for e in list(nx.dfs_edges(_G, source=node2id[start], depth_limit=k))]
    # G = nx.DiGraph()
    # G.add_edges_from(new_probs)
    #
    logging.info("Nodes:" + str(len(_G.nodes)))  # this includes also a few nodes that might be unreachable within k steps
    logging.info("Edges:" + str(len(_G.edges)))
    # path = k_edge_shortest_path(G, start=node2id[start], end=node2id[end], k=k)[:-1]  # we don't need the last node
    path = k_edge_shortest_path(_G, start=node2id[start], end=node2id[end], k=k)[:-1]  # we don't need the last node
    return [id2node[p] for p in path]

def k_edge_shortest_path(G, start, end, k):
    """
    Find shotest among all paths from start to end in G with k edges
    :param G: a digraph
    :return: list of vertices for a shortest path
    """
    n = len(G.nodes)

    prevs = np.full((n, n, k), -1)  # previous node for a path between two nodes and a length
    shortest = np.full((n, n, k), np.inf)
    logging.info(f"n: {n}")

    # save the weights to save time
    weights = np.full((n, n), np.inf)
    # for n1, n2 in nx.dfs_edges(G, source=start, depth_limit=k):  # we don't care about the rest
    for n1, n2 in G.edges:  # we don't care about the rest
        weights[n1, n2] = G.edges[n1, n2]['weight']

    # nodes = list(nx.dfs_preorder_nodes(G, source=start, depth_limit=k))
    nodes = list(nx.bfs_tree(G, source=start, depth_limit=k).nodes)
    predecessors = {n: list(G.predecessors(n)) for n in nodes}
    logging.info("Start and end predeccesors:")
    logging.info(start)
    logging.info(end)
    logging.info(predecessors[start])
    logging.info(predecessors[end])

    for n1, n2 in G.edges:
    # for n1, n2 in nx.dfs_edges(G, source=start, depth_limit=k):
        prevs[n1, n2, 0] = n1
        shortest[n1, n2, 0] = weights[n1, n2]

    for _k in range(1, k):
        logging.info(f"k: {_k}")
        for n1 in nodes:
            # logging.info(f"n1: {n1}")
            for n2 in nodes:
                # in_nodes = [(shortest[n1, e[0], _k-1] + G.edges[e]['weight'], e[0]) for e in G.in_edges(n2)]
                # in_nodes = [(shortest[n1, n, _k-1] + weights[n, n2], n) for n in predecessors[n2]]
                in_costs = shortest[n1, predecessors[n2], _k-1] + weights[predecessors[n2], n2]
                if len(in_costs) > 0:
                    m = np.argmin(in_costs)
                    shortest[n1, n2, _k], prevs[n1, n2, _k] = in_costs[m], predecessors[n2][m]

                # if len(in_nodes) > 0:
                #     shortest[n1, n2, _k], prevs[n1, n2, _k] = min(in_nodes)

    in_costs = shortest[start, predecessors[end], k-1] + weights[predecessors[end], end]
    m = np.argmin(in_costs)
    path = [predecessors[end][m], end]

    # in_nodes = [(shortest[start, e[0], k-1] + G.edges[e]['weight'], e[0]) for e in G.in_edges(end)]
    # path = [min(in_nodes)[1], end]
    total_weight = in_costs[m]
    for _k in range(1, k)[::-1]:
        path.insert(0, prevs[start, path[0], _k])
    return path  # we don't want to include start, since these are the ends!
    # return [start] + path


if __name__ == "__main__":
    G = nx.gnp_random_graph(n=8, p=0.5, seed=None, directed=True)
    # G = nx.gnc_graph(10)
    # G = nx.random_k_out_graph(n=10, k=5, alpha=0.1)
    # G = nx.scale_free_graph(n=10, alpha=0.1, beta=0.9)
    for e in G.edges:
        G[e[0]][e[1]]['weight'] = np.random.uniform(0, 1)
    start, end = list(G.nodes)[0], list(G.nodes)[-1]
    print(k_edge_shortest_path(G, start=0, end=7, k=2))
    print("done")

