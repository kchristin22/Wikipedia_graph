import wikipedia as wiki
import networkx as netx
from sentence_transformers import SentenceTransformer, util
from dataclasses import dataclass
from torch.multiprocessing import Pool
from typing import List
import itertools
from time import sleep


@dataclass
class SubjectInfo:
    subject: str
    summary: str
    page: wiki.WikipediaPage
    cor: float


def parent_node_already_exists(graph_edges, subject):
    return any(subject == edge[0] for edge in graph_edges)


def find_child(args):
    results: List[SubjectInfo] = []
    for arg in args:
        parent, parent_link, cor_limit, model = arg
        try:
            child_sum = wiki.summary(parent_link, auto_suggest=False)
            sleep(3)  # make this dynamic depending on cor limit
            child_subject = parent_link
        except wiki.exceptions.DisambiguationError as e:
            if any(parent_link.split()[0] in i for i in e.options):
                split_parts = parent_link.split()
                # keep all splits except the parenthesis
                gen_subject = ' '.join(
                    split_parts[:next((i for i, part in enumerate(split_parts) if part.startswith('(')), len(split_parts))])
                child_index = next(((option, i) for i, option in enumerate(e.options) if f'{gen_subject} ' in option), (None, 0))  # match the exact words in options or choose the first one
                print("child_index: ", child_index)
                while "(disambiguation)" in e.options[child_index]:  # avoid disambiguation error again
                    child_index = child_index + 1
                child_subject = e.options[child_index]
            else:
                # it has been shown that this occurs with numbers
                child_subject = next((option for option in e.options if int(parent_link.split()[0]) in option), None)
            try:
                child_sum = wiki.summary(child_subject, auto_suggest=False)
                sleep(3)
            except wiki.exceptions.PageError:
                continue
        except wiki.exceptions.PageError:
            continue
        child_page = wiki.page(child_subject, auto_suggest=False)
        sleep(3)
        sums = [parent.summary, child_sum]
        sum_embeddings = model.encode(sums)
        correlation = util.pytorch_cos_sim(sum_embeddings[0], sum_embeddings[1])
        if correlation.item() > cor_limit:
            results.append(SubjectInfo(child_subject, child_sum, child_page, correlation.item()))
        else:
            continue

    return results


def tree_scan(graph_edges, parent: SubjectInfo, cor_limit, depth: int, tree_depth: int, model, executor):
    args = list(zip([parent]*len(parent.page.links), parent.page.links, [cor_limit]*len(parent.page.links), [model]*len(parent.page.links)))
    num_processes = executor._processes  # default number of processes spawned based on your system
    # partition the args into chunks for each process
    chunk_size = len(args) // num_processes
    arg_chunks = [args[i:i + chunk_size] for i in range(0, len(args), chunk_size)]

    child_nodes = executor.map(find_child, arg_chunks)
    child_nodes = list(itertools.chain.from_iterable(child_nodes))  # remove separation of each process' list

    for child_node in child_nodes:
        if not child_node:
            continue
        # check if the child topic exists as a parent with a different casing or in plural
        lowercase_first_column = [edge[0].lower() for edge in graph_edges]
        found_index = next(
            (index for index, value in enumerate(lowercase_first_column) if child_node.subject.lower() in value), -1)
        if found_index != -1:
            child_node.subject = graph_edges[found_index][0]
        else:
            found_index = next(
                (index for index, value in enumerate(lowercase_first_column) if value in child_node.subject.lower()),
                -1)  # check if this child is the plural form of a parent
            if found_index != -1:
                child_node.subject = graph_edges[found_index][0]
        # check if this combination is already included in the edges:
        if (parent.subject, child_node.subject, child_node.cor) in graph_edges:
            continue
        # add_node and add weighted edge
        graph_edges.append((parent.subject, child_node.subject, child_node.cor))
        print("(" + str(parent.subject) + ", " + str(child_node.subject) + ", " + str(child_node.cor) + ")")
        # check if this node has already been scanned
        if depth < tree_depth and parent_node_already_exists(graph_edges, child_node.subject) == 0:
            child_node = SubjectInfo(child_node.subject, child_node.summary, child_node.page, child_node.cor)
            tree_scan(graph_edges, child_node, cor_limit, depth + 1, tree_depth, model, executor)


def wiki_graph(subject: str, cor_limit, tree_depth: int):

    graph = netx.DiGraph()
    graph_edges = []  # a list to store edges

    try:
        orig_sum = wiki.summary(subject)
        page = wiki.page(subject)
        page_title = wiki.page(subject).title
        if page_title != subject:  # avoid having duplicates of the root node afterwards
            subject = page_title
            print("Subject changed to " + subject)
    except wiki.DisambiguationError as e:
        subject = e.options[0]
        print("Subject changed to " + subject)
        orig_sum = wiki.summary(subject, auto_suggest=False)
        page = wiki.page(subject, auto_suggest=False)

    orig = SubjectInfo(subject, orig_sum, page, cor=1)

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    with Pool() as executor:
        # initialize with root node and depth to 1 (minimum depth)
        tree_scan(graph_edges, orig, cor_limit, 1, tree_depth, model, executor)

    executor.join()

    graph.add_weighted_edges_from(graph_edges)
    netx.write_gexf(graph, subject + str(cor_limit) + str(tree_depth) + "_graph_output.gexf")

    del graph_edges
    return graph


def find_important_edges(graph, subject: str, tree_depth: int):
    all_children = []
    cur_depth_nodes = [[] for _ in range(tree_depth + 1)]
    cur_depth_nodes[0].append(subject)
    for depth in range(tree_depth):
        for node in cur_depth_nodes[depth]:

            cur_depth_nodes[depth + 1].extend([i for i in graph.successors(node)])

            if node == subject:
                cur_children = graph.edges.data("weight",
                                                nbunch=node)  # return 3-tuples of edge source, edge destination and weight
                all_children.extend(cur_children)
                continue

            # if parent is not the root, we need to calculate the new weight of its children
            # based on its connection to the root

            # find parent node included in the previous depth
            cur_parent = [i for i in cur_depth_nodes[depth - 1] if i in graph.predecessors(node)]  # list
            this_node_weight = graph.edges[cur_parent[0], node]["weight"]  # this node's weight in regard to root

            for successor in graph.successors(node):
                this_successor_weight = graph.edges[node, successor]["weight"]
                # scale weight of this successor
                all_children.append([node, successor, this_successor_weight * this_node_weight])

    return all_children


def analyze_graph(graph, subject: str, tree_depth: int):

    graph_dict = {}

    # Approximations and Heuristics
    avg_cluster = netx.average_clustering(graph)
    graph_dict["avg clustering"] = avg_cluster
    print("average clustering: ", avg_cluster)

    # Centrality
    centrality = netx.degree_centrality(graph)
    graph_dict["degree centrality"] = centrality
    sorted_values = sorted(graph_dict["degree centrality"].items(), key=lambda x: x[1], reverse=True)
    print("Top 5 central nodes based on their degree: ", sorted_values[:5])

    in_centrality = netx.in_degree_centrality(graph)
    graph_dict["in degree centrality"] = in_centrality
    sorted_values = sorted(graph_dict["in degree centrality"].items(), key=lambda x: x[1], reverse=True)
    print("Top 5 central nodes based on their in degree: ", sorted_values[:5])

    out_centrality = netx.out_degree_centrality(graph)
    graph_dict["out degree centrality"] = out_centrality
    sorted_values = sorted(graph_dict["out degree centrality"].items(), key=lambda x: x[1], reverse=True)
    print("Top 5 central nodes based on their out degree: ", sorted_values[:5])

    eigenvector_centr = netx.eigenvector_centrality_numpy(graph)
    graph_dict["eigenvector centrality"] = eigenvector_centr
    sorted_values = sorted(graph_dict["eigenvector centrality"].items(), key=lambda x: x[1], reverse=True)
    print("Top 5 central nodes based on the eigenvectors: ", sorted_values[:5])

    voterank = netx.voterank(graph)
    graph_dict["voterank influential nodes"] = voterank
    print("influential nodes derived by VoteRank: ", voterank)

    # Components
    num_strongly_con = netx.number_strongly_connected_components(graph)
    graph_dict["num of strongly connected components"] = num_strongly_con
    print("number of strongly connected components: ", num_strongly_con)

    # Cycles
    try:
        center_cycles = netx.find_cycle(graph, subject, orientation="original")
        graph_dict["main theme cycles"] = list(center_cycles)
        print("Found edges forming cycle(s) that include the main theme: ", list(center_cycles))
    except netx.NetworkXNoCycle:
        print("No cycle found")
        graph_dict["main theme cycles"] = []

    # Link Analysis
    pagerank = netx.pagerank(graph)
    graph_dict["pagerank influential nodes"] = pagerank
    sorted_values = sorted(graph_dict["pagerank influential nodes"].items(), key=lambda x: x[1], reverse=True)
    print("Top 5 most influential nodes based on PageRank: ", sorted_values[:5])

    # Edge analysis
    out_edges = find_important_edges(graph, subject, tree_depth)
    graph_dict["all nodes correlation with the main theme"] = out_edges
    sorted_values = sorted(out_edges, key=lambda x: x[2], reverse=True)
    print("Top 5 edges with the highest weights: ", sorted_values[:5])

    return graph_dict


