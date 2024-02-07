import wikipedia as wiki
import networkx as netx
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from dataclasses import dataclass
from torch.multiprocessing import Manager, Process, Pool, active_children
from time import sleep


@dataclass
class SubjectInfo:
    subject: str
    summary: str
    page: wiki.WikipediaPage


def parent_node_already_exists(graph_edges, subject):
    return any(subject == edge[0] for edge in graph_edges)


def find_child(args):
    results = []
    for arg in args:
        parent, parent_link, cor_limit, model = arg
        # print("Trying for: ", parent_link)
        try:
            child_sum = wiki.summary(parent_link, auto_suggest=False)
            sleep(1.5)
            child_subject = parent_link
        except wiki.exceptions.DisambiguationError as e:
            # print(e.options)
            if any(parent_link.split()[0] in i for i in e.options):
                split_parts = parent_link.split()
                # keep all splits except the parenthesis
                gen_subject = ' '.join(
                    split_parts[:next((i for i, part in enumerate(split_parts) if part.startswith('(')), len(split_parts))])

                # print("After splits: ", gen_subject)

                child_subject = next((option for option in e.options if gen_subject in option), None)
                # print("After next:", child_subject)
            else:
                # print("Didn't find: ", parent_link)
                # it has been shown that this occurs with numbers
                child_subject = next((option for option in e.options if int(parent_link.split()[0]) in option), None)
            try:
                child_sum = wiki.summary(child_subject, auto_suggest=False)
                sleep(1.5)
            except wiki.exceptions.PageError:
                continue
        except wiki.exceptions.PageError:
            continue
        child_page = wiki.page(child_subject, auto_suggest=False)
        sleep(1.5)
        sums = [parent.summary, child_sum]
        sum_embeddings = model.encode(sums)
        correlation = util.pytorch_cos_sim(sum_embeddings[0], sum_embeddings[1])
        # print("made it here")
        if correlation.item() > cor_limit:
            print(child_subject)
            results.append(SubjectInfo(child_subject, child_sum, child_page))
        else:
            continue

    return results


def tree_scan(graph_edges, parent: SubjectInfo, cor_limit, depth: int, tree_depth: int, model, executor):
    processes = []

    print("here")
    args = list(zip([parent]*len(parent.page.links), parent.page.links, [cor_limit]*len(parent.page.links), [model]*len(parent.page.links)))
    num_processes = executor._processes  # minus active workers
    # Partition the args into chunks for each process
    chunk_size = len(args) // num_processes
    arg_chunks = [args[i:i + chunk_size] for i in range(0, len(args), chunk_size)]

    child_nodes = list(executor.map(find_child, arg_chunks))

    print("found children")
    for child_node in child_nodes:
        if child_node is None:
            continue
            print(child_node)
            # add_node and add weighted edge
            graph_edges.append((parent.subject, child_subject, correlation.item()))
            # graph.add_weighted_edges_from([(parent.subject, child_subject, correlation.item())])
            print("(" + str(parent.subject) + ", " + str(child_subject) + ", " + str(correlation.item()) + ")")
            if depth < tree_depth and parent_node_already_exists(graph_edges, child_subject) == 0:
                # check if this node has already been scanned
                child_node = SubjectInfo(child_subject, child_sum, child_page)
                tree_scan(graph_edges, child_node, cor_limit, depth + 1, tree_depth, model, executor)
                # task = executor.submit(tree_scan, graph_edges, child_node, cor_limit, depth + 1, tree_depth, model)
                # processes.append(task)

    # wait(processes, return_when="ALL_COMPLETED")  # parent waits for all children to finish before exiting


def wiki_graph(subject: str, cor_limit, tree_depth: int):

    graph = netx.DiGraph()
    graph_edges = Manager().list()  # A managed list to store edges

    try:
        orig_sum = wiki.summary(subject)
        page = wiki.page(subject)
    except wiki.DisambiguationError as e:
        subject = e.options[0]
        print("Subject changed to " + subject)
        orig_sum = wiki.summary(subject, auto_suggest=False)
        page = wiki.page(subject, auto_suggest=False)

    orig = SubjectInfo(subject, orig_sum, page)

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    with Pool() as executor:
        # initialize with root node and depth to 1 (minimum depth)
        tree_scan(graph_edges, orig, cor_limit, 1, tree_depth, model, executor)

    # graph_edges.shm.close()
    # graph_edges.shm.unlink()

    graph.add_weighted_edges_from(graph_edges)
    netx.write_gexf(graph, "wiki_graph_output.gexf")
