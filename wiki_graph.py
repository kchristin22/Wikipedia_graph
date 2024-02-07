import wikipedia as wiki
import networkx as netx
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from dataclasses import dataclass
from torch.multiprocessing import Manager, Pool
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

                child_subject = next((option for option in e.options if gen_subject in option), None)
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
    child_nodes = list(itertools.chain.from_iterable(child_nodes))

    for child_node in child_nodes:
        if not child_node:
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
    netx.write_gexf(graph, "wiki_graph_output.gexf")
    del graph_edges, graph
