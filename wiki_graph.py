import wikipedia as wiki
import networkx as netx
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import List


@dataclass
class SubjectInfo:
    subject: str
    summary: str
    page: wiki.WikipediaPage


def tree_scan(graph: netx.DiGraph, parent: SubjectInfo, cor_limit, depth: int, tree_depth: int, executor: ThreadPoolExecutor, model):
    threads: List[ThreadPoolExecutor.Future] = []

    for i in parent.page.links:
        try:
            child_sum = wiki.summary(i, auto_suggest=False)
            child_subject = i
        except wiki.exceptions.DisambiguationError as e:
            if e.options.index(i[0]) == -1:
                child_subject = e.options[e.options.index(int(i[0]))]  # it has been shown that this occurs with numbers
            else:
                child_subject = e.options[e.options.index(i[0])]
            try:
                child_sum = wiki.summary(child_subject, auto_suggest=False)
            except wiki.exceptions.PageError:
                continue
        except wiki.exceptions.PageError:
            continue
        child_page = wiki.page(child_subject, auto_suggest=False)
        sums = [parent.summary, child_sum]
        sum_embeddings = model.encode(sums)
        correlation = util.pytorch_cos_sim(sum_embeddings[0], sum_embeddings[1])
        if correlation.item() > cor_limit:
            # add_node and add weighted edge
            graph.add_weighted_edges_from([(parent.subject, child_subject, correlation.item())])
            print("(" + str(parent.subject) + ", " + str(child_subject) + ", " + str(correlation.item()) + ")")
            if depth < tree_depth and (graph.has_node(child_subject) == 0 or graph.out_degree(child_subject) == 0):
                # check if this node has already been scanned
                child_node = SubjectInfo(child_subject, child_sum, child_page)
                future = executor.submit(tree_scan(graph, child_node, cor_limit, depth + 1, tree_depth, executor, model))
                threads.append(future)

    for future in threads:  # parent waits for all children to finish before exiting
        future.result()


def wiki_graph(subject: str, cor_limit, tree_depth: int):

    graph = netx.DiGraph()

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

    with ThreadPoolExecutor() as executor:
        # initialize with root node and depth to 1 (minimum depth)
        tree_scan(graph, orig, cor_limit, 1, tree_depth, executor, model)

    netx.write_gexf(graph, "wiki_graph_output.gexf")
