import sys
import os
import wiki_graph
from torch.multiprocessing import set_start_method
import json


def main(argv):
    subject = ""
    cor_index = 1
    for i in range(len(argv)):
        try:
            float(argv[i])
            cor_index = i  # save index of correlation limit
            break
        except ValueError:
            subject += argv[i] + " "
    subject = subject[:-1]  # don't store the last space
    print("Searching in wikipedia for: " + subject)
    if subject == '' or float(argv[cor_index]) < 0 or float(argv[cor_index]) >= 1 or argv[cor_index + 1].isdigit() == 0 \
            or int(argv[cor_index + 1]) < 1:
        # arguments over index cor_index + 1 are neglected
        # correlation limit is the lower limit for the correlation metric which is in the range [0,1]
        # isdigit() returns false if tree depth is decimal
        print("Usage is: python3 ./main.py <subject: string> <correlation limit: float> <tree depth: int>")
        exit()
    # graph = wiki_graph.wiki_graph(subject, float(argv[cor_index]), int(argv[cor_index + 1]))
    graph = wiki_graph.netx.read_gexf(subject+ '_' + str(argv[cor_index]) + '_' + str(argv[cor_index + 1]) + "_graph_output.gexf")
    dictionary = wiki_graph.analyze_graph(graph, subject, int(argv[cor_index + 1]))
    with open("metrics_dictionary.json", "w") as fp:
        json.dump(dictionary, fp)  # encode dict into JSON

    del graph, dictionary
    return


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Usage is: python3 ./main.py <subject> <correlation limit> <tree depth>")
        exit()

    # Set TOKENIZERS_PARALLELISM to 'true'
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    try:
        set_start_method('spawn')  # needed to avoid model's re-initialization
    except RuntimeError as error:
        print("Error message:", error)
        pass

    main(sys.argv[1:])  # don't pass the program call to the main function

    exit()
