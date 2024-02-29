# Wikipedia_graph

The purpose of this project is to implement a network of Wikipedia articles related to a specific subject.
In more details, the Wikipedia page with the subject as a title serves as the root node and is searched for links
that reference other correlated articles. Afterwards, each such link is also scanned for articles related to it,
forming a graph reminiscing a tree.

#### Tools and packages used for graph formation and analysis: `Gephi`, `NetworkX`, `Wikipedia`

# Usage
`python3 ./main.py <subject> <correlation limit> <tree depth>`

#### Notes:
* This repo contains the files created for a specific program input: Quantum computing 0.5 2
* If you want to read a gexf file that you have already created, without the need to re-write it, you may uncomment line 28 and comment out line 27 in `main.py` and run the program with the input arguments that were previously used for the graph's formation. For instance, running: `python3 ./main.py Quantum computing 0.5 2`, will print the analysis of the graph acquired from `Quantum computing_0.5_2.gexf`
