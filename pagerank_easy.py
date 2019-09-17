import networkx as nx

#create instance
graph = nx.DiGraph()

# add node
graph.add_node(1)
graph.add_node(2)
graph.add_node(3)
graph.add_node(4)
graph.add_node(5)
graph.add_node(6)

# add edge
graph.add_edge(1,2)
graph.add_edge(1,3)
graph.add_edge(1,4)
graph.add_edge(2,3)
graph.add_edge(3,4)
graph.add_edge(3,5)
graph.add_edge(2,6)
graph.add_edge(5,6)
graph.add_edge(1,6)

# calculate pagerank
pagerank = nx.pagerank(graph, alpha=0.85)
pagerank_numpy = nx.pagerank_numpy(graph, alpha=0.85)
pagerank_scipy = nx.pagerank_scipy(graph, alpha=0.85)

# print result
print("networkx, numpy, scipy")
print(pagerank, pagerank_numpy, pagerank_scipy)

# visualize network
import matplotlib.pyplot as plt
from matplotlib import animation
import random

graph = nx.Graph()


def get_figure(node_number):
    graph.add_node(node_number,
                   Position=(random.randrange(0,100),
                             random.randrange(0,100)))
    graph.add_edge(node_number, random.choice(graph.nodes()))
    nx.draw(graph, pos=nx.get_node_attributes(graph, 'Position'))


fig = plt.figure(figsize=(10, 8))

anim = animation.FuncAnimation(fig, get_figure, frames=100)
# anim.save('graph_gif_animation.html', writer='imagemagick', fps=10)
plt.show()