import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

G = nx.DiGraph()
G.add_edge("T", "A")
G.add_edge("T", "B")
G.add_edge("A", "AA")
G.add_edge("A", "AB")
G.add_edge("B", "BA")
G.add_edge("B", "BB")

pos = {
    "T": (4, 1), 
    "B": (6, 0.5),
    "BA": (5, 0),
    "BB": (7, 0),
    "A": (2, 0.5), 
    "AA": (1, 0),
    "AB": (3, 0),
}
options = {
    "node_color": "white",
    "edgecolors": "black",
    "linewidths": 0,
    "width": 2,
}

_ = plt.figure(figsize=(6, 3))
nx.draw_networkx(G, pos=pos, **options)
plt.axis("off")
plt.savefig('./Plot/Hierarchy.png')
plt.show()