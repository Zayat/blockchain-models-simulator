#!/usr/bin/env python
# -*- mode: python; coding: utf-8; fill-column: 80; -*-


import matplotlib.pyplot as plt
import networkx as nx


class Visualizer(object):
    """Visualizes the blockchain created by the miners in the simulation."""

    def __init__(self):
        self._g = nx.DiGraph()

    def run(self, all_blocks, chain):
        # G.add_edges_from([(str(b.hash), str(b.previous.hash)) for b in MAIN_CHAIN_HASHES.blocks[1:]])
        for block in all_blocks:
            if block in chain.blocks:
                self.g.add_edge(str(block.hash) + ":" + str(block.miner.id),
                                str(block.previous.hash) + ":" + str(
                                    block.previous.miner.id), weight=0.7)
            else:
                self.g.add_edge(str(block.hash) + ":" + str(block.miner.id),
                                str(block.previous.hash) + ":" + str(
                                    block.previous.miner.id), weight=0.1)

        pos = nx.spring_layout(self._g)

        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=900, node_color='g', alpha=0.5)

        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for (u, v, d) in
                                                 G.edges(data=True) if
                                                 d['weight'] >= 0.5],
                               edge_color='r', arrowsize=20, arrows=True)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for (u, v, d) in
                                                 G.edges(data=True) if
                                                 d['weight'] < 0.5],
                               edge_color='b', arrowsize=20, arrows=True)

        plt.axis('off')
        plt.show()
