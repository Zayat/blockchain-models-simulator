import colorsys
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import humanfriendly
import time
import seaborn as sns
import pandas as pd
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
#from fa2 import ForceAtlas2
#from graphviz import Graph
import pandas as pd

import string
#import pygraphviz

from bcns import Def_Hardness
from utils import ApplicationPaths, LoggerFactory

logger_simulator = LoggerFactory.get_logger("logger_simulator")


class Plots:

    @staticmethod
    def shorten_duration_string(str):
        s = str
        s = s.replace(" day", "d")
        s = s.replace(" hour", "h")
        s = s.replace(" week", "w")
        s = s.replace(" month", "m")
        s = s.replace(" year", "y")
        s = s.replace(" and", ",")
        s = s.replace("s", "")

        return s


    @staticmethod
    def lambda_vs_efficiency(data_df, duration, save=False, prefix=''):
        #plt.plot(latency_to_hardness, blocks_count)
        g = sns.lineplot(x="lambda", y="efficiency", data=data_df)
        g.set(xscale="log");
        plt.axhline(y=100, color='grey', linestyle='--', linewidth=1)
        plt.xlabel('Hardness to latency ($\lambda$)')
        # plt.ylim(0)
        plt.ylabel('Overall efficiency')
        #plt.title(
        #    'Simulated a duration of %s by 2 equally powerful miners' % (
        #        Plots.shorten_duration_string(humanfriendly.format_timespan(duration))))
        logger_simulator.info(
            'Simulated a duration of %s' % (humanfriendly.format_timespan(duration)))
        if save:
            plt.savefig(ApplicationPaths.evaluation_results() + prefix + "lambda_vs_efficiency" + time.strftime(
                '%Y_%m_%d-%H_%M_%S'))
            data_df.to_csv(ApplicationPaths.evaluation_results() + prefix + "lambda_vs_efficiency" + time.strftime(
                '%Y_%m_%d-%H_%M_%S')+".dataframe.csv")
        plt.show()

    @staticmethod
    def miners_vs_numblocks(miners_count, blocks_count, hardness, duration, save=False):
        expected_block_count = duration / hardness
        plt.plot(miners_count, blocks_count)
        plt.axhline(y=expected_block_count, color='r', linestyle='-')
        plt.xlabel('Number of miners')
        # plt.ylim(0)
        plt.ylabel('Number of blocks mined')
        plt.title('Simulated a duration of %s with hardness %d' % (humanfriendly.format_timespan(duration), hardness))
        logger_simulator.info(
            'Simulated a duration of %s with hardness %d' % (humanfriendly.format_timespan(duration), hardness))
        logger_simulator.info("Number of miners:")
        logger_simulator.info(miners_count)
        logger_simulator.info("Number of blocks mined")
        logger_simulator.info(blocks_count)
        if save:
            plt.savefig(
                ApplicationPaths.experiment_results() + "/miners_vs_numblocks_" + time.strftime('%Y_%m_%d-%H_%M_%S'))

        plt.show()

    @staticmethod
    def duration_vs_avg_block_time(durations, blocks_count, hardness, expected_block_time, save=False):
        durations_strings = [Plots.shorten_duration_string(humanfriendly.format_timespan(d)) for d in durations]
        plt.plot(durations_strings, blocks_count)
        plt.axhline(y=expected_block_time, color='r', linestyle='-')
        plt.xlabel('Duration')
        # plt.ylim(0)
        plt.ylabel('Average block mining time')
        plt.title('Simulated different durations with hardness %d for 2 miners' % hardness)
        logger_simulator.info('Simulated different durations with hardness %d' % hardness)
        logger_simulator.info("Number of miners:")
        logger_simulator.info(durations)
        logger_simulator.info("Number of blocks mined")
        logger_simulator.info(blocks_count)
        if save:
            plt.savefig(
                time.strftime(ApplicationPaths.experiment_results() + "%Y_%m_%d-%H_%M_%S") + "_miners_vs_numblocks.png")

        plt.show()

    @staticmethod
    def latency_to_hardness_vs_numblocks(latency_to_hardness, blocks_count, hardness, duration, save=False):
        expected_block_count = duration / hardness
        plt.plot(latency_to_hardness, blocks_count)
        plt.axhline(y=expected_block_count, color='r', linestyle='-')
        plt.xlabel('Latency to hardness')
        # plt.ylim(0)
        plt.ylabel('Number of blocks mined')
        plt.title(
            'Simulated a duration of %s with hardness %d by 2 miners' % (
                Plots.shorten_duration_string(humanfriendly.format_timespan(duration)), hardness))
        logger_simulator.info(
            'Simulated a duration of %s with hardness %d' % (humanfriendly.format_timespan(duration), hardness))
        logger_simulator.info("Latency to hardness:")
        logger_simulator.info(latency_to_hardness)
        logger_simulator.info("Number of blocks mined")
        logger_simulator.info(blocks_count)
        if save:
            plt.savefig(ApplicationPaths.experiment_results() + "latency_to_hardness_vs_numblocks_" + time.strftime(
                '%Y_%m_%d-%H_%M_%S'))
        plt.show()

    @staticmethod
    def hashingpower_vs_numblocks(hpd, blocks_count, hardness, duration, save=False):
        expected_block_count = duration / hardness
        str_hpd = ["%.2f %.2f" % (a, b) for [a, b] in hpd]
        plt.plot(str_hpd, [b for b, m1, m2 in blocks_count])
        plt.axhline(y=expected_block_count, color='r', linestyle='-')
        plt.xlabel('Hash power distribution')
        # plt.ylim(0)
        plt.ylabel('Number of blocks mined')
        plt.title(
            'Simulated a duration of %s with hardness %d by 2 miners' % (
                Plots.shorten_duration_string(humanfriendly.format_timespan(duration), hardness)))
        logger_simulator.info(
            'Simulated a duration of %s with hardness %d' % (humanfriendly.format_timespan(duration), hardness))
        logger_simulator.info("Hash power distribution:")
        logger_simulator.info(hpd)
        logger_simulator.info("Number of blocks mined")
        logger_simulator.info(blocks_count)
        if save:
            plt.savefig(ApplicationPaths.experiment_results() + "hashingpower_vs_numblocks_" + time.strftime(
                '%Y_%m_%d-%H_%M_%S'))
        plt.show()

    plt_idx = 1

    @staticmethod
    def plot_grid(data, x, y, col=None, hue=None, col_wrap=None, height=None,
                  palette="Set2", style="ticks", marker="o", save=False, filename=None):
        logger_simulator.info("START: plot_grid")
        sns.set(style=style)
        grid = sns.FacetGrid(data, col=col, hue=hue,
                             col_wrap=col_wrap, height=height, palette=palette)
        grid.map(sns.lineplot, x, y, marker=marker)
        plt.legend()
        if save:
            filename = ApplicationPaths.experiment_results() + filename # +"_"+ time.strftime(
                #'%Y-%m-%d--%H-%M-%S')
            plt.savefig(filename)
            logger_simulator.info("The plot has been persisted on : " + filename)
        else:
            plt.show()
        logger_simulator.info("Done: plot_grid")


    @staticmethod
    def hashingpower_vs_numblocks_bar(hpd, blocks_count, hardness, duration, latency_to_hardness, save=False):
        global plt_idx
        # plt.subplots(20)
        if not isinstance(latency_to_hardness, type([])):
            latency_to_hardness = [latency_to_hardness]
        expected_block_count = duration / hardness
        str_hpd = ["%.1f %.1f" % (a, b) for (a, b) in hpd]
        str_hpd_lat_hardness = [str_hpd[i] + " (lh " + str(latency_to_hardness[i%len(latency_to_hardness)]) + ")" for i in range(0,len(str_hpd))]
        series_labels = ['Miner 1', 'Miner 2']
        data = [
            [b for a, b, c, d, e in blocks_count],
            [c for a, b, c, d, e in blocks_count]
        ]
        data_eff = [
            ["bc %.2f, eff: %.2f" % (b / a * 1.0, d) for a, b, c, d, e in blocks_count],
            ["bc %.2f, eff: %.2f" % (c / a * 1.0, e) for a, b, c, d, e in blocks_count]
        ]

        # print(plt_idx)
        # plt.subplot(20,1,plt_idx)
        # plt_idx += 1
        plt.figure(figsize=(30, 15))
        Plots.stacked_bar(
            data,
            data_eff,
            series_labels,
            category_labels=str_hpd_lat_hardness,
            show_values=True,
            value_format="{:.1f}",
            y_label="Number of blocks mined",
            x_label="Hash power distribution",
            p_title='Simulated a duration of %s with hardness %d by 2 miners'
                    % (Plots.shorten_duration_string(humanfriendly.format_timespan(duration)), hardness)
        )

        plt.axhline(y=expected_block_count, color='r', linestyle='-')
        logger_simulator.info(
            'Simulated a duration of %s with hardness %d' % (humanfriendly.format_timespan(duration), hardness))
        logger_simulator.info("Hash power distribution:")
        logger_simulator.info(hpd)
        logger_simulator.info("Number of blocks mined")
        logger_simulator.info(blocks_count)
        if save:
            plt.savefig(
                ApplicationPaths.experiment_results() + "/hashingpower_vs_numblocks_bar_" + time.strftime(
                    '%Y_%m_%d-%H_%M_%S'))
        plt.show()

    @staticmethod
    def weighted_graph(nodes_ids, nodes_hp, nodes_share, edges_length_matrix, used_blocks, latency_to_hardness, duration, save=False):
        '''Turned out to be not that trivial!
        https://medium.com/@hilbert.cantor/network-plot-with-plotly-and-graphviz-ebd7778073b'''
        e = Graph('InfoGraphic', filename='infographic.gv', engine='neato', format='pdf')
        el = Graph('InfoGraphic Legends', filename='infographic_legend.gv', engine='neato', format='pdf')

        e.attr('node', shape='circle', style='filled')
        el.attr('node', shape='circle', style='filled')

        nnodes_share= [i/100.0 for i in nodes_share]
        cmap = mpl.cm.get_cmap('Greens')

        #node_width_factor = sum(edges_length_matrix[0])/len(edges_length_matrix[0])
        #network_edge_factor = sum(edges_length_matrix[0])/200
        #edge: len=str(edges_length_matrix[i][j]/network_edge_factor))
        #node: width=str(nodes_hp[i]*node_width_factor/network_edge_factor)

        #node_width_factor = (sum(edges_length_matrix[0]) / len(nodes_ids))
        #network_edge_factor = sum(edges_length_matrix[0]) / 200

        #node_width_factor = max(10,(sum(edges_length_matrix[0])/(len(nodes_ids))))
        #network_edge_factor = (len(nodes_ids)/sum(edges_length_matrix[0]))*10

        node_width_factor = sum(edges_length_matrix[0])/1000
        network_edge_factor = 1.25* np.min(edges_length_matrix[np.nonzero(edges_length_matrix)])

        for i in range(0,len(nodes_ids)):
            col=mpl.colors.rgb2hex(cmap(nnodes_share[i])[:3])
            textcolor= "white" if nnodes_share[i] > 0.5 else "black"

            e.node(str(nodes_ids[i]), width=str(1*nodes_hp[i]), fillcolor=col, fontcolor=textcolor, fontsize='1')
            if nodes_hp[i] != 0:
                optimal_eff=1.0*used_blocks[i]/((duration/Def_Hardness)*nodes_hp[i])
            else:
                optimal_eff=0
            el.node(str(nodes_ids[i]), label=str('Miner %d, HP: %.3f, \n Node Eff. %.3f, Opt. Eff %.3f' % (nodes_ids[i],nodes_hp[i],nnodes_share[i],optimal_eff)),
                                                 fillcolor=col, fontsize='30', fontcolor=textcolor)

        for i in range(0,len(edges_length_matrix)):
            for j in range(0,len(edges_length_matrix[i])):
                if i != j:
                    e.edge(str(i), str(j), len=str((edges_length_matrix[i][j]/network_edge_factor)), color="lightgrey", style='dotted')
        #e.attr(label=r'\n\n Infographic ' + str(", ".join(['%.3f' % i for i in nodes_hp])) +

        e.attr(label=r'\n\n Infographic ' +
                    str("\nLatency distr mean %.3f, duration %s" % (latency_to_hardness, (Plots.shorten_duration_string(humanfriendly.format_timespan(duration))))))
        e.attr(fontsize=str(30))
        e.attr(size='18,15')
        e.attr(outputorder="edgesfirst")
        el.attr(label=r'\n\n Infographic Legend ' +
                    str("\nLatency distr mean %.3f, duration %s" % (latency_to_hardness, (Plots.shorten_duration_string(humanfriendly.format_timespan(duration))))))
        el.attr(fontsize='30')
        el.attr(size='18,15')


        e.view()
        el.view()
        if save:
            e.render(ApplicationPaths.experiment_results() + "/infographic_" + time.strftime(
                    '%Y_%m_%d-%H_%M_%S') + "_L" + str(latency_to_hardness))
            el.render(ApplicationPaths.experiment_results() + "/infographic_legend_" + time.strftime(
                    '%Y_%m_%d-%H_%M_%S') + "_L" + str(latency_to_hardness))
            df = pd.DataFrame(edges_length_matrix)
            pd.DataFrame(df).to_csv(ApplicationPaths.experiment_results() + "latencies_" + time.strftime(
                    '%Y_%m_%d-%H_%M_%S') + "_L" + str(latency_to_hardness),sep='\t', header=False, float_format='%.2f')

    @staticmethod
    def infographic2(nodes_ids, nodes_hp, nodes_share, edges_length_matrix, save=False):
        nodes = range(len(nodes_ids))
        node_sizes = []
        labels = {}
        for n in nodes:
            node_sizes.append(nodes_hp[n]*1000)
            labels[n] = nodes_ids[n]

        # Node sizes: [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]

        # Connect each node to its successor
        #edges = [(i, i + 1) for i in range(len(nodes) - 1)]

        # Create the graph and draw it with the node labels
        g = nx.Graph()
        #g.add_nodes_from(nodes)
        for i in range(len(nodes_ids)):
            g.add_node(nodes_ids[i], node_size=nodes_hp[i] * 1000)
        #g.add_edges_from(edges)
        for i in range(len(edges_length_matrix)):
            for j in range(len(edges_length_matrix[i])):
                if i != j:
                    g.add_edge(i, j, weight=edges_length_matrix[i][j])

        #nx.draw_spring(g, node_size=node_sizes, labels=labels, with_labels=True)
        nx.draw('/tmp/out.png', format='png', prog='neato')
        #forceatlas2.forceatlas2_networkx_layout(g)
        plt.show()

    @staticmethod
    def infographic(nodes_ids, nodes_hp, nodes_share, edges_length_matrix, save=False):
        '''
        forceatlas2 = ForceAtlas2(
            # Behavior alternatives
            outboundAttractionDistribution=True,  # Dissuade hubs
            linLogMode=False,  # NOT IMPLEMENTED
            adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
            edgeWeightInfluence=1.0,

            # Performance
            jitterTolerance=1.0,  # Tolerance
            barnesHutOptimize=True,
            barnesHutTheta=1.2,
            multiThreaded=False,  # NOT IMPLEMENTED

            # Tuning
            scalingRatio=2.0,
            strongGravityMode=False,
            gravity=1.0,

            # Log
            verbose=True)
        '''

        tot_node_shares = sum(nodes_share)
        nnodes_share= [i*1.0/tot_node_shares for i in nodes_share]
        G = nx.Graph()
        for i in range(len(nodes_ids)):
            G.add_node(nodes_ids[i], node_size=nodes_hp[i]*100)
        for i in range(len(edges_length_matrix)):
            for j in range(len(edges_length_matrix[i])):
                if i != j:
                    G.add_edge(i, j, weight=edges_length_matrix[i][j])

        #G = nx.from_numpy_matrix(A)
        #G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())), string.ascii_uppercase)))

        G = nx.drawing.nx_agraph.to_agraph(G)

        G.node_attr.update(color="green", style="filled")
        G.edge_attr.update(color="blue", width="2.0")

        G.draw('/tmp/out.png', format='png', prog='neato')

        #pos = nx.layout.graphviz_layout(G, weight='latency',prog='dot')
        #pos = graphviz_layout(G, prog='dot')

        # pos = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=2000)
        # nx.draw(G, pos,with_labels=True)
        # nx.draw_networkx_edge_labels(G, pos)
        #
        # M = G.number_of_nodes()
        # node_colors = range(2, M + 2)
        # node_alphas = nnodes_share #[(5 + i) / (M + 4) for i in range(M)]
        #
        # nodes = nx.draw_networkx_nodes(G, pos, node_size=nodes_hp, node_color=node_alphas)
        # edges = nx.draw_networkx_edges(G, pos, node_size=nodes_hp, arrowstyle='->',
        #                                arrowsize=10, edge_color="red", width=2)
        # set alpha value for each edge
        #for i in range(G.number_of_nodes()):
        #    nodes[i].set_alpha(node_alphas[i])

        #pc = mpl.collections.PatchCollection(nodes, cmap=plt.cm.Blues)
        #pc.set_array(node_alphas)
        #plt.colorbar(pc)

        ax = plt.gca()
        ax.set_axis_off()
        plt.show()

        # plt.xlabel('Hash power distribution')
        # # plt.ylim(0)
        # plt.ylabel('Number of blocks mined')
        # plt.title(
        #     'Simulated a duration of %s with hardness %d by 2 miners' % (
        #         Plots.shorten_duration_string(humanfriendly.format_timespan(duration), hardness)))
        # logger_simulator.info(
        #     'Simulated a duration of %s with hardness %d' % (humanfriendly.format_timespan(duration), hardness))
        # logger_simulator.info("Hash power distribution:")
        # logger_simulator.info(hpd)
        # logger_simulator.info("Number of blocks mined")
        # logger_simulator.info(blocks_count)
        # if save:
        #     plt.savefig(ApplicationPaths.experiment_results() + "hashingpower_vs_numblocks_" + time.strftime(
        #         '%Y_%m_%d-%H_%M_%S'))
        # plt.show()

    @staticmethod
    def stacked_bar(data, text, series_labels, category_labels=None,
                    show_values=False, value_format="{}", y_label=None, x_label=None, p_title=None,
                    grid=True, reverse=False):
        """
        adapted from https://stackoverflow.com/questions/44309507/stacked-bar-plot-using-matplotlib
        Plots a stacked bar chart with the data and labels provided.

        Keyword arguments:
        data            -- 2-dimensional numpy array or nested list
                        containing data for each series in rows
        text            -- 2-dimensional numpy array or nested list
                        containing text for each series in rows
        series_labels   -- list of series labels (these appear in
                        the legend)
        category_labels -- list of category labels (these appear
                        on the x-axis)
        show_values     -- If True then numeric value labels will
                        be shown on each bar
        value_format    -- Format string for numeric value labels
                        (default is "{}")
        y_label         -- Label for y-axis (str)
        x_label         -- Label for x-axis (str)
        p_title         -- plot title
        grid            -- If True display grid
        reverse         -- If True reverse the order that the
                        series are displayed (left-to-right
                        or right-to-left)
        """

        ny = len(data[0])
        ind = list(range(ny))

        axes = []
        cum_size = np.zeros(ny)

        if reverse:
            data = np.flip(data, axis=1)
            category_labels = reversed(category_labels)
        print(series_labels)
        for i, row_data in enumerate(data):
            axes.append(plt.bar(ind, row_data, bottom=cum_size,
                                label=series_labels[i]))
            cum_size += row_data

        if category_labels:
            plt.xticks(ind, category_labels)

        if y_label:
            plt.ylabel(y_label)

        if x_label:
            plt.xlabel(x_label)

        if p_title:
            plt.xlabel(p_title)

        plt.legend()

        if grid:
            plt.grid()

        if show_values:
            i = 0
            j = 0
            for axis in axes:
                for bar in axis:
                    w, h = bar.get_width(), bar.get_height()

                    plt.text(bar.get_x() + w / 2, bar.get_y() + h / 2,
                             ((text[i][j])), ha="center",
                             va="center")
                    # plt.text(bar.get_x() + w / 2, bar.get_y() + h / 2,
                    #         value_format.format(h), ha="center",
                    #         va="center")
                    j = (j + 1) % len(data[i])
                i = (i + 1) % len(data)
