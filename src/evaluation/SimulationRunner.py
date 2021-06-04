#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : SimulationRunner.py
# Author            : Mohamed Alzayat <alzayat@mpi-sws.org>
# Date              : 21.10.2020
# Last Modified Date: 21.10.2020
# Last Modified By  : Mohamed Alzayat <alzayat@mpi-sws.org>
import time
from itertools import product
from multiprocessing import Pool
from os import cpu_count

import numpy as np
from bcns import Configuration, Simulator, SimulatorCoordinated
from tqdm import tqdm
from utils import ApplicationPaths

from evaluation.plotters import persist_dataframe, persist_lat_matrix
from evaluation.SimulationConfigs import DefaultSimulationConfigs as DCFG


class SimulationRunner:

    def __init__(self, simulation_config=DCFG, verbose=True, quiet=True):

        np.random.seed(simulation_config.SEED)
        self.duration = simulation_config.DURATION
        self.num_blocks_to_generate = simulation_config.NUM_BLOCKS_TO_GENERATE
        self.num_miners = simulation_config.NUM_MINERS
        self.seed = simulation_config.SEED
        self.mean_network_latency_s = simulation_config.MEAN_NETWORK_LATENCY_S
        self.num_iter = simulation_config.NUM_ITER
        self.nodes_ids = simulation_config.NODES_IDS
        self.hpd_cfg = simulation_config.HPD_CFG
        self.hardness_cfg = simulation_config.HARDNESS_CFG
        self.lat_mat = simulation_config.LATENCY_ADJACECY_MATRIX
        self.persist_data = simulation_config.PERSIST_DATA
        self.experiment_identifier = simulation_config.EXPERIMENT_IDENTIFIER
        self.to_dataframe = simulation_config.TO_DATAFRAME
        self.simulator = simulation_config.SIMULATOR
        self.verbose = verbose
        self.quiet = quiet
        self.n_processes = cpu_count() - 1

    def run(self):
        experiments_stats = list()
        lat_mat = self.lat_mat
        print(np.array(lat_mat))
        hpd_hardness_list = list(
            product(self.hpd_cfg, self.hardness_cfg, range(0, self.num_iter)))
        #pbar = tqdm(desc='Running experiments', total=len(hpd_hardness_list), ascii=False)
        with Pool(self.n_processes) as p:
            experiments_stats = p.starmap(self.run_simulations, hpd_hardness_list)

        '''
        for hpd, hardness, iteration in hpd_hardness_list:
            print(f"Hardness ({hpd})")
            print(f"Iterations ({hpd}), ({hardness})")
            result = self.run_simulations(
                hpd=hpd, hardness=hardness, iteration=iteration)
            experiments_stats.append(result['data_info'])
        '''

        data_df = self.to_dataframe(
            experiments_stats, self.nodes_ids, nodes_count=self.num_miners)
        timestamp = time.strftime('%Y_%m_%d-%H_%M_%S')
        np_lat_mat = np.array(lat_mat)
        #print(data_df['miner'][['id', 'effective_hpd', 'hpd', 'iteration']].sort_values(['iteration','effective_hpd'], ascending=False))
        if self.persist_data:
            persist_dataframe(data_df['miner'],
                              self.experiment_identifier + "_miner_" + str(self.num_blocks_to_generate), timestamp)
            persist_dataframe(data_df['global'],
                              self.experiment_identifier + "_global_" + str(self.num_blocks_to_generate), timestamp)
            persist_lat_matrix(np.array(lat_mat),
                               self.experiment_identifier + "_lat_matrix_" + str(self.num_blocks_to_generate), timestamp)

            data_df['miner'][['id', 'effective_hpd', 'hpd', 'iteration']].sort_values(['iteration', 'effective_hpd'],
                                                                                      ascending=False).to_csv(ApplicationPaths.evaluation_results()+timestamp+"/" + 'sorted_effective_hpd_run.csv')

            data_df['miner'][['id', 'effective_hpd', 'hpd', 'iteration']].sort_values(['iteration', 'effective_hpd'],
                                                                                      ascending=False).groupby('id').agg('mean').reset_index().to_csv(ApplicationPaths.evaluation_results()+timestamp+"/" + 'grouped_sorted_effective_hpd_run.csv')

        return (data_df, timestamp, np_lat_mat)

    def run_simulations(self, hpd, hardness, iteration):
        print(f"Hardness ({hpd})")
        print(f"Iterations ({hpd}), ({hardness})")

        np.random.seed(iteration)
        conf = Configuration(self.num_miners, hardness, hpd=hpd, latd=self.lat_mat,
                             latm=self.mean_network_latency_s, verbose=self.verbose,
                             quiet=self.quiet)
        s = self.simulator(conf)
        #s = Simulator(conf)
        s.simulate(self.duration, self.num_blocks_to_generate)
        stats = s.get_stats()
        latency_to_hardness = self.mean_network_latency_s / hardness
        data_info = dict()
        data_info['chain_lenght'] = stats.chain_length
        data_info['efficiency'] = stats.efficiency
        data_info['latency_mean'] = self.mean_network_latency_s
        data_info['hardness'] = hardness
        data_info['latency_to_hardness'] = latency_to_hardness
        data_info['hlambda'] = hardness / self.mean_network_latency_s
        data_info['num_blocks'] = stats.num_blocks
        data_info['agg_fork_time'] = stats.agg_fork_time
        data_info['agg_delay'] = stats.agg_delay
        data_info['num_orphans'] = stats.num_orphans
        data_info['num_miners'] = stats.num_miners
        data_info['agg_stale_delay'] = stats.agg_stale_delay
        data_info['max_orphans'] = stats.max_orphans
        data_info['switches'] = stats.switches
        data_info['hpd'] = hpd
        data_info['miners'] = list()
        data_info['iteration'] = iteration
        data_info['simulated_time'] = s.simulated_time
        data_info['efficiency_optimal'] = 100.0 * (data_info['chain_lenght'] / (
            data_info['simulated_time'] / data_info['hardness']))
        for miner in stats.miners:
            data_info['miners'].append(
                {'id': miner.id,
                    'num_used_blocks': miner.num_used_blocks,
                    'efficiency': miner.efficiency,
                    'num_blocks': miner.num_blocks,
                    'hpd': miner.hashing_power,
                    'effective_hpd': miner.num_used_blocks/stats.chain_length,
                    'num_forks': miner.orphan_stats.count,
                    'fork_rate': (miner.orphan_stats.count / stats.chain_length) * 100,
                    'num_orphaned_blocks': miner.orphan_stats.csum,
                    'num_own_orphaned_blocks': miner.own_orphan_stats.csum,
                    'num_max_orphans_per_fork': miner.orphan_stats.max,
                    'num_avg_orphans_per_fork': miner.orphan_stats.mean,
                    'total_time_wasted': miner.wasted_time.csum,
                    'time_wasted_stale_delay': miner.stale_wasted_time.csum,
                    'orphaned_blocks': str([len(x) for x in miner.orphaned_blocks]),
                    'latency_to_hardness': latency_to_hardness,
                    'hlambda': hardness / self.mean_network_latency_s,
                    'latency_mean': self.mean_network_latency_s,
                    'hardness': hardness,
                    'global_hpd': hpd,
                    'chain_length': stats.chain_length,
                    'iteration': iteration,
                    'simulated_time': s.simulated_time,
                    'efficiency_optimal': 100.0 * (
                        miner.num_used_blocks / max(1, ((s.simulated_time / hardness)) * miner.hashing_power))
                 })
        if conf.verbose:
            s.show_stats(per_miner_stats=True)
        return data_info
