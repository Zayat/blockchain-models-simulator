#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : run_eval.py
# Author            : Mohamed Alzayat <alzayat@mpi-sws.org>
# Date              : 21.10.2020
# Last Modified Date: 21.10.2020
# Last Modified By  : Mohamed Alzayat <alzayat@mpi-sws.org>
import argparse

from bcns import sim
from evaluation import SimulationConfigs
from evaluation.SimulationConfigs import *
from evaluation.SimulationRunner import SimulationRunner
from evaluation.plotters import *
from utils import LoggerFactory
from utils import ApplicationPaths

logger_stats = LoggerFactory.get_logger("logger_stats")


def _config_arg_parser():
    parser = argparse.ArgumentParser(
        description="A simplified blockchain network simulator")

    parser.add_argument('-v', '--verbose', dest='verbose',
                        action='store_true',
                        default=True,
                        help='Enable verbose output')
    parser.add_argument('-q', '--quiet', dest='quiet',
                        default=False,
                        action='store_true',
                        help='Run simulator silently (disables progress meter)')
    parser.add_argument('-pt', '--plot_only', dest="plot_timestamp", default="",
                        help="Plot only for the data in the evaluation directory (identified by timestamp)")
    parser.add_argument('-c', '--config', dest="experiment_config", default="",
                        help="2: 2 p2p miners with varying lambda and hpd"
                             "2-c: 2 coordinated miners with varying lambda and hpd, specify the distance of the coordinator from M1 using -d "
                             "3: 3 p2p equidistant miners with varying lambda and hpd"
                             "3-c: 3 coordinated equidistant miners with varying hlambda and hpd, specify the distance of the coordinator from M3 using -d"
                             "5:  Real life p2p scenario based on top 5 mining pools"
                             "5e:  Real life p2p scenario based on top 5 ethereum mining pools"
                             "15:  Real life p2p scenario based on top 5 mining pools and 10 observers"
                        )
    parser.add_argument('-d', '--distance', dest="coordinator_distance", default=0,
                        help = "In the case of coordinated settings, please set the distance of the coordinator,\n "
                        "The distance is relative to M1 in case of 2 nodes, and relative to M3 in the case of 3 nodes #To be cleaned up")

    return parser

def default2_simulation(verbose=True, quiet=True, data=None, timestamp=None, np_lat_mat=None):
    logger_stats.info("Default 2 miners -- paper")
    if not data:
        config = Default2SimulationConfigs()
        sim_run = SimulationRunner(config, verbose=verbose, quiet=quiet)
        (data, timestamp, np_lat_mat) = sim_run.run()
        np_lat_mat = np.array(config.LATENCY_ADJACECY_MATRIX)


def default1f_coordinator_simulation(verbose=True, quiet=True, data=None, timestamp=None, np_lat_mat=None):
    logger_stats.info("Default 2 nodes and a coordinator, one of the nodes moving ")
    if not data:
        config = Default1fCoordinatorSimulationConfigs()
        sim_run = SimulationRunner(config, verbose=verbose, quiet=quiet)
        (data, timestamp, np_lat_mat) = sim_run.run()
        np_lat_mat = np.array(config.LATENCY_ADJACECY_MATRIX)

def default2_coordinator_simulation(verbose=True, quiet=True, data=None, timestamp=None, np_lat_mat=None):
    logger_stats.info("Default 2 nodes and a coordinator -- paper")
    if not data:
        config = Default2CoordinatorSimulationConfigs()
        sim_run = SimulationRunner(config, verbose=verbose, quiet=quiet)
        (data, timestamp, np_lat_mat) = sim_run.run()
        np_lat_mat = np.array(config.LATENCY_ADJACECY_MATRIX)


def default3_coordinator_simulation(verbose=True, quiet=True, data=None, timestamp=None, np_lat_mat=None):
    logger_stats.info("Default 3 nodes and a coordinators ")
    if not data:
        config = Default3CoordinatorSimulationConfigs()
        sim_run = SimulationRunner(config, verbose=verbose, quiet=quiet)
        (data, timestamp, np_lat_mat) = sim_run.run()
        np_lat_mat = np.array(config.LATENCY_ADJACECY_MATRIX)

def default4_p2p_simulation(verbose=True, quiet=True, data=None, timestamp=None, np_lat_mat=None):
    logger_stats.info("Default 4 p2p ")
    if not data:
        config = Default4P2PSimulationConfigs()
        sim_run = SimulationRunner(config, verbose=verbose, quiet=quiet)
        (data, timestamp, np_lat_mat) = sim_run.run()
        np_lat_mat = np.array(config.LATENCY_ADJACECY_MATRIX)

def default4_coordinator_simulation(verbose=True, quiet=True, data=None, timestamp=None, np_lat_mat=None):
    logger_stats.info("Default 4 nodes and a coordinator ")
    if not data:
        config = Default4CoordinatorSimulationConfigs()
        sim_run = SimulationRunner(config, verbose=verbose, quiet=quiet)
        (data, timestamp, np_lat_mat) = sim_run.run()
        np_lat_mat = np.array(config.LATENCY_ADJACECY_MATRIX)

def default3_simulation(verbose=True, quiet=True, data=None, timestamp=None, np_lat_mat=None):
    logger_stats.info("Default 3 p2p miners -- paper")
    if not data:
        config = Default3SimulationConfigs()
        sim_run = SimulationRunner(config, verbose=verbose, quiet=quiet)
        (data, timestamp, np_lat_mat) = sim_run.run()
        np_lat_mat = np.array(config.LATENCY_ADJACECY_MATRIX)

def default3f_simulation(verbose=True, quiet=True, data=None, timestamp=None, np_lat_mat=None):
    logger_stats.info("Default 3 p2p miners with one of the miners farther away")
    if not data:
        config = Default3fSimulationConfigs()
        sim_run = SimulationRunner(config, verbose=verbose, quiet=quiet)
        (data, timestamp, np_lat_mat) = sim_run.run()
        np_lat_mat = np.array(config.LATENCY_ADJACECY_MATRIX)


def default5_simulation(verbose=True, quiet=True, data=None, timestamp=None, np_lat_mat=None):
    logger_stats.info("Default 5 miners paper example")
    if not data:
        config = Default5SimulationConfigs()
        sim_run = SimulationRunner(config, verbose=verbose, quiet=quiet)
        (data, timestamp, np_lat_mat) = sim_run.run()
        np_lat_mat = np.array(config.LATENCY_ADJACECY_MATRIX)

def default5e_simulation(verbose=True, quiet=True, data=None, timestamp=None, np_lat_mat=None):
    logger_stats.info("Default 5 ethereum miners paper example")
    if not data:
        config = Default5EthereumSimulationConfigs()
        sim_run = SimulationRunner(config, verbose=verbose, quiet=quiet)
        (data, timestamp, np_lat_mat) = sim_run.run()
        np_lat_mat = np.array(config.LATENCY_ADJACECY_MATRIX)

def default15_simulation(verbose=True, quiet=True, data=None, timestamp=None, np_lat_mat=None):
    logger_stats.info("Default 15 miners paper example")
    if not data:
        config = Default15SimulationConfigs()
        sim_run = SimulationRunner(config, verbose=verbose, quiet=quiet)
        (data, timestamp, np_lat_mat) = sim_run.run()
        np_lat_mat = np.array(config.LATENCY_ADJACECY_MATRIX)


def default15ls_simulation(verbose=True, quiet=True, data=None, timestamp=None, np_lat_mat=None):
    logger_stats.info("Speed of light in fiber optics communication between 15 miners example")
    if not data:
        config = Default15LSSimulationConfigs()
        sim_run = SimulationRunner(config, verbose=verbose, quiet=quiet)
        (data, timestamp, np_lat_mat) = sim_run.run()
        np_lat_mat = np.array(config.LATENCY_ADJACECY_MATRIX)

def default15cs_simulation(verbose=True, quiet=True, data=None, timestamp=None, np_lat_mat=None):
    logger_stats.info("Speed of light communication between 15 miners example")
    if not data:
        config = Default15CSSimulationConfigs()
        sim_run = SimulationRunner(config, verbose=verbose, quiet=quiet)
        (data, timestamp, np_lat_mat) = sim_run.run()
        np_lat_mat = np.array(config.LATENCY_ADJACECY_MATRIX)


def default_attack_simulation(verbose=True, quiet=True, data=None, timestamp=None, np_lat_mat=None):
    logger_stats.info("default attackability simulation")
    if not data:
        config = DefaultAttackSimulationConfigs()
        sim_run = SimulationRunner(config, verbose=verbose, quiet=quiet)
        (data, timestamp, np_lat_mat) = sim_run.run()
        np_lat_mat = np.array(config.LATENCY_ADJACECY_MATRIX)


def default_centrality_simulation(verbose=True, quiet=True, data=None, timestamp=None, np_lat_mat=None):
    logger_stats.info("default_centrality_simulation")
    if not data:
        config = DefaultCentralitySimulationConfigs()
        sim_run = SimulationRunner(config, verbose=verbose, quiet=quiet)
        (data, timestamp, np_lat_mat) = sim_run.run()
        np_lat_mat = np.array(config.LATENCY_ADJACECY_MATRIX)


def default_centrality_capitals_simulation(verbose=True, quiet=True, data=None, timestamp=None, np_lat_mat=None):
    logger_stats.info("default_centrality_capitals_simulation")
    if not data:
        config = DefaultCapitalsCentralitySimulationConfigs()
        sim_run = SimulationRunner(config, verbose=verbose, quiet=quiet)
        (data, timestamp, np_lat_mat) = sim_run.run()
        np_lat_mat = np.array(config.LATENCY_ADJACECY_MATRIX)

def default_centrality_capitals_ls_simulation(verbose=True, quiet=True, data=None, timestamp=None, np_lat_mat=None):
    logger_stats.info("default_centrality_capitals_ls_simulation")
    if not data:
        config = DefaultCapitalsCentralityLSSimulationConfigs()
        sim_run = SimulationRunner(config, verbose=verbose, quiet=quiet)
        (data, timestamp, np_lat_mat) = sim_run.run()
        np_lat_mat = np.array(config.LATENCY_ADJACECY_MATRIX)

def default_centrality_capitals_cs_simulation(verbose=True, quiet=True, data=None, timestamp=None, np_lat_mat=None):
    logger_stats.info("default_centrality_capitals_cs_simulation")
    if not data:
        config = DefaultCapitalsCentralityCSSimulationConfigs()
        sim_run = SimulationRunner(config, verbose=verbose, quiet=quiet)
        (data, timestamp, np_lat_mat) = sim_run.run()
        np_lat_mat = np.array(config.LATENCY_ADJACECY_MATRIX)


def real_btc_exp_simulation(verbose=True, quiet=True, data=None, timestamp=None, np_lat_mat=None):
    logger_stats.info("real_btc_exp_simulation")
    if not data:
        config = RealLifeExpSimulationConfigs()
        sim_run = SimulationRunner(simulation_config=config, verbose=verbose, quiet=quiet)
        (data, timestamp, np_lat_mat) = sim_run.run()
        np_lat_mat = np.array(config.LATENCY_ADJACECY_MATRIX)


def real_btc_equ_simulation(verbose=True, quiet=True, data=None, timestamp=None, np_lat_mat=None):
    logger_stats.info("real_btc_equ_simulation")
    if not data:
        config = RealLifeEquSimulationConfigs()
        sim_run = SimulationRunner(simulation_config=config, verbose=verbose, quiet=quiet)
        (data, timestamp, np_lat_mat) = sim_run.run()
        np_lat_mat = np.array(config.LATENCY_ADJACECY_MATRIX)


def real_btc_pooled_exp_simulation(verbose=True, quiet=True, data=None, timestamp=None, np_lat_mat=None):
    logger_stats.info("real_btc_pooled_exp_simulation")
    if not data:
        config = RealLifeExpPooledSimulationConfigs()
        sim_run = SimulationRunner(simulation_config=config, verbose=verbose, quiet=quiet)
        (data, timestamp, np_lat_mat) = sim_run.run()
        np_lat_mat = np.array(config.LATENCY_ADJACECY_MATRIX)

def real_btc_pooled_equ_simulation(verbose=True, quiet=True, data=None, timestamp=None, np_lat_mat=None):
    logger_stats.info("real_btc_pooled_equ_simulation")
    if not data:
        config = RealLifeEquPooledSimulationConfigs()
        sim_run = SimulationRunner(simulation_config=config, verbose=verbose, quiet=quiet)
        (data, timestamp, np_lat_mat) = sim_run.run()
        np_lat_mat = np.array(config.LATENCY_ADJACECY_MATRIX)

if __name__ == '__main__':

    opts = _config_arg_parser().parse_args()
    ApplicationPaths.makedirs()
    data = None
    timestamp = None
    np_lat_mat = None

    if opts.plot_timestamp:
        timestamp = opts.plot_timestamp
        (data_global, data_miner, np_lat_mat) = get_eval_data(timestamp=opts.plot_timestamp)
        print(opts.plot_timestamp, opts.experiment_config)
        data = {'global': data_global, 'miner': data_miner}

    # print(np_lat_mat)
    # main(opts.verbose, opts.quiet)
    if opts.experiment_config == "tc":
        default_test_centrality_baseline_simulation(data=data, timestamp=timestamp, np_lat_mat=np_lat_mat)
    if opts.experiment_config == "2":
        default2_simulation(data=data, timestamp=timestamp, np_lat_mat=np_lat_mat)
    elif opts.experiment_config == "3":
        default3_simulation(data=data, timestamp=timestamp, np_lat_mat=np_lat_mat)
    elif opts.experiment_config == "rx":
        real_btc_exp_simulation(data=data, timestamp=timestamp, np_lat_mat=np_lat_mat)
    elif opts.experiment_config == "re":
        real_btc_equ_simulation(data=data, timestamp=timestamp, np_lat_mat=np_lat_mat)
    elif opts.experiment_config == "rpx":
        real_btc_pooled_exp_simulation(data=data, timestamp=timestamp, np_lat_mat=np_lat_mat)
    elif opts.experiment_config == "rpe":
        real_btc_pooled_equ_simulation(data=data, timestamp=timestamp, np_lat_mat=np_lat_mat)
    elif opts.experiment_config == "5":
        default5_simulation(data=data, timestamp=timestamp, np_lat_mat=np_lat_mat)
    elif opts.experiment_config == "5e":
        default5e_simulation(data=data, timestamp=timestamp, np_lat_mat=np_lat_mat)

    elif opts.experiment_config == "15":
        default15_simulation(data=data, timestamp=timestamp, np_lat_mat=np_lat_mat, verbose=False)
    elif opts.experiment_config == "15ls":
        default15ls_simulation(data=data, timestamp=timestamp, np_lat_mat=np_lat_mat)
    elif opts.experiment_config == "15cs":
        default15cs_simulation(data=data, timestamp=timestamp, np_lat_mat=np_lat_mat)
    elif opts.experiment_config == "attack":
        default_attack_simulation(data=data, timestamp=timestamp, np_lat_mat=np_lat_mat)
    elif opts.experiment_config == "centrality":
        default_centrality_simulation(data=data, timestamp=timestamp, np_lat_mat=np_lat_mat)
    elif opts.experiment_config == "capitals-centrality":
        default_centrality_capitals_simulation(data=data, timestamp=timestamp, np_lat_mat=np_lat_mat)
    elif opts.experiment_config == "capitals-centrality-ls":
        default_centrality_capitals_ls_simulation(data=data, timestamp=timestamp, np_lat_mat=np_lat_mat)
    elif opts.experiment_config == "capitals-centrality-cs":
        default_centrality_capitals_cs_simulation(data=data, timestamp=timestamp, np_lat_mat=np_lat_mat)
    elif opts.experiment_config == "c-1":
        COORDINATOR_DISTANCE(int(opts.coordinator_distance))
        default1_coordinator_simulation(data=data, timestamp=timestamp, np_lat_mat=np_lat_mat)
    elif opts.experiment_config == "c-1f":
        COORDINATOR_DISTANCE(int(opts.coordinator_distance))
        default1f_coordinator_simulation(data=data, timestamp=timestamp, np_lat_mat=np_lat_mat)
    elif opts.experiment_config == "2-c":
        COORDINATOR_DISTANCE(int(opts.coordinator_distance))
        default2_coordinator_simulation(data=data, timestamp=timestamp, np_lat_mat=np_lat_mat)
    elif opts.experiment_config == "coordinator2-equidistant":
        COORDINATOR_DISTANCE(int(opts.coordinator_distance))
        default2_coordinator_equidistant_simulation(data=data, timestamp=timestamp, np_lat_mat=np_lat_mat)
    elif opts.experiment_config == "coordinator2-farther":
        COORDINATOR_DISTANCE(int(opts.coordinator_distance))
        default2_coordinator_farther_simulation(data=data, timestamp=timestamp, np_lat_mat=np_lat_mat)
    elif opts.experiment_config == "3-c":
        COORDINATOR_DISTANCE(int(opts.coordinator_distance))
        default3_coordinator_simulation(data=data, timestamp=timestamp, np_lat_mat=np_lat_mat)
    elif opts.experiment_config == "4":
        COORDINATOR_DISTANCE(int(opts.coordinator_distance))
        default4_p2p_simulation(data=data, timestamp=timestamp, np_lat_mat=np_lat_mat)
    elif opts.experiment_config == "4-c":
        COORDINATOR_DISTANCE(int(opts.coordinator_distance))
        default4_coordinator_simulation(data=data, timestamp=timestamp, np_lat_mat=np_lat_mat)



    # visualise_setting_real(data_global, data_miner, timestamp=opts.plot_timestamp,
    #                       simulation_config=RealLifeSimulationConfigs)
