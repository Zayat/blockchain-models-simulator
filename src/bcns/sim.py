#!/usr/bin/env python
# -*- mode: python; coding: utf-8; fill-column: 80; -*-


import heapq
import sys
import time

import numpy

from bcns.events import Events as events
from utils import LoggerFactory

from .block import Block
from .chain import Chain
from .miner import Genesis_Miner_ID, Miner
from .stats import SimStats

logger_simulator = LoggerFactory.get_logger("logger_simulator")
logger_stats = LoggerFactory.get_logger("logger_stats")

# Pick a reasonable timer function.
_get_time = time.time

class Durations(object):
    """Constants for representing time durations."""
    MINUTE = 60
    HOUR = 60 * MINUTE
    DAY = 24 * HOUR
    WEEK = 7 * DAY
    MONTH = 30 * DAY
    YEAR = 365 * DAY


# Default mining hardness value.
Def_Hardness = 10 * Durations.MINUTE
Def_Latency = 1  # 1 second
Def_Pools = []

# Default distribution for assigning hashing power to miners.
Exp_HPD = lambda n: numpy.random.exponential(1.0, n)

# Equal distribution for assigning hashing power to miners.
Equ_HPD = lambda n: numpy.array([1.0] * n)

# Default distribution for assigning latencies between all pairs of miners.
# NOTE: Assuming links are symmetric.
Exp_LatD = lambda n, m, p: numpy.random.exponential(m, size=(n, n))

Equ_LatD = lambda n, m, p: numpy.full((n, n), m)

Equ_pooled_LatD = lambda n, m, p: pooled_latency_distribution(Equ_LatD(n, m, p), p, m)
Exp_pooled_LatD = lambda n, m, p: pooled_latency_distribution(Exp_LatD(n, m, p), p, m)


def pooled_latency_distribution(dist, pool_sizes, mean):
    assert (sum(pool_sizes) == len(dist))
    index = 0
    for p in pool_sizes:
        dist[index:index + p, index:index + p] = mean
        index = index + p
    return dist


class Configuration(object):
    """Configuration customizes the blockchain network simulator.

    Attributes:
        miners: A list of `Miner` instances.
        latencies: A table of latencies between all pair of miners.
        hardness: Mining hardness or difficulty parameter (i.e., rate parameter
            of the exponential distribution that governs the time it takes to
            solve the crypto puzzle)
    """

    def __init__(self, num_miners, hardness, hpd=Exp_HPD, latd=Exp_LatD, latm=Def_Latency, pools=[], miners_ids=[],
                 **kwargs):
        """Initializes a simulator configuration.

        Creates a simulator configuration specifying the hashing power of each
        miner and the network latency between each pair of miners. In generating
        the latencies, we assume that the miners are all connected together
        (i.e., they form a mesh).

        Args:
            num_miners: Number of miners to simulate.
            hardness: Mining hardness or difficulty parameter.
            hpd: A one-argument function that takes the number of miners and
                returns the distribution of hashing power among the miners.
            latd: A one-argument function that takes the number of miners and
                returns the latency distribution among the miners.
        """
        self.verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        self.quiet = kwargs['quiet'] if 'quiet' in kwargs else False
        self.hardness = hardness
        self._num_miners = num_miners
        self._hpd = hpd
        self._latd = latd
        self._latm = latm
        self._pools = pools
        self._hpowers = []
        self._gminer = None
        self._gblock = None
        self._miners_ids = miners_ids
        self.miners = []
        self.lat_mat = []

        self._gen_hashing_powers()
        self._make_genesis_miner()
        self._make_genesis_block()
        self._make_miners()
        self._make_latencies()

    def _gen_hashing_powers(self):
        if isinstance(self._hpd, type(())):
            hpowers = self._hpd
        else:
            hpowers = self._hpd(self._num_miners)
        total_power = sum(hpowers)
        self._hpowers = [i / total_power for i in hpowers]

    @property
    def hpowers(self):
        return self._hpowers

    def _make_genesis_miner(self):
        self._gminer = Miner(Genesis_Miner_ID, 0, None)

    def _make_genesis_block(self):
        self._gblock = Block(self._gminer, 0, 1, None)

    def _make_miners(self):

        if self.verbose:
            logger_simulator.info(self._gblock)

        if len(self._miners_ids) == 0:
            miners_ids = [i for i in range(0, self._num_miners)]

        self.miners = [Miner(miners_ids[i], self._hpowers[i],
                             Chain(self._gblock))
                       for i in range(0, self._num_miners)]

    def _make_latencies(self):
        # Asymmetric links
        # latencies = [[numpy.random.randint(100, 120000)/1000.0
        #               for i in range(0, num_miners)]
        #              for j in range(0,num_miners)]
        if not isinstance(self._latd, type(Exp_LatD)):
            self.lat_mat = self._latd
        else:
            lat_mat = self._latd(self._num_miners, self._latm, self._pools)
            self.lat_mat = (lat_mat + lat_mat.T) / 2
            self.lat_mat = self.lat_mat  # / 1000.0
            numpy.fill_diagonal(self.lat_mat, 0)


class Simulator(object):
    """Simplified blockchain network simulator."""

    def __init__(self, conf):
        self.AVG_BLOCK_TIME = conf.hardness
        self.miners = conf.miners
        self.latencies = conf.lat_mat
        self.specified_simulation_time = 0
        self.simulated_time = 0
        self.runtime = 0

        # Longest chain.
        self.main_chain = []

        self._verbose = conf.verbose
        self._quiet = conf.quiet

    def compute_longest_chain(self):
        """Compute longest chain of blocks mined."""
        max_len = 0
        for m in self.miners:
            if m.chain.blocks[-1].height > max_len:
                max_len = m.chain.blocks[-1].height
                self.main_chain = m.chain.blocks

    def show_stats(self, per_miner_stats=False):
        self.compute_longest_chain()

        logger_stats.info("Simulated %.2f hours of mining activity in %d ms" %
                          (self.simulated_time / Durations.HOUR, self.runtime))
        logger_stats.info("Average block mining time (or hardness): %d minutes" %
                          (self.AVG_BLOCK_TIME / Durations.MINUTE))

        global_stats = SimStats(self.AVG_BLOCK_TIME,
                                self.main_chain, self.miners)
        global_stats.print(per_miner_stats)

    def get_stats(self):
        self.compute_longest_chain()
        return SimStats(self.AVG_BLOCK_TIME,
                        self.main_chain, self.miners)

    def simulate(self, total_time, number_of_blocks_to_mine=-1):
        self.specified_simulation_time = total_time
        time_start = int(round(_get_time() * 1000))
        # The event priority queue, using a heap for efficiency
        # Format: (time_of_event, OBJECT)
        # OBJECT: (miner_to_handle_event, block_to_try_adding)
        time_events = []

        current_time = 0
        progress = 0
        prev_progress = 0

        _exp = numpy.random.exponential
        # Initial time events
        for m in self.miners:
            if m.hashing_power == 0:
                next_btime = sys.maxsize
            else:
                next_btime = current_time + _exp(self.AVG_BLOCK_TIME / m.hashing_power, 1)[0]
            m.next_block_time = next_btime
            block = Block(m, m.next_block_time, m.chain.blocks[-1].height + 1, m.chain.blocks[-1])
            # should be changed to have 1 active timer per miner [array of timers]!
            heapq.heappush(time_events, (m.next_block_time, (m, block)))

        # Simulation start, keep processing events untill time is up
        # Fetch first timer event
        current_time, (current_miner, miner_block) = heapq.heappop(time_events)
        bfound = 0
        while ((number_of_blocks_to_mine > 0 or total_time > current_time) and \
               (number_of_blocks_to_mine < 0 or number_of_blocks_to_mine > current_miner.chain.length)):
            # print(str(miner_block.hash))
            # TODO: Performance bottleneck at miner.add_block, miner._add_other_block and chain.replace_blocks
            r = current_miner.add_block(miner_block, current_time)
            #print((current_time, str(current_miner), r, str(miner_block.hash),str(miner_block.miner)))

            # if r == ANNOUNCE_NEW_BLOCK: The block has been mined successfully, and should be announced
            # if r == NEW_BLOCK_ADDED: The block has been received and successfully added to the chain of miner
            if r == events.ANNOUNCE_NEW_BLOCK:
                for m in self.miners:
                    if m.id == miner_block.miner.id:
                        continue
                    if self.latencies[current_miner.id][m.id] != float('inf'):
                        heapq.heappush(time_events,
                                    (current_time + self.latencies[current_miner.id][m.id],
                                        (m, miner_block)))
                # let's print some stats
                if 0:
                    e = self.get_stats()
                    miners_hpd = str(", ".join([str(m.hashing_power) for m in e.miners if m.hashing_power > 0]))
                    miners_eff = str(", ".join(
                        [str(100.0 * (m.num_used_blocks / max(1, (
                        (current_time / self.AVG_BLOCK_TIME)) * m.hashing_power)))
                         for m in e.miners if m.hashing_power > 0]))
                    print(str(bfound) + ", " + str(self.AVG_BLOCK_TIME) + ", " + str(current_time) + ", " + str(
                        e.efficiency) + ", " + miners_eff + ", " + miners_hpd)
                    bfound += 1


            if r == events.NEW_BLOCK_ADDED or r == events.ANNOUNCE_NEW_BLOCK:
                if current_miner.hashing_power != 0:
                    current_miner.next_block_time = current_time + \
                                                    numpy.random.exponential(
                                                        self.AVG_BLOCK_TIME / current_miner.hashing_power, 1)[0]
                    block = Block(current_miner, current_miner.next_block_time,
                                  current_miner.chain.blocks[-1].height + 1, current_miner.chain.blocks[-1])
                    #should be changed to have 1 active timer per miner [array of timers]!
                    current_time, (current_miner, miner_block) = \
                        heapq.heappushpop(time_events, (current_miner.next_block_time, (current_miner, block)))
            else:
                current_time, (current_miner, miner_block) = heapq.heappop(time_events)

            # if r == NEW_BLOCK_IGNORED:
            #    print "Same length"

            if not self._quiet:
                progress = ((current_time * 1.0) / total_time) * 100
                if progress > prev_progress + 10:
                    logger_simulator.info("#> ... ... ... ... ... %.2f%%" % progress)
                    prev_progress = progress

            # Fetch next timer event
            # print(self)
            self.simulated_time = current_time
            # TODO: Performance bottleneck at heapq.heappop
            #current_time, (current_miner, miner_block) = heapq.heappop(time_events)

        time_end = int(round(_get_time() * 1000))
        self.runtime = time_end - time_start

    def __str__(self):
        return "A total of " + str(sum([m.block_count for m in self.miners])) + " blocks were mined \n" \
               + "Miners:\n" + str("\n\n".join([str(m) for m in self.miners])) + "\n latencies: \n " + str(
            numpy.matrix(self.latencies))
