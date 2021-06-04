#!/usr/bin/env python
# -*- mode: python; coding: utf-8; fill-column: 80; -*-


import heapq
import sys
import time

import numpy

from bcns import Simulator
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

PAUSE_UNTILL_ACK = 1


class SimulatorCoordinated(Simulator):
    """Simplified blockchain network simulator for coordinated setting.
        TODO: Add support for observers?
        """

    def __init__(self, conf):
        super().__init__(conf)

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
                next_btime = current_time + \
                    _exp(self.AVG_BLOCK_TIME / m.hashing_power, 1)[0]
            m.next_block_time = next_btime
            block = Block(m, m.next_block_time,
                          m.chain.blocks[-1].height + 1, m.chain.blocks[-1])
            m.next_legit_hash = block.hash
            # should be changed to have 1 active timer per miner [array of timers]!
            heapq.heappush(time_events, (m.next_block_time, (m, block)))

        # Simulation start, keep processing events untill time is up
        # Fetch first timer event
        current_time, (current_miner, miner_block) = heapq.heappop(time_events)

        bfound = 0
        while ((number_of_blocks_to_mine > 0 or total_time > current_time) and
               (number_of_blocks_to_mine < 0 or number_of_blocks_to_mine > current_miner.chain.length)):
            #print(str(miner_block.hash))
            # TODO: Performance bottleneck at miner.add_block, miner._add_other_block and chain.replace_blocks
            r = current_miner.add_block(miner_block, current_time, coordinated=True)
            #print((current_time, str(current_miner), r, str(miner_block.hash),str(miner_block.miner)))

            # if r == ANNOUNCE_NEW_BLOCK: The block has been mined successfully, and should be announced
            # if r == NEW_BLOCK_ADDED: The block has been received and successfully added to the chain of miner
            if r == events.ANNOUNCE_NEW_BLOCK:
                try:

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

                except IndexError as e:
                    print(self.miners)
                    print(self.latencies)
                    print(current_miner)
                    print(m)

            if r == events.NEW_BLOCK_ADDED:
                for m in self.miners:
                    #if m.id == miner_block.miner.id or m.id == current_miner.id or m.hashing_power == 0:  # A miner do not send the block to original miner or coordinator
                    if m.id == current_miner.id or m.hashing_power == 0:  # A miner do not send the block to original miner or coordinator
                        continue
                    if self.latencies[current_miner.id][m.id] != float('inf'):
                        heapq.heappush(time_events,
                                       (current_time + self.latencies[current_miner.id][m.id],
                                              (m, miner_block)))

            # Disable sampling until a block is added (to simulate pausing after finding a block till we ACK)
            if not PAUSE_UNTILL_ACK or r == events.NEW_BLOCK_ADDED:
                if r == events.NEW_BLOCK_ADDED or r == events.ANNOUNCE_NEW_BLOCK or r == events.NEW_BLOCK_WASTED:
                #if r is not events.NEW_BLOCK_IGNORED:
                    if current_miner.hashing_power != 0:
                        current_miner.next_block_time = current_time + \
                            numpy.random.exponential(
                                self.AVG_BLOCK_TIME / current_miner.hashing_power, 1)[0]
                        block = Block(current_miner, current_miner.next_block_time,
                                      current_miner.chain.blocks[-1].height + 1, current_miner.chain.blocks[-1])
                        #should be changed to have 1 active timer per miner [array of timers]!
                        heapq.heappush(
                            time_events, (current_miner.next_block_time, (current_miner, block)))
                        current_miner.next_legit_hash = block.hash
                        #print("Next try: ", (current_miner.next_block_time, str(current_miner), str(block.hash), str(block.miner)))

            # if r == NEW_BLOCK_IGNORED:
            #    print "Same length"

            if not self._quiet:
                progress = ((current_time * 1.0) / total_time) * 100
                if progress > prev_progress + 10:
                    logger_simulator.info(
                        "#> ... ... ... ... ... %.2f%%" % progress)
                    prev_progress = progress

            # Fetch next timer event
            # print(self)
            self.simulated_time = current_time
            # TODO: Performance bottleneck at heapq.heappop
            current_time, (current_miner, miner_block) = heapq.heappop(time_events)

        time_end = int(round(_get_time() * 1000))
        self.runtime = time_end - time_start
