#!/usr/bin/env python
# -*- mode: python; coding: utf-8; fill-column: 80; -*-

from utils import LoggerFactory

logger_stats = LoggerFactory.get_logger("logger_stats")
logger_simulator = LoggerFactory.get_logger("logger_simulator")


class Tracker(object):
    """Tracks the changes to a variable and maintains a few basic statistics.

    Maintains how often the variable changed its value, the maximum and minimum
    values it received, the cumulative sum, and the mean of the variations.

    Attributes:
        count: Number of times the tracker was updated.
        min: Minimum value received.
        max: Maximum value received.
        mean: Average of the values received.
        csum: Cumulative sum of values received.
    """
    __slots__ = (
        '_count',
        '_min',
        '_max',
        '_csum',
        '_mean',
    )

    def __init__(self):
        self._count = 0
        self._min = float('inf')
        self._max = 0
        self._csum = 0
        self._mean = 0

    def update(self, val):
        """Update the tracker with a new value."""
        self._count += 1
        self._min = val if val < self._min else self._min
        self._max = val if val > self._max else self._max
        self._csum += val
        self._mean = self._csum / self._count

    @property
    def count(self):
        return self._count

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def csum(self):
        return self._csum

    @property
    def mean(self):
        return self._mean


class MinerStats(object):
    """Simple set of statistics pertaining to a miner's activity."""

    def __init__(self, miner, hash_chain):
        self._id = miner.id
        self._hashing_power = miner.hashing_power
        self._num_blocks = miner.block_count
        self._orphan_stats = miner.orphan_stats
        self._own_orphan_stats = miner.own_orphan_stats
        self._orphaned_blocks = miner.orphaned_blocks
        self._wasted_time = miner.wasted_time
        self._stale_wasted_time = miner.stale_wasted_time
        self._short_str = miner.short_str()
        self._hash_chain = hash_chain
        self._num_used_blocks = len([b.hash for b in miner.generated_blocks
                                     if b.hash in hash_chain])
        self._eff = 100.0 * self._num_used_blocks / max(1, self._num_blocks)


    @property
    def hashing_power(self):
        return self._hashing_power
    @property
    def id(self):
        return self._id

    @property
    def info(self):
        return self._short_str

    @property
    def num_blocks(self):
        return self._num_blocks

    @property
    def num_used_blocks(self):
        return self._num_used_blocks

    @property
    def wasted_time(self):
        return self._wasted_time
    @property
    def stale_wasted_time(self):
        return self._stale_wasted_time

    @property
    def orphaned_blocks(self):
        return self._orphaned_blocks

    @property
    def orphan_stats(self):
        return self._orphan_stats

    @property
    def own_orphan_stats(self):
        return self._own_orphan_stats

    @property
    def efficiency(self):
        return self._eff

    @property
    def hash_chain(self):
        return self._hash_chain

    def print(self):
        stats_str = "\n" + self.info + "\n"
        stats_str += "\t                            #blocks mined: %d\n" % self.num_blocks
        stats_str += "\t                 #blocks in (local) chain: %d\n" % self.num_used_blocks
        stats_str += "\t                       #forks encountered: %d\n" % self.orphan_stats.count
        stats_str += "\t                         #orphaned-blocks: %d\n" % self.orphan_stats.csum
        stats_str += "\t                   Max. #orphans per fork: %d\n" % self.orphan_stats.max
        stats_str += "\t     Time wasted due to network delays: %d ms\n" % self.wasted_time.csum
        stats_str += "\tTime wasted due to mining stale blocks: %d ms\n" % self.stale_wasted_time.csum
        stats_str += "\t           #used-blocks/#blocks mined: %.1f%%\n" % self.efficiency
        stats_str += "\t           #used-blocks/Total #blocks: %.1f%%\n" % (100.0*self.num_used_blocks/len(self.hash_chain))

        logger_simulator.info(stats_str)


class SimStats(object):
    """Simple set of statistics pertaining to the overall blockchain simulation.

    Attributes:
        chain_length: Length of the main blockchain (agreed upon by majority)
        num_miners: Number of miners in the simulation
        num_blocks: Aggregate number of blocks mined by all miners
        switches: Number of times the main chain witnessed a change (or fork)
        num_orphans: Aggregate number of orphan blocks across all miners
        max_orphans: Max. number of orphans mined by any miner
        miners_with_max_orphans: Miner(s) with max. number of orphans
        efficiency: ratio of chain length to number of blocks mined
        agg_delay: Time wasted due to network delays
        agg_fork_time: Time wasted due to forks
        worst_fork_time: Time wasted for worst forks (with max. orphans)
    """
    def __init__(self, hardness, chain, miners):
        self._hardness = hardness
        self._chain = chain
        self._hchain = {b.hash for b in self._chain}
        self._miners = [MinerStats(m, self._hchain) for m in miners]

        self._num_blocks = 0
        self._switches = 0
        self._num_orphans = 0
        self._max_orphans = 0
        self._miners_w_max_orphans = tuple()

        self._update()

    def _update(self):
        # 1 (genesis block) + blocks mined by each miner
        self._num_blocks = 1 + sum((m.num_blocks for m in self._miners))

        # Number of times main chain has changed is equal to the sum of the
        # number of times each miner had to orphan one or more blocks.
        self._switches = sum([m.orphan_stats.count for m in self._miners])

        self._num_orphans = sum([m.orphan_stats.csum for m in self._miners])
        self._max_orphans = max([m.orphan_stats.max for m in self._miners])
        self._miners_w_max_orphans = \
            tuple([m for m in self._miners
                   if m.orphan_stats.max == self._max_orphans])

        self._eff = 100.0 * self.chain_length / self.num_blocks
        self._agg_delay = sum([m.wasted_time.csum for m in self._miners])
        self._agg_stale_delay = sum([m.stale_wasted_time.csum for m in self._miners])
        self._agg_fork_time = self._hardness * self._num_orphans
        # TODO: Clarify what this metric quantifies.
        self._worst_fork_time = self._hardness * self.max_orphans

    @property
    def chain_length(self):
        return len(self._chain)

    @property
    def num_miners(self):
        return len(self._miners)

    @property
    def num_blocks(self):
        return self._num_blocks

    @property
    def switches(self):
        return self._switches

    @property
    def num_orphans(self):
        return self._num_orphans

    @property
    def max_orphans(self):
        return self._max_orphans

    @property
    def miners_with_max_orphans(self):
        return self._miners_w_max_orphans

    @property
    def efficiency(self):
        return self._eff

    @property
    def agg_delay(self):
        return self._agg_delay

    @property
    def agg_stale_delay(self):
        return self._agg_stale_delay

    @property
    def agg_fork_time(self):
        return self._agg_fork_time

    @property
    def worst_fork_time(self):
        return self._worst_fork_time

    @property
    def miners(self):
        return self._miners

    def print(self, per_miner_stats=False):
        stats_str = "\n===== Global Statistics =====\n"
        stats_str += "\n===== Global Statistics =====\n"
        stats_str += ">>> #blocks in longest chain: %d\n" % self.chain_length
        stats_str += "\t>>> #blocks mined: %d\n" % self.num_blocks
        stats_str += "\t>>> #miners: %d\n" % self.num_miners
        stats_str += "\t>>> #times miners encountered forks: %d\n" % self.switches
        stats_str += "\t>>> #orphaned-blocks: %d\n" % self.num_orphans
        stats_str += "\t>>> Max. #blocks orphaned by miners: %d\n" % self.max_orphans
        if self.max_orphans:
            stats_str += "\t>>> Miners with max. #orphan-blocks: %s\n" %\
                         ', '.join(str(m.id) for m in self.miners_with_max_orphans)
        # TODO: Compare chain length to the theoretical upper bound.
        stats_str += ">>> Longest chain length/#blocks mined: %.1f%%\n" % self.efficiency
        stats_str += ">>> Time wasted due to network delays: %d ms\n" % self.agg_delay
        stats_str += "\tTime wasted due to mining stale blocks: %d ms\n" % self.agg_stale_delay

        stats_str += "\tTime wasted in orphaned blocks: %d ms\n" % self.agg_fork_time
        stats_str += "\tTime wasted in worst case fork: %d ms\n" % self.worst_fork_time
        logger_simulator.info(stats_str)
        if not per_miner_stats:
            return

        stats_str += "\n===== Per-Miner Statistics =====\n"

        logger_stats.info(stats_str)
        for m in self._miners:
            m.print()
