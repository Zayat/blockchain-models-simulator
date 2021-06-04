#!/usr/bin/env python
# -*- mode: python; coding: utf-8; fill-column: 80; -*-


import numpy

from .miner import Genesis_Miner_ID


block_hash_ids = 0

def get_next_block_hash_id():
    global block_hash_ids
    block_hash_ids += 1
    return block_hash_ids

class Block(object):
    __slots__ = (
        'hash',        # Block hash (It is required to be unique)
        'miner',       # Miner who mined the block.
        'timestamp',   # Timestamp at which block was mined.
        'height',      # Height of the chain at
        'previous',    # Pointer to the parent or previous block.
        'acks',        # Acknowledgements, if any.
    )

    def __init__(self, miner, timestamp, height, previous, acks=None):
        self.hash = get_next_block_hash_id() #int(numpy.random.random() * 10000000000000)  # replace with hash?
        self.miner = miner
        self.timestamp = timestamp
        self.height = height
        self.previous = previous
        self.acks = acks

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return self.hash == other.hash

    def __lt__(self, other):
        return self.hash < other.hash

    def __le__(self, other):
        return self.hash <= other.hash

    def _is_genesis(self):
        return (not self.previous and
                self.miner.id == Genesis_Miner_ID)

    def __str__(self):
        if self._is_genesis():
            return ("Genesis-Block{id: %d, ts: %d, ht: %d}" %
                    (self.hash, self.timestamp, self.height))

        return ("Block{id: %d, miner: %d, ts: %d, ht: %d, prev: %d}" %
                (self.hash, self.miner.id, self.timestamp, self.height,
                 self.previous.hash))
