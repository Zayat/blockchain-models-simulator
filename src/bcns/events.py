#!/usr/bin/env python
# -*- mode: python; coding: utf-8; fill-column: 80; -*-

from enum import Enum

# Status codes resulting from trying to add a block to a miner chain.


class Events(Enum):
    # Block has been successfully mined and should be announced.
    ANNOUNCE_NEW_BLOCK = 1
    # Block has been received by a miner and successfully added to his/her chain.
    NEW_BLOCK_ADDED = 2
    # Block has been received by a miner, but not added to his/her chain.
    NEW_BLOCK_IGNORED = 3
    NEW_BLOCK_WASTED = 4

    def __str__(self):
        return self.name
