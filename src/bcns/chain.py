#!/usr/bin/env python
# -*- mode: python; coding: utf-8; fill-column: 80; -*-


class Chain(object):
    __slots__ = (
        'blocks',      # List of blocks (representing the blockchain).
        '_block_pos',  # Map of block hashes to their position in the chain.
    )

    def __init__(self, genesis_block):
        self.blocks = [genesis_block]
        self._block_pos = {genesis_block.hash: 0}

    @property
    def length(self):
        return len(self.blocks)

    def add_last_block(self, last_block):
        self.blocks.append(last_block)
        self._block_pos[last_block.hash] = len(self.blocks) - 1

    def get_hashes(self):
        return [b.hash for b in self.blocks]

    def get_non_common_suffix(self, block):
        _block = block
        new_blocks = []
        depth = block.height - self.length

        # Avoid re-evaluating function references.
        _append = new_blocks.append
        for i in range(depth):
            _append(_block)
            _block = _block.previous

        while _block.previous is not None:
            if _block.hash in self._block_pos:
                break

            _append(_block)
            _block = _block.previous

        return new_blocks, depth, _block

    def replace_blocks(self, new_blocks, common_parent):
        idx = self._block_pos[common_parent.hash]

        # Index of the last element in the new chain.
        last = idx + len(new_blocks)

        len_diff = (last + 1) - self.length
        if len_diff:
            self.blocks.extend([None]*len_diff)

        i = 0
        for b in new_blocks:
            pos = last - i

            # Delete the removed blocks also from _block_pos.
            if self.blocks[pos]:
                del self._block_pos[self.blocks[pos].hash]

            self.blocks[pos] = b
            self._block_pos[b.hash] = pos
            i+=1

        if len(self.blocks) > last:
            cl = len(self.blocks)
            for pos in range(last+1, cl):
                if self.blocks[pos]:
                    del self._block_pos[self.blocks[pos].hash]
            #self.blocks = self.blocks[0:last+1]
            del self.blocks[last+1:]

    def __str__(self):
        chain_seq = self.get_hashes()
        return " -> ".join((str(x) for x in chain_seq))
