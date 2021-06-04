#!/usr/bin/env python
# -*- mode: python; coding: utf-8; fill-column: 80; -*-

from bisect import bisect_left

def BinarySearch(a, x):
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    else:
        return -1