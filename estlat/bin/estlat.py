#!/usr/bin/env python3
# -*- mode: python; coding: utf-8; fill-column: 80; -*-
#
# estlat.py
#
"""
estlat.py
Estimate latencies (i.e., one-way delay) between two different locations.
"""


__author__  = ''
__version__ = '1.0'
__license__ = 'MIT'


import io
import itertools
import math
import sys


# Mean Earth radius (in km).
R = 6371.0


def geodesic(xp, yp):
    """geodesic((latx, lonx), (laty, lony)) - calculate the navigational
    distance between any two physical points on the surface of Earth using the
    Haversine formula.
    """
    latx, lonx = xp
    laty, lony = yp
    dlat = math.radians(latx - laty)
    dlon = math.radians(lonx - lony)
    a = ((math.sin(dlat / 2) * math.sin(dlat / 2)) +
         (math.cos(math.radians(latx)) * math.cos(math.radians(laty)) *
          math.sin(dlon / 2) * math.sin(dlon / 2)))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d


def load_cities(fp):
    """Load a list of cities and their latitude-longitude coordinates from a
    given file.
    """
    # Data format:
    # <city-state-country> <lat> <lng>
    # (NOTE: Latitude and longitude are assumed to be in degrees.)
    #
    # WASHINGTON-DC-US 38.904722 -77.016389
    #
    cities = []
    for loc, lat, lng in (line.strip().split() for line in io.open(fp)):
        lat, lng = float(lat), float(lng)
        cities.append((loc, (lat, lng)))
    return cities



def get_adj_list(cities_file):
    """Return a list of adjacencies (assuming all city-to-city edges are
    feasible) along with the length of each edge (in km).
    """
    cities = load_cities(cities_file)
    return itertools.combinations(cities, 2)


def latencies(pairs):
    """Compute latency between pairs of locations.
    """
    dists = [(x, y, geodesic(x[1], y[1])) for (x, y) in pairs]
    # Latency is computed assuming a max. speed of two-thirds of 'C' in fiber,
    # and an inflation factor of 3.2.
    # Stated differently, the computed values estimate the lowest feasible
    # latency assuming optic fiber connectivity over the shortest path between
    # the two locations.
    return [(x, y, d, 3.2*d/200.0) for (x, y, d) in dists ]


def main(*args):
    cities_file, out_file = args
    with io.open(out_file, 'w', encoding='utf-8') as f:
        # Header
        f.write(u"# <location_1> <lat_1> <lng_1>" +
                u" <location_2> <lat_2> <lng_2>" +
                u" <dist. (in km)> <latency (in ms)>\n")

        for (x, y, dist, lat) in latencies(get_adj_list(cities_file)):
            f.write(f"{x[0]} {x[1][0]:.4f} {x[1][1]:.4f}")
            f.write(f" {y[0]} {y[1][0]:.4f} {y[1][1]:.4f}")
            f.write(f" {dist:.2f} {lat:.2f}\n")


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 2:
        prog = sys.argv[0]
        sys.stderr.write(u"Usage: %s <cities-file> <out-file>\n" % (prog))
        sys.exit(1)

    main(*args)
