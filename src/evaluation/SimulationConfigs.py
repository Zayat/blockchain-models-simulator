import math
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd

from bcns import Durations, sim, Simulator, SimulatorCoordinated
from bcns.sim import Equ_LatD, Equ_pooled_LatD, Exp_LatD, Exp_pooled_LatD



def distance_between_2_points(a: tuple, b: tuple) -> float:
    x1, y1 = a
    x2, y2 = b
    return round(math.sqrt((x2 - x1)**2 + (y2 - y1)**2), 6)


def prepare_test_centrality_lat_mat_baseline(nodes):
    return Equ_LatD(3, 1, 0).tolist()


def prepare_test_centrality_lat_mat_1(nodes):
    lat_mat = OrderedDict.fromkeys(nodes)

    M1_lat = np.array([0, 1, 2]) / 2
    M2_lat = np.array([1, 0, 1]) / 2
    M3_lat = np.array([2, 1, 0]) / 2

    latencies = [float('0')] * len(nodes)

    for n in nodes:
        lat_mat[n] = dict(zip(nodes, latencies))
        if n is 'M1':
            lat_mat[n] = dict(zip(nodes, M1_lat))
        if n is 'M2':
            lat_mat[n] = dict(zip(nodes, M2_lat))
        if n is 'M3':
            lat_mat[n] = dict(zip(nodes, M3_lat))

    for n in nodes:
        if n is not 'M1' and n is not 'M2' and n is not 'M3':
            lat_mat[n]['M1'] = lat_mat['M1'][n]
            lat_mat[n]['M2'] = lat_mat['M2'][n]
            lat_mat[n]['M3'] = lat_mat['M3'][n]

    lat_mat = [[lat_mat[i][j] for i in nodes] for j in nodes]
    return lat_mat


def prepare_test_centrality_lat_mat_2(nodes):
    lat_mat = OrderedDict.fromkeys(nodes)

    M1_lat = np.array([0, 1, 4]) / 4
    M2_lat = np.array([1, 0, 3]) / 4
    M3_lat = np.array([4, 3, 0]) / 4

    latencies = [float('0')] * len(nodes)

    for n in nodes:
        lat_mat[n] = dict(zip(nodes, latencies))
        if n is 'M1':
            lat_mat[n] = dict(zip(nodes, M1_lat))
        if n is 'M2':
            lat_mat[n] = dict(zip(nodes, M2_lat))
        if n is 'M3':
            lat_mat[n] = dict(zip(nodes, M3_lat))

    for n in nodes:
        if n is not 'M1' and n is not 'M2' and n is not 'M3':
            lat_mat[n]['M1'] = lat_mat['M1'][n]
            lat_mat[n]['M2'] = lat_mat['M2'][n]
            lat_mat[n]['M3'] = lat_mat['M3'][n]

    lat_mat = [[lat_mat[i][j] for i in nodes] for j in nodes]
    return lat_mat


def prepare2_lat_mat_asymmetric(nodes):
    lat_mat = OrderedDict.fromkeys(nodes)

    M1_lat = (np.array([1, 0, 1, 2, 3, 4, 5, 4]) / 4) 
    M2_lat = (np.array([5, 4, 3, 2, 1, 0, 1, 4]) / 4)

    latencies = [float('0')] * len(nodes)

    for n in nodes:
        lat_mat[n] = dict(zip(nodes, latencies))
        if n is 'M1':
            lat_mat[n] = dict(zip(nodes, M1_lat))
        if n is 'M2':
            lat_mat[n] = dict(zip(nodes, M2_lat))

    for n in nodes:
        if n is not 'M1' and n is not 'M2':
            lat_mat[n]['M1'] = lat_mat['M1'][n]
            lat_mat[n]['M2'] = lat_mat['M2'][n]

    lat_mat = [[lat_mat[i][j] for i in nodes] for j in nodes]

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i < j:
                lat_mat[i][j] = lat_mat[i][j] * 100

    return lat_mat


def prepare2_lat_mat(nodes):
    lat_mat = OrderedDict.fromkeys(nodes)
    # 'OLM1',	'M1',	'OM1',	'OM',	'OM2',	'M2',	'ORM2',	'OEQ'
    OLM1_lat = (np.array([0, 1, 2, 3, 4, 5, 6, 4.5825]) / 4)
    M1_lat = (np.array([1, 0, 1, 2, 3, 4, 5, 4]) / 4)
    ORM1_lat = (np.array([2, 1, 0, 1, 2, 3, 4, 4.583]) / 4)
    OM_lat = (np.array([3, 2, 1, 0, 1, 2, 3, 4.472135955])/4)
    OLM2_lat = (np.array([4, 3, 2, 1, 0, 1, 2, 4.583]) / 4)
    M2_lat = (np.array([5, 4, 3, 2, 1, 0, 1, 4]) / 4)
    ORM2_lat = (np.array([6, 5, 4, 3, 2, 1, 0, 4.5825])/4)
    
    lm1 = (-1, 0)
    m1 = (0, 0)
    rm1 = (1, 0)
    cm12 = (2, 0)
    lm2 = (3, 0)
    m2 = (4, 0)
    rm2 = (5, 0)
    m3 = (2, math.sqrt(12))
    OEQ_lat = (np.array([distance_between_2_points(lm1, m3),
                        4,
                        distance_between_2_points(rm1, m3),
                        distance_between_2_points(cm12, m3),
                        distance_between_2_points(m3, lm2),
                        4,
                        distance_between_2_points(m3, rm2),
                        0]) / 4)


    lat_mat = [OLM1_lat, M1_lat, ORM1_lat, OM_lat,
               OLM2_lat, M2_lat, ORM2_lat, OEQ_lat]
    lat_mat = list(map(lambda x: x.tolist(), lat_mat))

    return lat_mat

def prepare1_coordinators_lat_mat_proportional(proportion):
    C_lat = [0, proportion]
    M1_lat = [proportion, 0]

    lat_mat = [C_lat, M1_lat]
    return lat_mat

def prepare1f_coordinators_lat_mat_proportional(proportion):
    C_lat = [0, 0.5, 0.5 + proportion]
    M1_lat = [0.5, 0, float('inf')]
    M2_lat = [0.5 + proportion, float('inf'), 0]

    lat_mat = [C_lat, M1_lat, M2_lat]
    return lat_mat

def prepare2_coordinators_lat_mat_proportional(proportion):
    C_lat = [0, proportion * 1, (1-proportion) * 1]
    M1_lat = [proportion * 1, 0, float('inf')]
    M2_lat = [(1-proportion) * 1, float('inf'), 0]

    lat_mat = [C_lat, M1_lat, M2_lat]
    return lat_mat

def prepare2_coordinators_lat_mat_proportional_M1_Farther(proportion, factor):
    C_lat = [0, proportion * factor, (1-proportion) * 1]
    M1_lat = [proportion * factor, 0, float('inf')]
    M2_lat = [(1-proportion) * 1, float('inf'), 0]

    lat_mat = [C_lat, M1_lat, M2_lat]
    return lat_mat


def prepare3_coordinators_lat_mat_proportional(proportion):
    m1 = (0,0)
    m2 = (1,0)
    m3 = (0.5, math.sqrt(0.75))
    cp = (0.5, math.sqrt(0.75)-proportion)
    C_lat = [0, distance_between_2_points(cp, m1), distance_between_2_points(cp, m2), distance_between_2_points(cp, m3)]

    M1_lat = [distance_between_2_points(cp, m1), 0, float('inf'), float('inf')]
    M2_lat = [distance_between_2_points(cp, m2), float('inf'), 0, float('inf')]
    M3_lat = [distance_between_2_points(cp, m3), float('inf'), float('inf'), 0]

    lat_mat = [C_lat, M1_lat, M2_lat, M3_lat]
    return lat_mat


def prepare4_p2p_lat_mat_proportional(proportion):
    m1 = (0,1)
    m2 = (1,1)
    m3 = (1,0)
    m4 = (0,0)

    M1_lat = [0, 1, 1.41421, 1]
    M2_lat = [1, 0, 1, 1.41421]
    M3_lat = [1.41421, 1, 0, 1]
    M4_lat = [1, 1.41421, 1, 0]

    lat_mat = [M1_lat, M2_lat, M3_lat, M4_lat]
    return lat_mat

def prepare4_coordinators_lat_mat_proportional(proportion):
    m1 = (0, 1)
    m2 = (1, 1)
    m3 = (1, 0)
    m4 = (0, 0)
    cp = (1-proportion, 1-proportion)

    C_lat = [0,
             distance_between_2_points(cp, m1),
             distance_between_2_points(cp, m2),
             distance_between_2_points(cp, m3),
             distance_between_2_points(cp, m4)]

    M1_lat = [distance_between_2_points(cp, m1), 0, float('inf'), float('inf'), float('inf')]
    M2_lat = [distance_between_2_points(cp, m2), float('inf'), 0, float('inf'), float('inf')]
    M3_lat = [distance_between_2_points(cp, m3), float('inf'), float('inf'), 0, float('inf')]
    M4_lat = [distance_between_2_points(cp, m4), float('inf'), float('inf'), float('inf'), 0]

    lat_mat = [C_lat, M1_lat, M2_lat, M3_lat, M4_lat]
    return lat_mat


def prepare2_coordinators_lat_mat_middle():
    C_lat = [0, .5, .5]
    M1_lat = [.5, 0, float('inf')]
    M2_lat = [.5, float('inf'), 0]

    lat_mat = [C_lat, M1_lat, M2_lat]
    return lat_mat


def prepare2_coordinators_lat_mat_near_weaker():
    C_lat = [0, 0.1, 0.9]
    M1_lat = [0.1, 0, float('inf')]
    M2_lat = [0.9, float('inf'), 0]

    lat_mat = [C_lat, M1_lat, M2_lat]

    return lat_mat


def prepare2_coordinators_lat_mat_near_stronger():
    C_lat = [0, 0.9, 0.1]
    M1_lat = [0.9, 0, float('inf')]
    M2_lat = [0.1, float('inf'), 0]

    lat_mat = [C_lat, M1_lat, M2_lat]

    return lat_mat


def prepare2_coordinators_lat_mat_no_relay(nodes):
    M1_lat = (np.array([0, 1, 2]))
    C_lat = (np.array([1, 0, 1]) / 1000)
    M2_lat = (np.array([2, 1, 0]))

    lat_mat = [M1_lat, C_lat, M2_lat]
    lat_mat = list(map(lambda x: x.tolist(), lat_mat))

    return lat_mat


def prepare3_lat_mat_farther(nodes):
    lat_mat = OrderedDict.fromkeys(nodes)

    M1_lat = np.array([1, 0, 1, 2, 3, 4, 5, 4*10]) / 4
    M2_lat = np.array([5, 4, 3, 2, 1, 0, 1, 4*10]) / 4
    M3_lat = np.array([11, 10, 9, 8, 9, 10, 11, 0])

    latencies = [float('0')] * len(nodes)

    for n in nodes:
        lat_mat[n] = dict(zip(nodes, latencies))
        if n is 'M1':
            lat_mat[n] = dict(zip(nodes, M1_lat))
        if n is 'M2':
            lat_mat[n] = dict(zip(nodes, M2_lat))
        if n is 'M3':
            lat_mat[n] = dict(zip(nodes, M3_lat))

    for n in nodes:
        if n is not 'M1' and n is not 'M2' and n is not 'M3':
            lat_mat[n]['M1'] = lat_mat['M1'][n]
            lat_mat[n]['M2'] = lat_mat['M2'][n]
            lat_mat[n]['M3'] = lat_mat['M3'][n]

    lat_mat = [[lat_mat[i][j] for i in nodes] for j in nodes]
    return lat_mat


def prepare3_lat_mat_fixed_asymetric(nodes):
    lat_mat = OrderedDict.fromkeys(nodes)

    M1_lat = (np.array([1, 0, 1, 2, 3, 400, 5, 4]) / 4)
    M2_lat = (np.array([5, 4, 3, 2, 1, 0, 1, 4]) / 4)
    M3_lat = (
        np.array([4.5825, 4, 4.583, 4.472135955, 4.583, 400, 4.5825, 0]) / 4)

    latencies = [float('0')] * len(nodes)

    for n in nodes:
        lat_mat[n] = dict(zip(nodes, latencies))
        if n is 'M1':
            lat_mat[n] = dict(zip(nodes, M1_lat))
        if n is 'M2':
            lat_mat[n] = dict(zip(nodes, M2_lat))
        if n is 'M3':
            lat_mat[n] = dict(zip(nodes, M3_lat))

    for n in nodes:
        if n is not 'M1' and n is not 'M2' and n is not 'M3':
            lat_mat[n]['M1'] = lat_mat['M1'][n]
            lat_mat[n]['M2'] = lat_mat['M2'][n]
            lat_mat[n]['M3'] = lat_mat['M3'][n]

    lat_mat = [[lat_mat[i][j] for i in nodes] for j in nodes]
    return lat_mat


def prepare3_lat_mat(nodes):
    lat_mat = OrderedDict.fromkeys(nodes)

    M1_lat = (np.array([1, 0, 1, 2, 3, 4, 5, 4]) / 4)
    M2_lat = (np.array([5, 4, 3, 2, 1, 0, 1, 4]) / 4)
    ##Coordinates:
    lm1 = (-1, 0)
    m1 = (0,0)
    rm1 = (1,0)
    cm12 = (2,0)
    lm2 = (3,0)
    m2 = (4,0)
    rm2 = (5,0)
    m3 = (2, math.sqrt(12))
    M3_lat = (np.array([distance_between_2_points(lm1, m3),
                        4,
                        distance_between_2_points(rm1, m3),
                        distance_between_2_points(cm12, m3),
                        distance_between_2_points(m3, lm2),
                        4,
                        distance_between_2_points(m3, rm2),
                        0]) / 4)
    #print(M3_lat)
    latencies = [float('0')] * len(nodes)

    for n in nodes:
        lat_mat[n] = dict(zip(nodes, latencies))
        if n is 'M1':
            lat_mat[n] = dict(zip(nodes, M1_lat))
        if n is 'M2':
            lat_mat[n] = dict(zip(nodes, M2_lat))
        if n is 'M3':
            lat_mat[n] = dict(zip(nodes, M3_lat))

    for n in nodes:
        if n is not 'M1' and n is not 'M2' and n is not 'M3':
            lat_mat[n]['M1'] = lat_mat['M1'][n]
            lat_mat[n]['M2'] = lat_mat['M2'][n]
            lat_mat[n]['M3'] = lat_mat['M3'][n]

    lat_mat = [[lat_mat[i][j] for i in nodes] for j in nodes]
    return lat_mat


def prepare5_lat_mat_fixed(nodes):
    #self.NODES_IDS = ['WA-US', 'SI-CN', 'RE-IS', 'LI-CH', 'MO-RU']
    '''# <location_1> <lat_1> <lng_1> <location_2> <lat_2> <lng_2> <dist. (in km)> <latency (in ms)>
        WASHINGTON-DC-US 38.9047 -77.0164 SICHUAN-NA-CN 30.1333 102.9333 12338.40 197.41
        WASHINGTON-DC-US 38.9047 -77.0164 REYKJAVÍK-NA-IS 64.1333 -21.9333 4512.89 72.21
        WASHINGTON-DC-US 38.9047 -77.0164 LINTHAL-NA-CH 46.9167 9.0000 6703.91 107.26
        WASHINGTON-DC-US 38.9047 -77.0164 MOSCOW-NA-RU 55.7500 37.6167 7820.54 125.13
        SICHUAN-NA-CN 30.1333 102.9333 REYKJAVÍK-NA-IS 64.1333 -21.9333 8489.56 135.83
        SICHUAN-NA-CN 30.1333 102.9333 LINTHAL-NA-CH 46.9167 9.0000 7891.06 126.26
        SICHUAN-NA-CN 30.1333 102.9333 MOSCOW-NA-RU 55.7500 37.6167 5761.37 92.18
        REYKJAVÍK-NA-IS 64.1333 -21.9333 LINTHAL-NA-CH 46.9167 9.0000 2680.24 42.88
        REYKJAVÍK-NA-IS 64.1333 -21.9333 MOSCOW-NA-RU 55.7500 37.6167 3307.89 52.93
        LINTHAL-NA-CH 46.9167 9.0000 MOSCOW-NA-RU 55.7500 37.61672196.05 35.14

    '''
    #               ['WA-US', 'SI-CN', 'RE-IS', 'LI-CH', 'MO-RU']
    WA_lat = np.array([0, 197.41, 72.21, 107.26, 125.13])/ (1000*1.5)
    SI_lat = np.array([-1, 0, 135.83, 126.26, 92.18])/ (1000*1.5)
    RE_lat = np.array([-1, -1, 0, 42.88, 52.93])/ (1000*1.5)
    LI_lat = np.array([-1, -1, -1, 0, 35.14])/ (1000*1.5)
    MO_lat = np.array([-1, -1, -1, -1, 0])/ (1000*1.5)

    lat_mat = [WA_lat, SI_lat, RE_lat, LI_lat, MO_lat]

    for i in range(len(lat_mat)):
        for j in range(len(lat_mat)):
            if i > j:
                lat_mat[i][j] = lat_mat[j][i]

    return lat_mat


def prepare100_lat_mat_fixed_centrality(nodes):
    latencies = pd.read_csv('evaluation/100_cities.txt', delim_whitespace=True)
    lat_dict = defaultdict(dict)

    for i in range(len(latencies)):
        row = latencies.iloc[i]
        lat_dict[row['location_1']][row['location_2']] = row['latency_ms']
        lat_dict[row['location_2']][row['location_1']] = row['latency_ms']

    lat_mat = [[float('0') for i in nodes] for j in nodes]

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                lat_mat[i][j] = (lat_dict[nodes[i]][nodes[j]] / (1000*1.5))

    return lat_mat


def prepare240_lat_mat_fixed_capital_centrality(nodes):
    latencies = pd.read_csv(
        'evaluation/cities_capitals_lat_lng_latency.txt', delim_whitespace=True)
    lat_dict = defaultdict(dict)

    for i in range(len(latencies)):
        row = latencies.iloc[i]
        lat_dict[row['location_1']][row['location_2']] = row['latency_ms']
        lat_dict[row['location_2']][row['location_1']] = row['latency_ms']

    lat_mat = [[float('0') for i in nodes] for j in nodes]

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                lat_mat[i][j] = (lat_dict[nodes[i]][nodes[j]]/(1000*1.5))

    return lat_mat


def prepare15_lat_mat_ls_fixed_capital_centrality(nodes):
    latencies = pd.read_csv(
        'evaluation/cities_capitals_lat_lng_latency.txt', delim_whitespace=True)
    lat_dict = defaultdict(dict)

    for i in range(len(latencies)):
        row = latencies.iloc[i]
        lat_dict[row['location_1']][row['location_2']] = row['latency_ms']
        lat_dict[row['location_2']][row['location_1']] = row['latency_ms']

    lat_mat = [[float('0') for i in nodes] for j in nodes]

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                lat_mat[i][j] = (lat_dict[nodes[i]][nodes[j]] / (1000*3.2))

    return lat_mat


def prepare240_lat_mat_cs_fixed_capital_centrality(nodes):
    latencies = pd.read_csv(
        'evaluation/cities_capitals_lat_lng_latency.txt', delim_whitespace=True)
    lat_dict = defaultdict(dict)

    for i in range(len(latencies)):
        row = latencies.iloc[i]
        lat_dict[row['location_1']][row['location_2']] = row['latency_ms']
        lat_dict[row['location_2']][row['location_1']] = row['latency_ms']

    lat_mat = [[float('0') for i in nodes] for j in nodes]

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                lat_mat[i][j] = (lat_dict[nodes[i]][nodes[j]] / (1000*3.2*1.5))

    return lat_mat


def prepare15_lat_mat_fixed(nodes):
    # nodes= ['WASHINGTON-DC-US', 'SICHUAN-NA-CN', 'REYKJAVÍK-NA-IS',
    #   'LINTHAL-NA-CH', 'MOSCOW-NA-RU', 'TBILISI-NA-GE', 'KIEV-NA-UK',
    #  'ANKARA-NA-TR', 'SKOPJE-NA-MK', 'HELSINKI-NA-FI', 'MANNHEIM-BW-DE',
    #  'SINGAPORE-NA-SG', 'ASHBURN-VA-US', 'FRANKFURT-HE-DE', 'NUREMBURG-BV-DE']

    latencies = pd.read_csv('evaluation/adjlst-2.txt', delim_whitespace=True)
    nodes = ['WASHINGTON-DC-US', 'SICHUAN-NA-CN', 'REYKJAVÍK-NA-IS',
                 'LINTHAL-NA-CH', 'MOSCOW-NA-RU', 'TBILISI-NA-GE', 'KIEV-NA-UK',
                 'ANKARA-NA-TR', 'SKOPJE-NA-MK', 'HELSINKI-NA-FI', 'MANNHEIM-BW-DE',
                 'SINGAPORE-NA-SG', 'ASHBURN-VA-US', 'FRANKFURT-HE-DE', 'NUREMBURG-BV-DE']

    lat_mat = [[float('0') for i in nodes] for j in nodes]

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                f1 = latencies[latencies['location_1'] == nodes[i]
                               ][latencies['location_2'] == nodes[j]]
                if len(f1) == 0:
                    f2 = latencies[latencies['location_2'] == nodes[i]
                                   ][latencies['location_1'] == nodes[j]]
                    result = f2['latency_ms'].iloc[0]
                else:
                    result = f1['latency_ms'].iloc[0]

                lat_mat[i][j] = (result/(1000*1.5))

    return lat_mat


def prepare15_ls_lat_mat_fixed(nodes):
    # nodes= ['WASHINGTON-DC-US', 'SICHUAN-NA-CN', 'REYKJAVÍK-NA-IS',
    #   'LINTHAL-NA-CH', 'MOSCOW-NA-RU', 'TBILISI-NA-GE', 'KIEV-NA-UK',
    #  'ANKARA-NA-TR', 'SKOPJE-NA-MK', 'HELSINKI-NA-FI', 'MANNHEIM-BW-DE',
    #  'SINGAPORE-NA-SG', 'ASHBURN-VA-US', 'FRANKFURT-HE-DE', 'NUREMBURG-BV-DE']

    latencies = pd.read_csv('evaluation/adjlst-2.txt', delim_whitespace=True)
    nodes = ['WASHINGTON-DC-US', 'SICHUAN-NA-CN', 'REYKJAVÍK-NA-IS',
                 'LINTHAL-NA-CH', 'MOSCOW-NA-RU', 'TBILISI-NA-GE', 'KIEV-NA-UK',
                 'ANKARA-NA-TR', 'SKOPJE-NA-MK', 'HELSINKI-NA-FI', 'MANNHEIM-BW-DE',
                 'SINGAPORE-NA-SG', 'ASHBURN-VA-US', 'FRANKFURT-HE-DE', 'NUREMBURG-BV-DE']

    lat_mat = [[float('0') for i in nodes] for j in nodes]

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                f1 = latencies[latencies['location_1'] == nodes[i]
                               ][latencies['location_2'] == nodes[j]]
                if len(f1) == 0:
                    f2 = latencies[latencies['location_2'] == nodes[i]
                                   ][latencies['location_1'] == nodes[j]]
                    result = f2['latency_ms'].iloc[0]
                else:
                    result = f1['latency_ms'].iloc[0]

                lat_mat[i][j] = (result/(1000*3.2))

    return lat_mat


def prepare15_cs_lat_mat_fixed(nodes):
    # nodes= ['WASHINGTON-DC-US', 'SICHUAN-NA-CN', 'REYKJAVÍK-NA-IS',
    #   'LINTHAL-NA-CH', 'MOSCOW-NA-RU', 'TBILISI-NA-GE', 'KIEV-NA-UK',
    #  'ANKARA-NA-TR', 'SKOPJE-NA-MK', 'HELSINKI-NA-FI', 'MANNHEIM-BW-DE',
    #  'SINGAPORE-NA-SG', 'ASHBURN-VA-US', 'FRANKFURT-HE-DE', 'NUREMBURG-BV-DE']

    latencies = pd.read_csv('evaluation/adjlst-2.txt', delim_whitespace=True)
    nodes = ['WASHINGTON-DC-US', 'SICHUAN-NA-CN', 'REYKJAVÍK-NA-IS',
                 'LINTHAL-NA-CH', 'MOSCOW-NA-RU', 'TBILISI-NA-GE', 'KIEV-NA-UK',
                 'ANKARA-NA-TR', 'SKOPJE-NA-MK', 'HELSINKI-NA-FI', 'MANNHEIM-BW-DE',
                 'SINGAPORE-NA-SG', 'ASHBURN-VA-US', 'FRANKFURT-HE-DE', 'NUREMBURG-BV-DE']

    lat_mat = [[float('0') for i in nodes] for j in nodes]

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                f1 = latencies[latencies['location_1'] == nodes[i]
                               ][latencies['location_2'] == nodes[j]]
                if len(f1) == 0:
                    f2 = latencies[latencies['location_2'] == nodes[i]
                                   ][latencies['location_1'] == nodes[j]]
                    result = f2['latency_ms'].iloc[0]
                else:
                    result = f1['latency_ms'].iloc[0]

                lat_mat[i][j] = (result/(1000*3.2*1.5))

    return lat_mat


def to_dataframe_prepare_test_centrality_lat_mat_baseline(experiments_stats, nodes_ids):
    df = pd.DataFrame(experiments_stats)
    miner_df = list()
    for miner in df.miners:
        miner_df.append(pd.DataFrame(miner))
    miner_df = pd.concat(miner_df)
    df.drop(columns=['miners'], inplace=True)
    df.hpd = df.hpd.apply(lambda x: f"[{x[0]}, {x[1]}, {x[2]}]")
    miner_df.global_hpd = miner_df.global_hpd.apply(
        lambda x: f"[{x[0]}, {x[1]}, {x[2]}]")
    miner_df.id = miner_df.id.map(dict(zip(range(0, 3), nodes_ids)))
    return {'miner': miner_df, 'global': df}


def to_dataframe2(experiments_stats, nodes_ids, nodes_count=2):
    df = pd.DataFrame(experiments_stats)
    miner_df = list()
    for miner in df.miners:
        miner_df.append(pd.DataFrame(miner))
    miner_df = pd.concat(miner_df)
    df.drop(columns=['miners'], inplace=True)
    df.hpd = df.hpd.apply(lambda x: f"[{x[1]}, {x[5]}]")
    miner_df.global_hpd = miner_df.global_hpd.apply(
        lambda x: f"[{x[1]}, {x[5]}]")
    miner_df.id = miner_df.id.map(dict(zip(range(0, 8), nodes_ids)))
    return {'miner': miner_df, 'global': df}

def to_dataframe1_coordinators(experiments_stats, nodes_ids, nodes_count=2):
    df = pd.DataFrame(experiments_stats)
    miner_df = list()
    for miner in df.miners:
        miner_df.append(pd.DataFrame(miner))
    miner_df = pd.concat(miner_df)
    df.drop(columns=['miners'], inplace=True)
    df.hpd = df.hpd.apply(lambda x: f"[{x[0]}, {x[1]}]")
    miner_df.global_hpd = miner_df.global_hpd.apply(
        lambda x: f"[{x[0]}, {x[1]}]")
    miner_df.id = miner_df.id.map(
        dict(zip(range(0, len(nodes_ids)), nodes_ids)))
    return {'miner': miner_df, 'global': df}

def to_dataframe2_coordinators(experiments_stats, nodes_ids, nodes_count=3):
    df = pd.DataFrame(experiments_stats)
    miner_df = list()
    for miner in df.miners:
        miner_df.append(pd.DataFrame(miner))
    miner_df = pd.concat(miner_df)
    df.drop(columns=['miners'], inplace=True)
    df.hpd = df.hpd.apply(lambda x: f"[{x[1]}, {x[2]}]")
    miner_df.global_hpd = miner_df.global_hpd.apply(
        lambda x: f"[{x[1]}, {x[2]}]")
    miner_df.id = miner_df.id.map(
        dict(zip(range(0, len(nodes_ids)), nodes_ids)))
    return {'miner': miner_df, 'global': df}

def to_dataframe3_coordinators(experiments_stats, nodes_ids, nodes_count=3):
    df = pd.DataFrame(experiments_stats)
    miner_df = list()
    for miner in df.miners:
        miner_df.append(pd.DataFrame(miner))
    miner_df = pd.concat(miner_df)
    df.drop(columns=['miners'], inplace=True)
    df.hpd = df.hpd.apply(lambda x: f"[{x[1]}, {x[2]}, {x[3]}]")
    miner_df.global_hpd = miner_df.global_hpd.apply(
        lambda x: f"[{x[1]}, {x[2]}, {x[3]}]")
    miner_df.id = miner_df.id.map(
        dict(zip(range(0, len(nodes_ids)), nodes_ids)))
    return {'miner': miner_df, 'global': df}

def to_dataframe4(experiments_stats, nodes_ids, nodes_count=3):
    df = pd.DataFrame(experiments_stats)
    miner_df = list()
    for miner in df.miners:
        miner_df.append(pd.DataFrame(miner))
    miner_df = pd.concat(miner_df)
    df.drop(columns=['miners'], inplace=True)
    df.hpd = df.hpd.apply(lambda x: f"[{x[0]}, {x[1]}, {x[2]}, {x[3]}]")
    miner_df.global_hpd = miner_df.global_hpd.apply(
        lambda x: f"[{x[0]}, {x[1]}, {x[2]}, {x[3]}]")
    miner_df.id = miner_df.id.map(
        dict(zip(range(0, len(nodes_ids)), nodes_ids)))
    return {'miner': miner_df, 'global': df}

def to_dataframe4_coordinators(experiments_stats, nodes_ids, nodes_count=3):
    df = pd.DataFrame(experiments_stats)
    miner_df = list()
    for miner in df.miners:
        miner_df.append(pd.DataFrame(miner))
    miner_df = pd.concat(miner_df)
    df.drop(columns=['miners'], inplace=True)
    df.hpd = df.hpd.apply(lambda x: f"[{x[1]}, {x[2]}, {x[3]}, {x[4]}]")
    miner_df.global_hpd = miner_df.global_hpd.apply(
        lambda x: f"[{x[1]}, {x[2]}, {x[3]}, {x[4]}]")
    miner_df.id = miner_df.id.map(
        dict(zip(range(0, len(nodes_ids)), nodes_ids)))
    return {'miner': miner_df, 'global': df}


def to_dataframe3(experiments_stats, nodes_ids, nodes_count=3):
    df = pd.DataFrame(experiments_stats)
    miner_df = list()
    for miner in df.miners:
        miner_df.append(pd.DataFrame(miner))
    miner_df = pd.concat(miner_df)
    df.drop(columns=['miners'], inplace=True)
    df.hpd = df.hpd.apply(lambda x: f"[{x[1]}, {x[5]}, {x[7]}]")
    miner_df.global_hpd = miner_df.global_hpd.apply(
        lambda x: f"[{x[1]}, {x[5]}, {x[7]}]")
    miner_df.id = miner_df.id.map(dict(zip(range(0, 8), nodes_ids)))
    return {'miner': miner_df, 'global': df}


def to_dataframe5(experiments_stats, nodes_ids, nodes_count=5):
    df = pd.DataFrame(experiments_stats)
    miner_df = list()
    for miner in df.miners:
        miner_df.append(pd.DataFrame(miner))
    miner_df = pd.concat(miner_df)
    df.drop(columns=['miners'], inplace=True)
    df.hpd = df.hpd.apply(
        lambda x: f"[{x[0]}, {x[1]}, {x[2]}, {x[3]}, {x[4]}]")
    miner_df.global_hpd = miner_df.global_hpd.apply(
        lambda x:  f"[{x[0]}, {x[1]}, {x[2]}, {x[3]}, {x[4]}]")
    miner_df.id = miner_df.id.map(dict(zip(range(0, 5), nodes_ids)))
    return {'miner': miner_df, 'global': df}


def to_dataframe15(experiments_stats, nodes_ids, nodes_count=15):
    df = pd.DataFrame(experiments_stats)
    miner_df = list()
    for miner in df.miners:
        miner_df.append(pd.DataFrame(miner))
    miner_df = pd.concat(miner_df)
    df.drop(columns=['miners'], inplace=True)
    df.hpd = df.hpd.apply(
        lambda x: f"[{x[0]}, {x[1]}, {x[2]}, {x[3]}, {x[4]}]")
    miner_df.global_hpd = miner_df.global_hpd.apply(
        lambda x:  f"[{x[0]}, {x[1]}, {x[2]}, {x[3]}, {x[4]}]")
    miner_df.id = miner_df.id.map(dict(zip(range(0, nodes_count), nodes_ids)))
    return {'miner': miner_df, 'global': df}


def to_dataframe_real_bc(experiments_stats, nodes_ids):
    df = pd.DataFrame(experiments_stats)
    miner_df = list()
    for miner in df.miners:
        miner_df.append(pd.DataFrame(miner))
    miner_df = pd.concat(miner_df)
    df.drop(columns=['miners'], inplace=True)
    df.hpd = df.hpd.apply(lambda x: f"[{x[0]}, {x[1]}, {x[2]}]")
    miner_df.global_hpd = miner_df.global_hpd.apply(
        lambda x: f"[{x[0]}, {x[1]}, {x[2]}]")
    miner_df.id = miner_df.id.map(
        dict(zip(range(0, len(nodes_ids)), nodes_ids)))
    return {'miner': miner_df, 'global': df}


coord_dist = 0
def COORDINATOR_DISTANCE(val=-1):
    global coord_dist
    if val >= 0:
        coord_dist = val
    return coord_dist

class Default2SimulationConfigs(object):
    def __init__(self):

        self.EXPERIMENT_IDENTIFIER = "2miners_toy_example_"
        self.PERSIST_DATA = True
        self.DURATION = 2 * Durations.HOUR
        self.NUM_MINERS = 8
        self.SEED = 0
        self.NUM_BLOCKS_TO_GENERATE = 1000 #100000 #Make it faster for the reviewers :)
        self.NUM_ITER = 10
        self.NODES_IDS = ['OLM1', 'M1', 'OM1', 'OM', 'OM2', 'M2', 'ORM2', 'OEQ']
        self.LATENCY_ADJACECY_MATRIX = prepare2_lat_mat(self.NODES_IDS)
        self.MEAN_NETWORK_LATENCY_S = np.mean(self.LATENCY_ADJACECY_MATRIX)  # 1
        self.MEAN_NETWORK_LATENCY_S = 1 

        self.HPD_CFG = [(0, 0.5, 0, 0, 0, 0.5, 0, 0),
                   (0, 0.4, 0, 0, 0, 0.6, 0, 0),
                   (0, 0.3, 0, 0, 0, 0.7, 0, 0),
                   (0, 0.2, 0, 0, 0, 0.8, 0, 0),
                   (0, 0.1, 0, 0, 0, 0.9, 0, 0)]

        #self.HPD_CFG = [(0, 0.5, 0, 0, 0, 0.5, 0, 0)]
        #self.HPD_CFG = [(0, x, 0, 0, 0, 1-x, 0, 0) for x in np.arange(0,1.01, 0.05)]

        self.HARDNESS_CFG = [10000 * self.MEAN_NETWORK_LATENCY_S,
                        1000 * self.MEAN_NETWORK_LATENCY_S,
                        100 * self.MEAN_NETWORK_LATENCY_S,
                        10 * self.MEAN_NETWORK_LATENCY_S,
                        #8 * self.MEAN_NETWORK_LATENCY_S,
                        #6 * self.MEAN_NETWORK_LATENCY_S,
                        #4 * self.MEAN_NETWORK_LATENCY_S,
                        #5 * self.MEAN_NETWORK_LATENCY_S,
                        #2 * self.MEAN_NETWORK_LATENCY_S,
                        self.MEAN_NETWORK_LATENCY_S,
                        #self.MEAN_NETWORK_LATENCY_S / 2,
                        #self.MEAN_NETWORK_LATENCY_S / 4,
                        #self.MEAN_NETWORK_LATENCY_S / 5,
                        #self.MEAN_NETWORK_LATENCY_S / 6,
                        #self.MEAN_NETWORK_LATENCY_S / 8,
                        self.MEAN_NETWORK_LATENCY_S / 10,
                        self.MEAN_NETWORK_LATENCY_S / 100]
        #self.HARDNESS_CFG = [ 1.0*10.0**e for e in np.arange(-2, 4.1, 0.1)]
        #self.HARDNESS_CFG = [ 1.0*10.0**e for e in np.arange(-4.2, 4.2, 0.2)]

        self.SIMULATOR = Simulator
        self.TO_DATAFRAME = to_dataframe2


class Default1fCoordinatorSimulationConfigs(object):
    def __init__(self):
        self.EXPERIMENT_IDENTIFIER = "2 nodes 1 moving toy_example_coordinated_" + str(
            COORDINATOR_DISTANCE())
        self.PERSIST_DATA = True
        self.DURATION = 5 * Durations.HOUR
        self.NUM_MINERS = 3
        NUM_COORDINATORS = 1
        NUM_OBSERVERS = 0
        self.SEED = 0
        self.NUM_BLOCKS_TO_GENERATE = 10000
        self.NUM_ITER = 10
        self.NODES_IDS = ['C', 'M1', 'M2']
        self.LATENCY_ADJACECY_MATRIX = prepare1f_coordinators_lat_mat_proportional(
            COORDINATOR_DISTANCE() / 10.0)
        print("This is the 1 moving miners coordinated latency matrix")
        print(self.LATENCY_ADJACECY_MATRIX)
        self.MEAN_NETWORK_LATENCY_S = 1

        self.HPD_CFG = [(0, .5, .5),
                        (0, .7, .3),
                        (0, .9, .1)]

        self.HARDNESS_CFG = [
            #10000 * self.MEAN_NETWORK_LATENCY_S,
            1000 * self.MEAN_NETWORK_LATENCY_S,
            100 * self.MEAN_NETWORK_LATENCY_S,
            10 * self.MEAN_NETWORK_LATENCY_S,
            self.MEAN_NETWORK_LATENCY_S,
            self.MEAN_NETWORK_LATENCY_S / 10]

        self.SIMULATOR = SimulatorCoordinated
        self.TO_DATAFRAME = to_dataframe2_coordinators


class Default2CoordinatorSimulationConfigs(object):
    def __init__(self):
        self.EXPERIMENT_IDENTIFIER = "2coordinators_toy_example_coordinated_"+str(COORDINATOR_DISTANCE()) 
        self.PERSIST_DATA = True
        self.DURATION = 2 * Durations.HOUR
        self.NUM_MINERS = 3
        NUM_COORDINATORS = 1
        NUM_OBSERVERS = 0
        self.SEED = 0
        self.NUM_BLOCKS_TO_GENERATE = 100000
        self.NUM_ITER = 10
        self.NODES_IDS = ['C', 'M1', 'M2']
        self.LATENCY_ADJACECY_MATRIX = prepare2_coordinators_lat_mat_proportional(COORDINATOR_DISTANCE()/10.0)
        # self.MEAN_NETWORK_LATENCY_S = np.mean(self.LATENCY_ADJACECY_MATRIX)  # 1
        self.MEAN_NETWORK_LATENCY_S = 1
    
        self.HPD_CFG = [(0, .5, .5),
                   (0, .4, .6),
                   (0, .3, .7),
                   (0, .2, .8),
                   (0, .1, .9)]
        #self.HPD_CFG = [(0, .1, .9)]
    
        #self.HPD_CFG = [(0, 0.5, 0, 0, 0, 0.5, 0, 0)]
        #self.HPD_CFG = [(0, x, 0, 0, 0, 1-x, 0, 0) for x in np.arange(0,1.01, 0.05)]
    
        self.HARDNESS_CFG = [
            10000 * self.MEAN_NETWORK_LATENCY_S,
            1000 * self.MEAN_NETWORK_LATENCY_S,
            100 * self.MEAN_NETWORK_LATENCY_S,
            10 * self.MEAN_NETWORK_LATENCY_S,
            8 * self.MEAN_NETWORK_LATENCY_S,
            6 * self.MEAN_NETWORK_LATENCY_S,
            4 * self.MEAN_NETWORK_LATENCY_S,
            5 * self.MEAN_NETWORK_LATENCY_S,
            2 * self.MEAN_NETWORK_LATENCY_S,
            self.MEAN_NETWORK_LATENCY_S,
            self.MEAN_NETWORK_LATENCY_S / 2,
            self.MEAN_NETWORK_LATENCY_S / 4,
            self.MEAN_NETWORK_LATENCY_S / 5,
            self.MEAN_NETWORK_LATENCY_S / 6,
            self.MEAN_NETWORK_LATENCY_S / 8,
            self.MEAN_NETWORK_LATENCY_S / 10,
            self.MEAN_NETWORK_LATENCY_S / 100
        ]
        #self.HARDNESS_CFG = [ self.MEAN_NETWORK_LATENCY_S]
    
        #self.HARDNESS_CFG = [ 1.0*10.0**e for e in np.arange(-2, 4.1, 0.1)]
        #self.HARDNESS_CFG = [ 1.0*10.0**e for e in np.arange(-4.2, 4.2, 0.2)]
    
        self.SIMULATOR = SimulatorCoordinated
        self.TO_DATAFRAME = to_dataframe2_coordinators


class Default2CoordinatorEquidistantSimulationConfigs(object):
    def __init__(self):
        self.EXPERIMENT_IDENTIFIER = "2coordinators_toy_example_coordinated_equidistant" + str(
            COORDINATOR_DISTANCE())
        self.PERSIST_DATA = True
        self.DURATION = 2 * Durations.HOUR
        self.NUM_MINERS = 3
        NUM_COORDINATORS = 1
        NUM_OBSERVERS = 0
        self.SEED = 0
        self.NUM_BLOCKS_TO_GENERATE = 100000
        self.NUM_ITER = 10
        self.NODES_IDS = ['C', 'M1', 'M2']
        self.LATENCY_ADJACECY_MATRIX = prepare2_coordinators_lat_mat_proportional(
            COORDINATOR_DISTANCE() / 10.0)

        self.MEAN_NETWORK_LATENCY_S = 1

        self.HPD_CFG = [(0, .5, .5),
                        (0, .99, .01),
                        (0, .4, .6),
                        (0, .3, .7),
                        (0, .2, .8),
                        (0, .1, .9),
                        (0, .05, .95)
                        ]

        self.HARDNESS_CFG = [
            1000 * self.MEAN_NETWORK_LATENCY_S,
            100 * self.MEAN_NETWORK_LATENCY_S,
            10 * self.MEAN_NETWORK_LATENCY_S,
            self.MEAN_NETWORK_LATENCY_S,
            self.MEAN_NETWORK_LATENCY_S / 10,
            self.MEAN_NETWORK_LATENCY_S / 100
        ]

        self.SIMULATOR = SimulatorCoordinated
        self.TO_DATAFRAME = to_dataframe2_coordinators


class Default2CoordinatorFartherSimulationConfigs(object):
    def __init__(self):
        self.EXPERIMENT_IDENTIFIER = "2coordinators_toy_example_coordinated_farther" + str(
            COORDINATOR_DISTANCE())
        self.PERSIST_DATA = True
        self.DURATION = 2 * Durations.HOUR
        self.NUM_MINERS = 3
        NUM_COORDINATORS = 1
        NUM_OBSERVERS = 0
        self.SEED = 0
        self.NUM_BLOCKS_TO_GENERATE = 100000
        self.NUM_ITER = 10
        self.NODES_IDS = ['C', 'M1', 'M2']
        self.LATENCY_ADJACECY_MATRIX = prepare2_coordinators_lat_mat_proportional_M1_Farther(
            COORDINATOR_DISTANCE() / 10.0, 2)
        # self.MEAN_NETWORK_LATENCY_S = np.mean(self.LATENCY_ADJACECY_MATRIX)  # 1
        self.MEAN_NETWORK_LATENCY_S = 1

        self.HPD_CFG = [(0, .5, .5),
                        (0, .99, .01),
                        (0, .4, .6),
                        (0, .3, .7),
                        (0, .2, .8),
                        (0, .1, .9),
                        (0, .05, .95)
                        ]

        self.HARDNESS_CFG = [
            1000 * self.MEAN_NETWORK_LATENCY_S,
            100 * self.MEAN_NETWORK_LATENCY_S,
            10 * self.MEAN_NETWORK_LATENCY_S,
            self.MEAN_NETWORK_LATENCY_S,
            self.MEAN_NETWORK_LATENCY_S / 10,
            self.MEAN_NETWORK_LATENCY_S / 100
        ]
        # self.HARDNESS_CFG = [ self.MEAN_NETWORK_LATENCY_S]

        # self.HARDNESS_CFG = [ 1.0*10.0**e for e in np.arange(-2, 4.1, 0.1)]
        # self.HARDNESS_CFG = [ 1.0*10.0**e for e in np.arange(-4.2, 4.2, 0.2)]

        self.SIMULATOR = SimulatorCoordinated
        self.TO_DATAFRAME = to_dataframe2_coordinators

class Default3CoordinatorSimulationConfigs(object):
    def __init__(self):
    
        self.EXPERIMENT_IDENTIFIER = "3coordinators_toy_example_coordinated_"+str(COORDINATOR_DISTANCE()) 
        self.PERSIST_DATA = True
        self.DURATION = 2 * Durations.HOUR
        self.NUM_MINERS = 4
        NUM_COORDINATORS = 1
        NUM_OBSERVERS = 0
        self.SEED = 0
        self.NUM_BLOCKS_TO_GENERATE = 100000
        self.NUM_ITER = 10
        self.NODES_IDS = ['C', 'M1', 'M2', 'M3']
        self.LATENCY_ADJACECY_MATRIX = prepare3_coordinators_lat_mat_proportional(COORDINATOR_DISTANCE()/10.0)
        print("This is the 3 miners coordinated latency matrix")
        print(self.LATENCY_ADJACECY_MATRIX)
        self.MEAN_NETWORK_LATENCY_S = 1

        #Switch M2 and M3 hash powers to simulate moving the coordinator from the weaker node 
        self.HPD_CFG = [(0, .33, .33, .33),
                   (0, .3,  .3,  .4),
                   (0, .2,  .2,  .6),
                   (0, .1,  .1,  .8)]
    
        self.HARDNESS_CFG = [
            10000 * self.MEAN_NETWORK_LATENCY_S,
            1000 * self.MEAN_NETWORK_LATENCY_S,
            100 * self.MEAN_NETWORK_LATENCY_S,
            10 * self.MEAN_NETWORK_LATENCY_S,
            self.MEAN_NETWORK_LATENCY_S,
            self.MEAN_NETWORK_LATENCY_S / 10,
            self.MEAN_NETWORK_LATENCY_S / 100
        ]
    
        self.SIMULATOR = SimulatorCoordinated
        self.TO_DATAFRAME = to_dataframe3_coordinators


class Default4P2PSimulationConfigs(object):
    def __init__(self):
        self.EXPERIMENT_IDENTIFIER = "4p2p_toy_example_coordinated_" + str(
            COORDINATOR_DISTANCE())
        self.PERSIST_DATA = True
        self.DURATION = 2 * Durations.HOUR
        self.NUM_MINERS = 4
        self.SEED = 0
        self.NUM_BLOCKS_TO_GENERATE = 100000
        self.NUM_ITER = 10
        self.NODES_IDS = ['C', 'M1', 'M2', 'M3', 'M4']
        self.LATENCY_ADJACECY_MATRIX = prepare4_p2p_lat_mat_proportional(
            COORDINATOR_DISTANCE() / 10.0)
        print("This is the 3 miners coordinated latency matrix")
        print(self.LATENCY_ADJACECY_MATRIX)
        self.MEAN_NETWORK_LATENCY_S = 1

        self.HPD_CFG = [(0.25, 0.25, 0.25, 0.25),
                        (0.7, 0.1, 0.1, 0.1)]

        self.HARDNESS_CFG = [
            10000 * self.MEAN_NETWORK_LATENCY_S,
            1000 * self.MEAN_NETWORK_LATENCY_S,
            100 * self.MEAN_NETWORK_LATENCY_S,
            10 * self.MEAN_NETWORK_LATENCY_S,
            self.MEAN_NETWORK_LATENCY_S,
            self.MEAN_NETWORK_LATENCY_S / 10,
            self.MEAN_NETWORK_LATENCY_S / 100
        ]

        self.SIMULATOR = Simulator
        self.TO_DATAFRAME = to_dataframe4

class Default4CoordinatorSimulationConfigs(object):
    def __init__(self):
        self.EXPERIMENT_IDENTIFIER = "4coordinators_toy_example_coordinated_" + str(
            COORDINATOR_DISTANCE())
        self.PERSIST_DATA = True
        self.DURATION = 2 * Durations.HOUR
        self.NUM_MINERS = 5
        NUM_COORDINATORS = 1
        NUM_OBSERVERS = 0
        self.SEED = 0
        self.NUM_BLOCKS_TO_GENERATE = 100000
        self.NUM_ITER = 10
        self.NODES_IDS = ['C', 'M1', 'M2', 'M3', 'M4']
        self.LATENCY_ADJACECY_MATRIX = prepare4_coordinators_lat_mat_proportional(
            COORDINATOR_DISTANCE() / 10.0)
        print("This is the 3 miners coordinated latency matrix")
        print(self.LATENCY_ADJACECY_MATRIX)
        self.MEAN_NETWORK_LATENCY_S = 1

        self.HPD_CFG = [(0, 0.25, 0.25, 0.25, 0.25),
                        (0, 0.7, 0.1, 0.1, 0.1)]
        #(0, 0.1, 0.7, 0.1, 0.1)

        self.HARDNESS_CFG = [
            10000 * self.MEAN_NETWORK_LATENCY_S,
            1000 * self.MEAN_NETWORK_LATENCY_S,
            100 * self.MEAN_NETWORK_LATENCY_S,
            10 * self.MEAN_NETWORK_LATENCY_S,
            self.MEAN_NETWORK_LATENCY_S,
            self.MEAN_NETWORK_LATENCY_S / 10,
            self.MEAN_NETWORK_LATENCY_S / 100
        ]

        self.SIMULATOR = SimulatorCoordinated
        self.TO_DATAFRAME = to_dataframe4_coordinators


class Default3SimulationConfigs(object):
    def __init__(self):
    
        self.EXPERIMENT_IDENTIFIER = "3miners_toy_example_paper_"
        self.PERSIST_DATA = True
        self.DURATION = 2 * Durations.HOUR
        self.NUM_MINERS = 8
        self.SEED = 0
        self.NUM_BLOCKS_TO_GENERATE = 100000
        self.NUM_ITER = 10
        self.NODES_IDS = ['OLM1', 'M1', 'OM1', 'OM', 'OM2', 'M2', 'ORM2', 'M3']
        self.HPD_CFG = [(0, .33, 0, 0, 0, .33, 0, .33),
                   (0, .3, 0, 0, 0, .4, 0, .3),
                   (0, .2, 0, 0, 0, .6, 0, .2),
                   (0, .1, 0, 0, 0, .8, 0, .1)]
        #self.HPD_CFG = [(0, .33, 0, 0, 0, .33, 0, .33)]
        self.LATENCY_ADJACECY_MATRIX = prepare3_lat_mat(self.NODES_IDS)
        self.MEAN_NETWORK_LATENCY_S = np.mean(self.LATENCY_ADJACECY_MATRIX)
        self.MEAN_NETWORK_LATENCY_S = 1
    
        '''self.HARDNESS_CFG = [100000 * self.MEAN_NETWORK_LATENCY_S,
                        10000 * self.MEAN_NETWORK_LATENCY_S,
                        1000 * self.MEAN_NETWORK_LATENCY_S,
                        100 * self.MEAN_NETWORK_LATENCY_S,
                        10 * self.MEAN_NETWORK_LATENCY_S,
                        self.MEAN_NETWORK_LATENCY_S,
                        self.MEAN_NETWORK_LATENCY_S / 10,
                        self.MEAN_NETWORK_LATENCY_S / 100,
                        17, 600
                        ]
        self.HARDNESS_CFG = [ 1.0*10.0**e for e in np.arange(-4.2, 4.2, 0.2)]'''
        self.HARDNESS_CFG = [10000 * self.MEAN_NETWORK_LATENCY_S,
                        1000 * self.MEAN_NETWORK_LATENCY_S,
                        100 * self.MEAN_NETWORK_LATENCY_S,
                        10 * self.MEAN_NETWORK_LATENCY_S,
                        self.MEAN_NETWORK_LATENCY_S,
                        self.MEAN_NETWORK_LATENCY_S / 10,
                        self.MEAN_NETWORK_LATENCY_S / 100]
    
        self.SIMULATOR = Simulator
        self.TO_DATAFRAME = to_dataframe3


class Default3fSimulationConfigs(object):
    def __init__(self):
    
        self.EXPERIMENT_IDENTIFIER = "3miners_toy_example"
        self.PERSIST_DATA = True
        self.DURATION = 2 * Durations.HOUR
        self.NUM_MINERS = 8
        self.SEED = 0
        self.MEAN_NETWORK_LATENCY_S = 1
        self.NUM_BLOCKS_TO_GENERATE = 100000
        self.NUM_ITER = 1
        self.NODES_IDS = ['OLM1', 'M1', 'OM1', 'OM', 'OM2', 'M2', 'ORM2', 'M3']
        self.HPD_CFG = [(0, .33, 0, 0, 0, .33, 0, .33)]
        self.HARDNESS_CFG = [1000 * self.MEAN_NETWORK_LATENCY_S,
                        100 * self.MEAN_NETWORK_LATENCY_S,
                        10 * self.MEAN_NETWORK_LATENCY_S,
                        self.MEAN_NETWORK_LATENCY_S,
                        self.MEAN_NETWORK_LATENCY_S / 10,
                        self.MEAN_NETWORK_LATENCY_S / 100]
        self.LATENCY_ADJACECY_MATRIX = prepare3_lat_mat_farther(self.NODES_IDS)
        self.SIMULATOR = Simulator
        self.TO_DATAFRAME = to_dataframe3


class Default5SimulationConfigs(object):
    def __init__(self):
    
        '''
        # <location_1> <lat_1> <lng_1> <location_2> <lat_2> <lng_2> <dist. (in km)> <latency (in ms)>
            WASHINGTON-DC-US 38.9047 -77.0164 SICHUAN-NA-CN 30.1333 102.9333 12338.40 197.41
            WASHINGTON-DC-US 38.9047 -77.0164 REYKJAVÍK-NA-IS 64.1333 -21.9333 4512.89 72.21
            WASHINGTON-DC-US 38.9047 -77.0164 LINTHAL-NA-CH 46.9167 9.0000 6703.91 107.26
            WASHINGTON-DC-US 38.9047 -77.0164 MOSCOW-NA-RU 55.7500 37.6167 7820.54 125.13
            SICHUAN-NA-CN 30.1333 102.9333 REYKJAVÍK-NA-IS 64.1333 -21.9333 8489.56 135.83
            SICHUAN-NA-CN 30.1333 102.9333 LINTHAL-NA-CH 46.9167 9.0000 7891.06 126.26
            SICHUAN-NA-CN 30.1333 102.9333 MOSCOW-NA-RU 55.7500 37.6167 5761.37 92.18
            REYKJAVÍK-NA-IS 64.1333 -21.9333 LINTHAL-NA-CH 46.9167 9.0000 2680.24 42.88
            REYKJAVÍK-NA-IS 64.1333 -21.9333 MOSCOW-NA-RU 55.7500 37.6167 3307.89 52.93
            LINTHAL-NA-CH 46.9167 9.0000 MOSCOW-NA-RU 55.7500 37.61672196.05 35.14
    
        https://datalight.me/blog/researches/infographics/datalight-publishes-a-list-of-countries-with-the-largest-number-of-bitcoin-nodes/
        np.array([2625, 411, 698, 159, 276])/4169
        array([0.6296474 , 0.09858479, 0.16742624, 0.03813864, 0.06620293])
    
        '''
        self.EXPERIMENT_IDENTIFIER = "5miners_paper"
        self.PERSIST_DATA = True
        self.DURATION = 2 * Durations.HOUR
        self.NUM_MINERS = 5
        self.SEED = 0
        self.NUM_BLOCKS_TO_GENERATE = 100000
        self.NUM_ITER = 10
        self.NODES_IDS = ['WA-US', 'SI-CN', 'RE-IS', 'LI-CH', 'MO-RU']
        self.NODES_IDS = ['WASHINGTON-DC-US', 'SICHUAN-NA-CN', 'REYKJAVÍK-NA-IS',
                        'LINTHAL-NA-CH', 'MOSCOW-NA-RU']
        self.HPD_CFG = [(0.2, 0.2, 0.2, 0.2, 0.2),
                   (0.05, 0.8, 0.07, 0.03, 0.05),
                   (0.62, 0.1, 0.17, 0.04, 0.07),
                   ]
        self.LATENCY_ADJACECY_MATRIX = prepare5_lat_mat_fixed(self.NODES_IDS)
        self.MEAN_NETWORK_LATENCY_S = np.mean(self.LATENCY_ADJACECY_MATRIX)
        self.HARDNESS_CFG = [10000 * self.MEAN_NETWORK_LATENCY_S,
                            1000 * self.MEAN_NETWORK_LATENCY_S,
                            100 * self.MEAN_NETWORK_LATENCY_S,
                            10 * self.MEAN_NETWORK_LATENCY_S,
                            self.MEAN_NETWORK_LATENCY_S,
                            self.MEAN_NETWORK_LATENCY_S / 10,
                            self.MEAN_NETWORK_LATENCY_S / 100
                        ]


        self.SIMULATOR = Simulator
        self.TO_DATAFRAME = to_dataframe5

class Default5EthereumSimulationConfigs(object):
    def __init__(self):
        self.EXPERIMENT_IDENTIFIER = "5miners_ethereum_paper"
        self.PERSIST_DATA = True
        self.DURATION = 2 * Durations.HOUR
        self.NUM_MINERS = 5
        self.SEED = 0
        self.NUM_BLOCKS_TO_GENERATE = 100000
        self.NUM_ITER = 10
        self.NODES_IDS = ['WA-US', 'SI-CN', 'RE-IS', 'LI-CH', 'MO-RU']
        self.NODES_IDS = ['WASHINGTON-DC-US', 'SICHUAN-NA-CN', 'REYKJAVÍK-NA-IS',
                        'LINTHAL-NA-CH', 'MOSCOW-NA-RU']
        self.HPD_CFG = [(0.2, 0.2, 0.2, 0.2, 0.2),
                   (0.25, 0.25, 0.12, 0.11, 0.05),
                   (0.25, 0.25, 0.05, 0.11, 0.12),
                   (0.05, 0.25, 0.12, 0.11, 0.25),
                   (0.25, 0.05, 0.12, 0.11, 0.25),
                   (0.05, 0.11, 0.25, 0.25, 0.12),
                   ]
        self.LATENCY_ADJACECY_MATRIX = prepare5_lat_mat_fixed(self.NODES_IDS)
        self.MEAN_NETWORK_LATENCY_S = np.mean(self.LATENCY_ADJACECY_MATRIX)
        self.HARDNESS_CFG = [
#                            10000 * self.MEAN_NETWORK_LATENCY_S,
                            1000 * self.MEAN_NETWORK_LATENCY_S,
                            100 * self.MEAN_NETWORK_LATENCY_S,
                            10 * self.MEAN_NETWORK_LATENCY_S,
                            self.MEAN_NETWORK_LATENCY_S,
                            self.MEAN_NETWORK_LATENCY_S / 10,
#                            self.MEAN_NETWORK_LATENCY_S / 100
                        ]


        self.SIMULATOR = Simulator
        self.TO_DATAFRAME = to_dataframe5


class Default15SimulationConfigs(object):
    def __init__(self):
    
        self.EXPERIMENT_IDENTIFIER = "15miners_observers_paper_"
        self.PERSIST_DATA = True
        self.DURATION = 2 * Durations.HOUR
        self.NUM_MINERS = 15
        self.SEED = 0
        self.NUM_BLOCKS_TO_GENERATE = 100000
        self.NUM_ITER = 10
        self.NODES_IDS = ['WASHINGTON-DC-US', 'SICHUAN-NA-CN', 'REYKJAVÍK-NA-IS',
                     'LINTHAL-NA-CH', 'MOSCOW-NA-RU', 'TBILISI-NA-GE', 'KIEV-NA-UK',
                     'ANKARA-NA-TR', 'SKOPJE-NA-MK', 'HELSINKI-NA-FI', 'MANNHEIM-BW-DE',
                     'SINGAPORE-NA-SG', 'ASHBURN-VA-US', 'FRANKFURT-HE-DE', 'NUREMBURG-BV-DE']
        self.LATENCY_ADJACECY_MATRIX = prepare15_lat_mat_fixed(self.NODES_IDS)
        self.MEAN_NETWORK_LATENCY_S = np.mean(self.LATENCY_ADJACECY_MATRIX[0:5][0:5])
    
        self.HPD_CFG = [(0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                   (0.05, 0.8, 0.07, 0.03, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                   (0.62, 0.1, 0.17, 0.04, 0.07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                   (0.05, 0.8, 0.05, 0.05, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                   (0.04, 0.85, 0.04, 0.03, 0.04, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                   ]
    
        self.HARDNESS_CFG = [10000 * self.MEAN_NETWORK_LATENCY_S,
                        1000 * self.MEAN_NETWORK_LATENCY_S,
                        100 * self.MEAN_NETWORK_LATENCY_S,
                        10 * self.MEAN_NETWORK_LATENCY_S,
                        self.MEAN_NETWORK_LATENCY_S,
                        self.MEAN_NETWORK_LATENCY_S / 10,
                        self.MEAN_NETWORK_LATENCY_S / 100,
                        17, 600]
    
        self.SIMULATOR = Simulator
        self.TO_DATAFRAME = to_dataframe15


class Default15LSSimulationConfigs(object):
    def __init__(self):
    
        self.EXPERIMENT_IDENTIFIER = "15LSminers_observers_paper_"
        self.PERSIST_DATA = True
        self.DURATION = 2 * Durations.HOUR
        self.NUM_MINERS = 15
        self.SEED = 0
        self.NUM_BLOCKS_TO_GENERATE = 100000
        self.NUM_ITER = 10
    
        self.NODES_IDS = ['WASHINGTON-DC-US', 'SICHUAN-NA-CN', 'REYKJAVÍK-NA-IS',
                     'LINTHAL-NA-CH', 'MOSCOW-NA-RU', 'TBILISI-NA-GE', 'KIEV-NA-UK',
                     'ANKARA-NA-TR', 'SKOPJE-NA-MK', 'HELSINKI-NA-FI', 'MANNHEIM-BW-DE',
                     'SINGAPORE-NA-SG', 'ASHBURN-VA-US', 'FRANKFURT-HE-DE', 'NUREMBURG-BV-DE']
        self.LATENCY_ADJACECY_MATRIX = prepare15_ls_lat_mat_fixed(self.NODES_IDS)
        self.MEAN_NETWORK_LATENCY_S = np.mean(self.LATENCY_ADJACECY_MATRIX)
    
        self.HPD_CFG = [(0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                   (0.05, 0.8, 0.07, 0.03, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                   (0.62, 0.1, 0.17, 0.04, 0.07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                   (0.05, 0.8, 0.05, 0.05, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                   (0.04, 0.85, 0.04, 0.03, 0.04, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                   ]
        self.HARDNESS_CFG = [10000 * self.MEAN_NETWORK_LATENCY_S,
                        1000 * self.MEAN_NETWORK_LATENCY_S,
                        100 * self.MEAN_NETWORK_LATENCY_S,
                        10 * self.MEAN_NETWORK_LATENCY_S,
                        self.MEAN_NETWORK_LATENCY_S,
                        self.MEAN_NETWORK_LATENCY_S / 10,
                        self.MEAN_NETWORK_LATENCY_S / 100,
                        17, 600]
        #self.HARDNESS_CFG = [17, 600]
    
        self.SIMULATOR = Simulator
        self.TO_DATAFRAME = to_dataframe15


class Default15CSSimulationConfigs(object):
    def __init__(self):
    
        self.EXPERIMENT_IDENTIFIER = "15CSminers_observers_paper_"
        self.PERSIST_DATA = True
        self.DURATION = 2 * Durations.HOUR
        self.NUM_MINERS = 15
        self.SEED = 0
        self.NUM_BLOCKS_TO_GENERATE = 100000
        self.NUM_ITER = 10
    
        self.NODES_IDS = ['WASHINGTON-DC-US', 'SICHUAN-NA-CN', 'REYKJAVÍK-NA-IS',
                     'LINTHAL-NA-CH', 'MOSCOW-NA-RU', 'TBILISI-NA-GE', 'KIEV-NA-UK',
                     'ANKARA-NA-TR', 'SKOPJE-NA-MK', 'HELSINKI-NA-FI', 'MANNHEIM-BW-DE',
                     'SINGAPORE-NA-SG', 'ASHBURN-VA-US', 'FRANKFURT-HE-DE', 'NUREMBURG-BV-DE']
        self.LATENCY_ADJACECY_MATRIX = prepare15_cs_lat_mat_fixed(self.NODES_IDS)
        self.MEAN_NETWORK_LATENCY_S = np.mean(self.LATENCY_ADJACECY_MATRIX)
    
        self.HPD_CFG = [(0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                   (0.05, 0.8, 0.07, 0.03, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                   (0.62, 0.1, 0.17, 0.04, 0.07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                   (0.05, 0.8, 0.05, 0.05, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                   (0.04, 0.85, 0.04, 0.03, 0.04, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                   ]
        self.HARDNESS_CFG = [10000 * self.MEAN_NETWORK_LATENCY_S,
                        1000 * self.MEAN_NETWORK_LATENCY_S,
                        100 * self.MEAN_NETWORK_LATENCY_S,
                        10 * self.MEAN_NETWORK_LATENCY_S,
                        self.MEAN_NETWORK_LATENCY_S,
                        self.MEAN_NETWORK_LATENCY_S / 10,
                        self.MEAN_NETWORK_LATENCY_S / 100,
                        17, 600]
        #self.HARDNESS_CFG = [17, 600]
    
        self.SIMULATOR = Simulator
        self.TO_DATAFRAME = to_dataframe15


class DefaultCentralitySimulationConfigs(object):
    def __init__(self):
    
        self.EXPERIMENT_IDENTIFIER = "centrality_miners_observers_paper_" 
        self.PERSIST_DATA = True
        self.DURATION = 2 * Durations.HOUR
        self.NUM_MINERS = 100
        self.SEED = 33
    
        self.NUM_BLOCKS_TO_GENERATE = 100000
        self.NUM_ITER = 10
    
        cities = pd.read_csv('evaluation/100_cities_lat_lng.txt', delim_whitespace=True)
        self.NODES_IDS = cities['city'].tolist()
        # print('centrality_miners_observers_paper')
        # print(self.NODES_IDS)
        self.HPD_CFG = [tuple(1 for i in range(0, 100))]
        self.LATENCY_ADJACECY_MATRIX = prepare100_lat_mat_fixed_centrality(self.NODES_IDS)
        self.MEAN_NETWORK_LATENCY_S = np.mean(self.LATENCY_ADJACECY_MATRIX)
    
        self.HARDNESS_CFG = [17, 600]
    
        self.SIMULATOR = Simulator
        self.TO_DATAFRAME = to_dataframe15


class DefaultCapitalsCentralitySimulationConfigs(object):
    def __init__(self):
    
        self.EXPERIMENT_IDENTIFIER = "capitals_centrality_miners_observers_paper_" 
        self.PERSIST_DATA = True
        self.DURATION = 2 * Durations.HOUR
        self.NUM_MINERS = 240
        self.SEED = 33
    
        self.NUM_BLOCKS_TO_GENERATE = 100000
        self.NUM_ITER = 10
        cities = pd.read_csv('evaluation/cities_capitals_lat_lng.txt', delim_whitespace=True)
        self.NODES_IDS = cities['city'].tolist()
        # print(self.NODES_IDS)
        self.HPD_CFG = [tuple(1 for i in range(0, 240))]
    
    
        self.LATENCY_ADJACECY_MATRIX = prepare240_lat_mat_fixed_capital_centrality(self.NODES_IDS)
        self.MEAN_NETWORK_LATENCY_S = np.mean(self.LATENCY_ADJACECY_MATRIX)
        
        self.HARDNESS_CFG = [self.MEAN_NETWORK_LATENCY_S/10.0]
    
        self.SIMULATOR = Simulator
        self.TO_DATAFRAME = to_dataframe15


class DefaultCapitalsCentralityLSSimulationConfigs(object):
    def __init__(self):
    
        self.EXPERIMENT_IDENTIFIER = "capitals_centrality_ls_miners_observers_paper_"
        self.PERSIST_DATA = True
        self.DURATION = 2 * Durations.HOUR
        self.NUM_MINERS = 240
        self.SEED = 33
    
        self.NUM_BLOCKS_TO_GENERATE = 100000
        self.NUM_ITER = 10
    
        cities = pd.read_csv('evaluation/cities_capitals_lat_lng.txt', delim_whitespace=True)
        self.NODES_IDS = cities['city'].tolist()
        # print(self.NODES_IDS)
        self.HPD_CFG = [tuple(1 for i in range(0, 240))]
    
        self.LATENCY_ADJACECY_MATRIX = prepare15_lat_mat_ls_fixed_capital_centrality(
            self.NODES_IDS)
        self.MEAN_NETWORK_LATENCY_S = np.mean(self.LATENCY_ADJACECY_MATRIX)
    
        #self.HARDNESS_CFG = [600]
        self.HARDNESS_CFG = [17, 600]
    
    
        self.SIMULATOR = Simulator
        self.TO_DATAFRAME = to_dataframe15


class DefaultCapitalsCentralityCSSimulationConfigs(object):
    def __init__(self):
    
        self.EXPERIMENT_IDENTIFIER = "capitals_centrality_cs_miners_observers_paper_" 
        self.PERSIST_DATA = True
        self.DURATION = 2 * Durations.HOUR
        self.NUM_MINERS = 240
        self.SEED = 33
    
        self.NUM_BLOCKS_TO_GENERATE = 100000
        self.NUM_ITER = 10
    
        cities = pd.read_csv('evaluation/cities_capitals_lat_lng.txt', delim_whitespace=True)
        self.NODES_IDS = cities['city'].tolist()
        # print(self.NODES_IDS)
        self.HPD_CFG = [tuple(1 for i in range(0, 240))]
    
        self.LATENCY_ADJACECY_MATRIX = prepare240_lat_mat_cs_fixed_capital_centrality(
            self.NODES_IDS)
        self.MEAN_NETWORK_LATENCY_S = np.mean(self.LATENCY_ADJACECY_MATRIX)
    
        #self.HARDNESS_CFG = [600]
        self.HARDNESS_CFG = [17, 600]
    
    
        self.SIMULATOR = Simulator
        self.TO_DATAFRAME = to_dataframe15


class DefaultAttackSimulationConfigs(object):
    def __init__(self):
    
        self.EXPERIMENT_IDENTIFIER = "attack_miners_observers_paper"
        self.PERSIST_DATA = True
        self.DURATION = 2 * Durations.HOUR
        self.NUM_MINERS = 15
        self.SEED = 1
    
        self.NUM_BLOCKS_TO_GENERATE = 100000
        self.NUM_ITER = 10
    
        self.NODES_IDS = ['WASHINGTON-DC-US', 'SICHUAN-NA-CN', 'REYKJAVÍK-NA-IS',
                     'LINTHAL-NA-CH', 'MOSCOW-NA-RU', 'TBILISI-NA-GE', 'KIEV-NA-UK',
                     'ANKARA-NA-TR', 'SKOPJE-NA-MK', 'HELSINKI-NA-FI', 'MANNHEIM-BW-DE',
                     'SINGAPORE-NA-SG', 'ASHBURN-VA-US', 'FRANKFURT-HE-DE', 'NUREMBURG-BV-DE']
        self.HPD_CFG = [  # (0.049, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.05),
            (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            #                (1, 1, 1, 1, 1, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0),
            #              (0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.02, 0.02, 0.25, 0.1),
            #              (0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.45, 0.04)
        ]
        self.LATENCY_ADJACECY_MATRIX = prepare15_lat_mat_fixed(self.NODES_IDS)
        self.MEAN_NETWORK_LATENCY_S = np.mean(self.LATENCY_ADJACECY_MATRIX)/1000.0
    
        self.HARDNESS_CFG = [600]
    #    self.HARDNESS_CFG = [1000*self.MEAN_NETWORK_LATENCY_S, 600]
    
        self.SIMULATOR = Simulator
        self.TO_DATAFRAME = to_dataframe15


class RealLifeExpSimulationConfigs(object):
    def __init__(self):
    
        self.EXPERIMENT_IDENTIFIER = "top_miners_bitcoin_exp"
        self.PERSIST_DATA = True
        self.DURATION = 2 * Durations.HOUR
        self.NUM_MINERS = 19
        self.SEED = 0
        self.MEAN_NETWORK_LATENCY_S = 0.3
        self.NUM_BLOCKS_TO_GENERATE = 1000
        self.NUM_ITER = 2
        self.NODES_IDS = ["BTC.com",
                     "AntPool",
                     "F2Pool",
                     "SlushPool",
                     "Poolin",
                     "ViaBTC",
                     "BTC.TOP",
                     "unknown",
                     "Huobi.pool",
                     "BitFury",
                     "BitClub",
                     "1M1X",
                     "Bitcoin.com",
                     "DPOOL",
                     "WAYI.CN",
                     "Bixin",
                     "tigerpool.net",
                     "KanoPool",
                     "BitcoinRussia"]
        self.HPD_CFG = [(17.19,
                    12.40,
                    11.35,
                    9.90,
                    9.68,
                    7.05,
                    6.85,
                    6.27,
                    4.15,
                    4.06,
                    2.65,
                    1.96,
                    1.96,
                    1.81,
                    1.25,
                    1.00,
                    0.38,
                    0.09,
                    0.02)]
        self.HARDNESS_CFG = [1000 * self.MEAN_NETWORK_LATENCY_S, 100 * self.MEAN_NETWORK_LATENCY_S, 10 * self.MEAN_NETWORK_LATENCY_S,
                        1 * self.MEAN_NETWORK_LATENCY_S, .1 * self.MEAN_NETWORK_LATENCY_S]
        lat_mat = Exp_LatD(self.NUM_MINERS, self.MEAN_NETWORK_LATENCY_S, 0).tolist()
        self.LATENCY_ADJACECY_MATRIX = lat_mat
        self.MEAN_NETWORK_LATENCY_S = np.mean(self.LATENCY_ADJACECY_MATRIX)/1000.0
    
        self.SIMULATOR = Simulator
        self.TO_DATAFRAME = to_dataframe_real_bc


class RealLifeEquSimulationConfigs(object):
    def __init__(self):
    
        self.EXPERIMENT_IDENTIFIER = "top_miners_bitcoin_equ"
        self.PERSIST_DATA = True
        self.DURATION = 2 * Durations.HOUR
        self.NUM_MINERS = 19
        self.SEED = 0
        self.MEAN_NETWORK_LATENCY_S = 0.3
        self.NUM_BLOCKS_TO_GENERATE = 1000
        self.NUM_ITER = 2
        self.NODES_IDS = ["BTC.com",
                     "AntPool",
                     "F2Pool",
                     "SlushPool",
                     "Poolin",
                     "ViaBTC",
                     "BTC.TOP",
                     "unknown",
                     "Huobi.pool",
                     "BitFury",
                     "BitClub",
                     "1M1X",
                     "Bitcoin.com",
                     "DPOOL",
                     "WAYI.CN",
                     "Bixin",
                     "tigerpool.net",
                     "KanoPool",
                     "BitcoinRussia"]
        self.HPD_CFG = [(17.19,
                    12.40,
                    11.35,
                    9.90,
                    9.68,
                    7.05,
                    6.85,
                    6.27,
                    4.15,
                    4.06,
                    2.65,
                    1.96,
                    1.96,
                    1.81,
                    1.25,
                    1.00,
                    0.38,
                    0.09,
                    0.02)]
        self.HARDNESS_CFG = [1000 * self.MEAN_NETWORK_LATENCY_S, 100 * self.MEAN_NETWORK_LATENCY_S, 10 * self.MEAN_NETWORK_LATENCY_S,
                        1 * self.MEAN_NETWORK_LATENCY_S, .1 * self.MEAN_NETWORK_LATENCY_S]
        lat_mat = Equ_LatD(self.NUM_MINERS, self.MEAN_NETWORK_LATENCY_S, 0).tolist()
        self.LATENCY_ADJACECY_MATRIX = lat_mat
        self.MEAN_NETWORK_LATENCY_S = np.mean(self.LATENCY_ADJACECY_MATRIX)/1000.0
    
        self.SIMULATOR = Simulator
        self.TO_DATAFRAME = to_dataframe_real_bc


class RealLifeExpPooledSimulationConfigs(object):
    def __init__(self):
    
        self.EXPERIMENT_IDENTIFIER = "top_miners_bitcoin_exp_pooled"
        self.PERSIST_DATA = True
        self.DURATION = 2 * Durations.HOUR
        self.NUM_MINERS = 19
        self.SEED = 0
        self.MEAN_NETWORK_LATENCY_S = 0.3
        self.NUM_BLOCKS_TO_GENERATE = 10000
        self.NUM_ITER = 1
        NODES_NAMES = ["BTC.com",
                       "AntPool",
                       "F2Pool",
                       "SlushPool",
                       "Poolin",
                       "ViaBTC",
                       "BTC.TOP",
                       "unknown",
                       "Huobi.pool",
                       "BitFury",
                       "BitClub",
                       "1M1X",
                       "Bitcoin.com",
                       "DPOOL",
                       "WAYI.CN",
                       "Bixin"]
        self.HPD_CFG = [(17,
                    12,
                    11,
                    10,
                    10,
                    7,
                    7,
                    6,
                    4,
                    4,
                    3,
                    2,
                    2,
                    2,
                    1,
                    1)]
        self.NUM_MINERS = sum(self.HPD_CFG[0])
    
        pools = self.HPD_CFG[0]
        self.HPD_CFG = [tuple([(1)] * self.NUM_MINERS)]
    
        self.NODES_IDS = []
        for i in range(len(NODES_NAMES)):
            for j in range(pools[i]):
                self.NODES_IDS.append(NODES_NAMES[i] + "_" + str(j))
    
        self.HARDNESS_CFG = [1000 * self.MEAN_NETWORK_LATENCY_S, 100 * self.MEAN_NETWORK_LATENCY_S, 10 * self.MEAN_NETWORK_LATENCY_S,
                        1 * self.MEAN_NETWORK_LATENCY_S, .1 * self.MEAN_NETWORK_LATENCY_S]
        lat_mat = Exp_pooled_LatD(
            self.NUM_MINERS, self.MEAN_NETWORK_LATENCY_S, pools).tolist()
        self.LATENCY_ADJACECY_MATRIX = lat_mat
        self.SIMULATOR = Simulator
        self.TO_DATAFRAME = to_dataframe_real_bc


class RealLifeEquPooledSimulationConfigs(object):
    def __init__(self):
        self.EXPERIMENT_IDENTIFIER = "top_miners_bitcoin_equ_pooled"
        self.PERSIST_DATA = True
        self.DURATION = 2 * Durations.HOUR
        self.NUM_MINERS = 19
        self.SEED = 0
        self.MEAN_NETWORK_LATENCY_S = 0.3
        self.NUM_BLOCKS_TO_GENERATE = 10000
        self.NUM_ITER = 1
        NODES_NAMES = ["BTC.com",
                       "AntPool",
                       "F2Pool",
                       "SlushPool",
                       "Poolin",
                       "ViaBTC",
                       "BTC.TOP",
                       "unknown",
                       "Huobi.pool",
                       "BitFury",
                       "BitClub",
                       "1M1X",
                       "Bitcoin.com",
                       "DPOOL",
                       "WAYI.CN",
                       "Bixin"]
        self.HPD_CFG = [(17,
                    12,
                    11,
                    10,
                    10,
                    7,
                    7,
                    6,
                    4,
                    4,
                    3,
                    2,
                    2,
                    2,
                    1,
                    1)]
        self.NUM_MINERS = sum(self.HPD_CFG[0])
        pools = self.HPD_CFG[0]
        self.HPD_CFG = [tuple([(1)] * self.NUM_MINERS)]
        self.NODES_IDS = []
        for i in range(len(NODES_NAMES)):
            for j in range(pools[i]):
                self.NODES_IDS.append(NODES_NAMES[i] + "_" + str(j))
    
        self.HARDNESS_CFG = [1000 * self.MEAN_NETWORK_LATENCY_S, 100 * self.MEAN_NETWORK_LATENCY_S, 10 * self.MEAN_NETWORK_LATENCY_S,
                        1 * self.MEAN_NETWORK_LATENCY_S, .1 * self.MEAN_NETWORK_LATENCY_S]
        lat_mat = Equ_pooled_LatD(
            self.NUM_MINERS, self.MEAN_NETWORK_LATENCY_S, pools).tolist()
        self.LATENCY_ADJACECY_MATRIX = lat_mat
        self.SIMULATOR = Simulator
        self.TO_DATAFRAME = to_dataframe_real_bc


class DefaultSimulationConfigs(Default2SimulationConfigs):
    pass
