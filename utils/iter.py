__author__ = "Tianyi Zhang"

import numpy as np

from obspy import read      ## Optional for mseed file reading with obspy module
from geographiclib.geodesic import Geodesic  ## Optional for calculating the distance between two geographic location points (with latitude and longitude) on Earth


StationLatLon = dict()
with open("./data/stations_full.txt", 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        NetworkCode = line.split()[0]
        StationCode = line.split()[1]

        StationLatLon[NetworkCode+'.'+StationCode] = [float(line.split()[2]),float(line.split()[3])]

dis = []
station_latlon = []

input_file = "sample.txt"
####
"station_name" "dis"
"station_name" "dis"
"station_name" "dis"
####

with open(input_file) as f:
    for line in f:
        StationName, dist = line.split(" ")
        station_latlon.append((StationLatLon[StationName][0],StationLatLon[StationName][1]))
        dis.append(float(dist))

dis = np.array(dis)
station_latlon = np.array(station_latlon)
target = np.array((0, 0))
ans = target
ansE = -1

#print(SourceLat, SourceLon)
with open("log.txt", "w") as log:
    log.write(f"input_file: {input_file}\n")

for iter in range(10000):
    F = np.array((0, 0))
    E = 0
    k = 4e-10
    drop_out = np.random.choice(range(129), size=80, replace=False)
    for i in drop_out:
        EventStationPair = Geodesic.WGS84.Inverse(*target,*station_latlon[i], outmask=1929)
        part_dis = EventStationPair['s12']
        F_part = k * (part_dis - dis[i]) * (station_latlon[i] - target)
        E += k * 0.5 * (part_dis - dis[i]) * (part_dis - dis[i])
        F = F + F_part
    if ansE == -1 or ansE > E:
        ans = target
        ansE = E
    if iter % 500 == 0:
        with open("log.txt", "a") as log:
            log.write(f"iter:{iter}, F:{F}, E:{E}, now_pos:{target}\n")
    if E < 100:
        if E < 10:
            target = target + F * 0.05
        else:
            target = target + F * 0.2
    else:
        target = target + F
print(ans)
with open("log.txt", "a") as log:
    log.write(f"ans:{ans}, ansE:{ansE}\n")
#print(ans, ansE)
#print(ans[0] - SourceLat, ans[1] - SourceLon)