#!/raid1/zl382/Libs/anaconda3/bin/python3

import obspy
from obspy import read
from obspy.core import Stream
from obspy.clients.fdsn import Client
from obspy import Catalog, UTCDateTime
import matplotlib.pyplot as plt
import os.path
import time
from obspy.geodetics import kilometers2degrees
from geographiclib.geodesic import Geodesic
import numpy as np
import sys
import shutil
import requests
from bs4 import BeautifulSoup
from multiprocessing import Pool

EventListName = "Glacial"

nproc = 20

fl1 = 1/200.
fl2 = 1/150.
fl3 = 1/20.
fl4 = 1/10.

catalog = dict()
with open("/raid1/Codes/LearningEQs/GLEA_1993_2013_merged.txt", 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        EventName = line.split()[-1]

        catalog[EventName] = dict()
        catalog[EventName]['Entry'] = line
        catalog[EventName]['CentroidLat'] = float(line.split()[0])
        catalog[EventName]['CentroidLon'] = float(line.split()[1])
        catalog[EventName]['Amplitude'] = float(line.split()[2])

        CentroidLat = float(line.split()[0])
        CentroidLon = float(line.split()[1])
        Amplitude = float(line.split()[2])
        catalog[EventName]['ForceAzimuth'] = float(line.split()[3])
        catalog[EventName]['CentroidTimeshift'] = float(line.split()[4])
        catalog[EventName]['Year'] = int(line.split()[5])
        catalog[EventName]['Month'] = int(line.split()[6])
        catalog[EventName]['day'] = int(line.split()[7])
        catalog[EventName]['hr'] = int(line.split()[8])
        catalog[EventName]['mn'] = int(line.split()[9])
        catalog[EventName]['sc'] = float(line.split()[10])
        catalog[EventName]['Magnitude'] = float(line.split()[-4])
        catalog[EventName]['Region'] = line.split()[-3]
        catalog[EventName]['DetectionType'] = int(line.split()[-2])
        catalog[EventName]['EventName'] = line.split()[-1]
        catalog[EventName]['OriginTime'] = UTCDateTime(catalog[EventName]['Year'], catalog[EventName]['Month'], catalog[EventName]['day'],
                                                       catalog[EventName]['hr'], catalog[EventName]['mn'],catalog[EventName]['sc']) + catalog[EventName]['CentroidTimeshift']


print(len(catalog), "glacial events found in catalog.") 

# with open('FIND_catalog_%s.txt' %EventListName, 'w+') as f:
#     f.write(cat.__str__(print_all=True))          

with open('FIND_OUTPUT_%s.txt' %EventListName, 'w+') as f:
    line = '%%%%%%%%%%%%%%%%%% This is the Event Output Result %%%%%%%%%%%%%%%%%%%%%%\n'
    f.write(line)

EventDir='/raid1/Data/LearningEQs/%s/' %(EventListName)
# Make directories for data
if not os.path.exists(EventDir):
    os.makedirs(EventDir)
    print(EventDir)

# load IRIS client
irisclient = Client("IRIS") 

def DownloadEvent(EventKey):
# for EVENT in cat:

    EventName = catalog[EventKey]['EventName']
    PathName = '%s%s.mseed' %(EventDir, EventName)
    EvtStartTime = catalog[EventName]['OriginTime']
    EvtEndTime = EvtStartTime + 7200.


    count = 0
    failed_waveform_count = 0
    failed_response_count = 0
    for clientname in ["IRIS"]:

        if os.path.exists(PathName):
            print(EventName, " already exists")
            DataStream = read(PathName,format='MSEED')
        else:
            DataStream = Stream()

        # Select what stations are present at the time
        try:
            inventory = irisclient.get_stations(network="IU,II", starttime=EvtStartTime, endtime=EvtEndTime)
        except:
            print("Failed get stations inventory")
            continue
        
        for network in inventory:
            for station in network:

                if len(DataStream.select(id='%s.%s.*' %(network.code, station.code))) > 0:
                    continue
                else:                

                    try:
                        st = irisclient.get_waveforms(network.code, station.code, "*", "LHZ", EvtStartTime, EvtEndTime,attach_response=True)
                    except:
                        with open('FIND_OUTPUT_%s.txt' %EventListName, 'a+') as f:
                            line = 'Failed client get waveform %s %s!!\n' %(network.code, station.code)
                            failed_waveform_count += 1
                        print(line)
                        continue

                    try:
                        st.remove_response(output="DISP", pre_filt=[fl1,fl2,fl3,fl4])
                    except:
                        with open('FIND_OUTPUT_%s.txt' %EventListName, 'a+') as f:
                            line = 'Failed remove response %s %s!!\n' %(network.code, station.code)
                            failed_response_count += 1
                        print(line)
                        continue

                    DataStream += st
                    count += 1

    if len(DataStream)>=10: # delete the Event that has less 10 stations

        with open('FIND_OUTPUT_%s.txt' %EventListName, 'a+') as f:
            line = "****%d failed waveform request\n****%d failed response remove\n%d (new +%d) waveform found in GSN network at %s\n" %(failed_waveform_count, failed_response_count, len(DataStream), count, EventName)
            f.write(line)
        # print(len(inventory.get_contents()['stations']), "found in GSN network at %s" %EventName)
        print(line)

        DataStream.write(PathName,format='MSEED',encoding='FLOAT64')
        with open('FIND_OUTPUT_%s.txt' %EventListName, 'a+') as f:
            line = "Waveform saved in %s\n" %PathName
            f.write(line)
        print(line)

    else:
        if os.path.exists(PathName):
            os.remove(PathName)


with Pool(nproc) as p:
    p.map(DownloadEvent,catalog)  # Multiprocessing DownloadEvent