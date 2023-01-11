#!/raid1/zl382/Libs/anaconda3/bin/python3

import obspy
from obspy import read
from obspy.core.event import read_events
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
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
from obspy.core import Stream

nproc = 20

EventListName = "Earthquakes"


MinDepth = 0  # avoid shallow events for clean stf
MaxDepth = 600
MinMagnitude = 5.0
MaxMagnitude = 5.5
MagnitudeType = 'MW'
OrderBy = 'time-asc'
StartTime = UTCDateTime("1993-01-01T00:00:00")
EndTime = UTCDateTime("2013-12-31T23:59:00")


# define a filter band to prevent amplifying noise during the deconvolution
# not being used, as responses are not being removed
fl1 = 1/200.
fl2 = 1/150.
fl3 = 1/20.
fl4 = 1/10.

# load IRIS client
irisclient = Client("IRIS") 

cat = read_events("LearningEQs.xml")
# cat = irisclient.get_events(starttime=StartTime, endtime=EndTime, mindepth=MinDepth, maxdepth=MaxDepth,
#                             minmagnitude=MinMagnitude, maxmagnitude=MaxMagnitude, magnitudetype=MagnitudeType, orderby=OrderBy) 

# cat.write("LearningEQs.xml", format="QUAKEML")


OUTPUTFILE = 'FIND_OUTPUT_%s.txt' %EventListName 
with open(OUTPUTFILE, 'w+') as f:
    line = '%%%%%%%%%%%%%%%%%% This is the Event Output Result %%%%%%%%%%%%%%%%%%%%%%\n'
    f.write(line)

with open('FIND_catalog_%s.txt' %EventListName, 'w+') as f:
    f.write(cat.__str__(print_all=True))  

CatalogFILE = 'Earthquake_catalog.txt'
with open(CatalogFILE, 'w+') as f:
    line = '%%%%%%%%%%%%%%%%%% This is the Earthquake Catalog %%%%%%%%%%%%%%%%%%%%%%\n'
    f.write(line)


def DownloadEvent(EVENT):
# for EVENT in cat:
    TimeString = str(EVENT.origins[0].time)
    EventName = TimeString[0:4] + TimeString[5:7] + TimeString[8:10]
    EventMag = EVENT.magnitudes[0]['mag']
    EvtLat = EVENT.origins[0]['latitude']
    EvtLon = EVENT.origins[0]['longitude']
    EvtDep = EVENT.origins[0]['depth']/1.e3 # convert to km from m
    EvtStartTime = EVENT.origins[0].time
    EvtEndTime = EvtStartTime + 7200.


    EventDir='./%s/' %(EventListName)
    FilePathName = '%s%s.mseed' %(EventDir, EventName)
    # Make directories for data
    if not os.path.exists(EventDir):
        # print("%s already exists" %EventDir)
        # EventDir = EventDir + 'A'
        os.makedirs(EventDir)
        print(EventDir, " created")
        # continue
    # else:
    #     print("%s already exists" %EventDir)


    if os.path.exists(FilePathName):
        print(EventName, "MSEED file already exists")
        DataStream = read(FilePathName,format='MSEED')
    else:
        DataStream = Stream()    

    count = 0
    failed_waveform_count = 0
    failed_response_count = 0 
    for clientname in ["IRIS"]:        
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
                        with open(OUTPUTFILE, 'a+') as f:
                            line = 'Failed client get waveform %s %s!!\n' %(network.code, station.code)
                            failed_waveform_count += 1
                        print(line)
                        continue

                    try:
                        st.remove_response(output="DISP", pre_filt=[fl1,fl2,fl3,fl4])
                    except:
                        with open(OUTPUTFILE, 'a+') as f:
                            line = 'Failed remove response %s %s!!\n' %(network.code, station.code)
                            failed_response_count += 1
                        print(line)
                        continue

                    DataStream += st
                    count += 1

    if len(DataStream)>=10: # delete the Event that has less 10 stations

        with open(OUTPUTFILE, 'a+') as f:
            line = "****%d failed waveform request\n****%d failed response remove\n%d (new +%d) waveform found in GSN network at %s\n" %(failed_waveform_count, failed_response_count, len(DataStream), count, EventName)
            f.write(line)
        # print(len(inventory.get_contents()['stations']), "found in GSN network at %s" %EventName)
        print(line)

        DataStream.write(FilePathName,format='MSEED',encoding='FLOAT64')

        with open(OUTPUTFILE, 'a+') as f:
            line = "Waveform saved in %s\n" %FilePathName
            f.write(line)
        print(line)

        with open(CatalogFILE, 'a+') as f:
            line = EVENT.__str__().split('\n')[0][7::] + ' | %.1f km'%EvtDep +'\n'
            f.write(line)
        print(line)        
    else:
        if os.path.exists(FilePathName):
            os.remove(FilePathName)


# space = len(cat)//248 # around 400

with Pool(nproc) as p:
    p.map(DownloadEvent,cat[0::30])  # Multiprocessing DownloadEvent
