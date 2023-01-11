import glob
from obspy import read
from obspy.core.event import read_events
from obspy.clients.fdsn import Client
from obspy import Catalog, UTCDateTime

import numpy as np
import os

EventListName = "Glacial"
# EventListName = "Earthquakes"

if EventListName == "Glacial":
    catalog = dict()
    with open("./GLEA_1993_2013_merged.txt", 'r') as f:
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

    CatalogFILE = 'Glacial_catalog.txt'
elif EventListName == "Earthquakes":
    CatalogFILE = 'Earthquakes_catalog.txt'
    cat = read_events("LearningEQs.xml")

with open(CatalogFILE, 'w+') as f:
    line = '#################### This is Catalog File ####################\n'
    f.write(line)



DataDir = './%s/' %EventListName
EventPathList = glob.glob(DataDir + '*.mseed')

StationFILE = 'Stations_FULL.txt'

GSNStationList = []
with open(StationFILE, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        GSNStationList.append(line.split())

DataArray = np.full((len(GSNStationList),7200),0.0)

for EventPath in EventPathList:
    EventName = EventPath.split('/')[-1][0:-6]
    DataStream = read(EventPath,format='MSEED')
    DataStream.filter('bandpass', freqmin=1/150.,freqmax=1/20.)

    # DataStream.plot()

    for iGSNStation, GSNStation in enumerate(GSNStationList):
        SelectTrace = DataStream.select(id='%s.%s*LHZ' %(GSNStation[0], GSNStation[1]))

        if len(SelectTrace) < 1:
            continue
        
        Num = min(len(SelectTrace[0].data),7200)
        if abs(SelectTrace[0].data[0:Num]).max() == 0:
            print("max is 0!!!!")
            continue
        DataArray[iGSNStation][0:Num] = SelectTrace[0].data[0:Num]/(abs(SelectTrace[0].data[0:Num]).max())

    BinaryFileName = '%s%s.bin' %(DataDir, EventName)
    # file = open(BinaryFileName,'wb')
    # file.write(DataArray.tobytes())

    DataArray.tofile(BinaryFileName)
    print(DataArray.min(),DataArray.max())
    print(BinaryFileName, " File wriiten !!")

    if np.isnan(DataArray.max()):
        print("error")
        os.remove(BinaryFileName)
        os.remove(EventPath)        
    # DataArrayRead = np.fromfile(BinaryFileName,dtype='float64').reshape(129,7200)

    if EventListName == "Glacial":
        with open(CatalogFILE, 'a+') as f:
            line = catalog[EventName]['Entry']
            f.write(line)
    elif EventListName == "Earthquakes":
        currentcat = cat.filter("time > %s"%(DataStream[0].stats.starttime -30), "time < %s"%(DataStream[0].stats.starttime + 30))
        if len(currentcat) !=1:
            os.remove(BinaryFileName)
            os.remove(EventPath)
            continue

        EvtDep = currentcat[0].origins[0]['depth']/1.e3
        with open(CatalogFILE, 'a+') as f:
            line = currentcat.__str__().split('\n')[1] + ' | %.1f km'%EvtDep +'\n'
            f.write(line)
     