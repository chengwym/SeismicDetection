import glob
from obspy import read
from obspy.clients.fdsn import Client

# Make FULL station list for Earthquakes & Glacial events
EventListName = "FULL"

# DataDir = './%s/' %EventListName
EventPathList1 = glob.glob('./Glacial/' + '*.mseed')
EventPathList2 = glob.glob('./Earthquakes/' + '*.mseed')

# EventPathList = EventPathList1.extend(EventPathList2)

StationFILE = 'Stations_%s.txt' %EventListName
with open(StationFILE, 'w+') as f:
    line = '#################### This is Station File ####################\n## %-10s %-10s %-10s %-10s\n' %('Network','Station','Latitude','Longitude')
    f.write(line)

irisclient = Client("IRIS") 

GSNStationList = []
for EventPath in EventPathList1:
    DataStream = read(EventPath,format='MSEED')

    for Trace in DataStream:
        NetworkCode = Trace.id.split('.')[0]
        StationCode = Trace.id.split('.')[1]

        EvtStartTime = Trace.stats.starttime
        EvtEndTime = EvtStartTime + 7200.

        Code = NetworkCode + '.' + StationCode

        if Code in GSNStationList:
            continue
        else:
            GSNStationList.append(Code)
            inventory = irisclient.get_stations(network=NetworkCode, station=StationCode, starttime=EvtStartTime, endtime=EvtEndTime)

            with open(StationFILE, 'a+') as f:
                line = '%-10s %-10s %-10.2f %-10.2f\n' %(NetworkCode,StationCode,inventory[0][0].latitude,inventory[0][0].longitude)
                f.write(line)


