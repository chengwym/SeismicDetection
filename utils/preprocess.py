__author__ = "Jingbo Cheng, Tianyi Zhang"

import os
from PIL import Image
from obspy import read
from geographiclib.geodesic import Geodesic

# EventListName = 'earthquakes'

earthquakes = os.listdir('../data/earthquakes')
glacial = os.listdir('../data/glacial')
earthquakes = [f'../data/earthquakes/{file}' for file in earthquakes]
glacial = [f'../data/glacial/{file}' for file in glacial]
paths = earthquakes + glacial

StationLatLon = {}
with open('../data/stations_full.txt', 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        NetworkCode = line.split()[0]
        StationCode = line.split()[1]

        StationLatLon[f'{NetworkCode}.{StationCode}'] = [float(line.split()[2]), float(line.split()[3])]

for index, path in enumerate(paths):
    DataStream = read(path, format='MSEED')
    words = path.split('/')
    EventListName = words[-2]
    FileName = words[-1].split('.')[0]
    StartString = str(DataStream[0].stats.starttime)[:-11]
    if EventListName == 'earthquakes':
        with open('../data/earthquakes_catalog.txt', 'r') as f:
            for line in f:
                if line.startswith(StartString):
                    SourceLat = float(line.split('|')[1].split(',')[0])
                    SourceLon = float(line.split('|')[1].split(',')[1])
                    depth = float(line.split('|')[3].split(' ')[1])
    elif EventListName == 'glacial':
        with open('../data/glacial_catalog.txt', 'r') as f:
            for line in f:
                if FileName in line:
                    SourceLat = float(line.split()[0])
                    SourceLon = float(line.split()[1])
                    amizuth = float(line.split()[3])
    else:
        print('key not found')

    itrace_to_distance = {}

    for itrace, trace in enumerate(DataStream):
        StationName = f'{DataStream[itrace].stats.network}.{DataStream[itrace].stats.station}'
        EventStationPair = Geodesic.WGS84.Inverse(SourceLat,SourceLon,StationLatLon[StationName][0],StationLatLon[StationName][1], outmask=1929)
        itrace_to_distance[itrace] = EventStationPair['s12']
    
    for itrace, trace in enumerate(DataStream):
        DataStream[itrace].stats.distance = 30000 * 1000

    for itrace, trace in enumerate(DataStream):
        DataStream[itrace].stats.distance = 1000 * 1000
        distance = EventStationPair['s12'] // 1000
        DataStream.plot(outfile=f'../data/{EventListName}_img2_1/{EventListName}_{FileName}_{distance}.jpg', type='section')
        img = Image.open(f'../data/{EventListName}_img2_1/{EventListName}_{FileName}_{distance}.jpg')
        img.crop((80, 60, 160, 529)).save(f'../data/{EventListName}_img2_1/{EventListName}_{FileName}_{distance}.jpg')
        DataStream[itrace].stats.distance = 30000 * 1000
    
    for itrace, trace in enumerate(DataStream):
        DataStream[itrace].stats.distance = EventStationPair['s12']

    DataStream.plot(outfile=f'../data/{EventListName}_img/{EventListName}_{FileName}.jpg', type='section')
    img = Image.open(f'../data/{EventListName}_img/{EventListName}_{FileName}.jpg')
    img.crop((80, 60, 960, 529)).save(f'../data/{EventListName}_img/{EventListName}_{FileName}.jpg')

    if EventListName == 'earthquakes':
        img.crop((80, 60, 960, 529)).save(f'../data/{EventListName}_img2_2/{EventListName}_{FileName}_{depth}.jpg')
    else:
        img.crop((80, 60, 960, 529)).save(f'../data/{EventListName}_img2_3/{EventListName}_{FileName}_{amizuth}.jpg')

    print(f'{index} {EventListName} {FileName} has been done')