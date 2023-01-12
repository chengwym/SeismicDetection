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
    elif EventListName == 'glacial':
        with open('../data/glacial_catalog.txt', 'r') as f:
            for line in f:
                if FileName in line:
                    SourceLat = float(line.split()[0])
                    SourceLon = float(line.split()[1])
    else:
        print('key not found')

    for itrace, trace in enumerate(DataStream):
        StationName = f'{DataStream[itrace].stats.network}.{DataStream[itrace].stats.station}'
        EventStationPair = Geodesic.WGS84.Inverse(SourceLat,SourceLon,StationLatLon[StationName][0],StationLatLon[StationName][1], outmask=1929)
        DataStream[itrace].stats.distance = EventStationPair['s12']

    DataStream.plot(outfile=f'../data/{EventListName}_img/{EventListName}_{FileName}.png', type='section')
    img = Image.open(f'../data/{EventListName}_img/{EventListName}_{FileName}.png')
    img.crop((80, 60, 960, 529)).save(f'../data/{EventListName}_img/{EventListName}_{FileName}.png')

    print(f'{index} {EventListName} {FileName} has been done')