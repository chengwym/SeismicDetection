def get_earthquakes_dir(path):
    earthquakes_dir = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            words = line.split('|')
            depth = float(words[3].split(' ')[1])
            date = words[0][: 10].replace('-', '')
            earthquakes_dir[date] = depth
    return earthquakes_dir

def get_glacial_dir(path):
    glacial_dir = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            words = line.split()
            azimuth = float(words[3])
            date = words[-1]
            glacial_dir[date] = azimuth
    return glacial_dir