import glob

csvfiles = sorted(glob.glob('2d-data/*.csv'))

for csvfile in csvfiles:

    name = csvfile.split('/')[-1].split('.')[0]
    with open(csvfile, 'r') as f:
        rows = f.readlines()
        rowzeros = rows[0].split(',')

        newrowzero = '/home/junkado/Desktop/keio/hard/focusright/{0}.mp4,\n'.format(name)
        for col in rowzeros[1:]:
            newrowzero += col

        with open(csvfile, 'w') as ff:
            ff.write(newrowzero)
            ff.writelines(rows[1:])
