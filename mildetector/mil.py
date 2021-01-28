import glob, sys, os, csv
import numpy as np
import pandas as pd

from .milmixin import MILEvalMixin, MILVisMixin, MILTrainMixin, MILBase
from .objects import Pitch, Data


class MIL(MILEvalMixin, MILVisMixin, MILTrainMixin, MILBase):
    

    def setData(self, positiveCsvFileName, negativeCsvFileName, dicimate, videoExtension='mp4', csvExtension='csv', **bagkwargs):
        print("reading and converting into feature vectors...")
        # read video info txt
        with open('2d-data/video_dir_info.txt', 'r') as f:
            self.videosdir = f.readline()

        def _readCSV(path):
            df = pd.read_csv(path, header=None)
            filenames = df[0].tolist()
            infos = df[1].tolist()
            
            ret = {}
            for filename, info in zip(filenames, infos):
                videoname = filename + '.' + videoExtension
                infoDic = {'hand': info[0], 'person': info[1:]}
                ret[videoname] = infoDic
            return ret

        self.positives = _readCSV(os.path.join(self.rootdir, 'video', self.experience, positiveCsvFileName))
        self.negatives = _readCSV(os.path.join(self.rootdir, 'video', self.experience, negativeCsvFileName))

        self.positiveCsvFileName = positiveCsvFileName
        self.negativeCsvFileName = negativeCsvFileName

        csvAllFilePaths = sorted(glob.glob(os.path.join(self.rootdir, '2d-data', '*.' + csvExtension)))

        # mean
        if self.method in ['align', 'dirvec', 'img']:
            mean = self.calc_mean(csvAllFilePaths)
        else:
            mean = None
        self.pitches = []
        self.dicimate = dicimate


        sys.stdout.write('\r [{0}{1}]:{2}%'.format('#'*0, ' '*20, 0))
        sys.stdout.flush()

        for index, csvfilepath in enumerate(csvAllFilePaths):
            percent = (index + 1) / len(csvAllFilePaths)
            sys.stdout.write('\r [{0}{1}]:{2:d}%'.format('#'*int(percent*20), ' '*(20 - int(percent*20)), int(percent*100)))
            sys.stdout.flush()

            with open(csvfilepath, 'r') as f:
                reader = csv.reader(f)
                videopath = next(reader)[0]
                videoname = videopath.split('/')
                videoname = videoname[-1]

                header = next(reader)

                time_row = [row for row in reader]

                """
                x, y, c, times = data.elim_nan()

                if videoname in hard_csvfiles:
                    self.bags.append(np.array([[x[int(i / 2)][time] if i % 2 == 0 else y[int(i / 2)][time] for i in range(len(x) * 2)] for time in times]))
                    self.labels.append(1)

                elif videoname in easy_csvfiles:
                    self.bags.append(np.array([[x[int(i / 2)][time] if i % 2 == 0 else y[int(i / 2)][time] for i in range(len(x) * 2)] for time in times]))
                    self.labels.append(-1)
                """
                # data.nonantimes(update=True)
                label = self.get_label(videoname)
                if label == 1: # positive
                    hand = self.positives[videoname]['hand']
                    person = hand + self.positives[videoname]['person']
                elif label == -1: # negative
                    hand = self.negatives[videoname]['hand']
                    person = hand + self.negatives[videoname]['person']
                else:
                    hand = None
                    person = None
                    continue
                data = Data(videopath=videopath, width=header[1], height=header[3],
                                frame_num=header[5], fps=header[7], time_rows=time_row, hand=hand,
                                dicimate=dicimate, runenv=self.runenv, debug=self.debug)
                pitch = Pitch(label, person, csvfilepath, data)
                pitch.preprocess(self.method, mean=mean, **bagkwargs)
                self.pitches.append(pitch)

        sys.stdout.write('\n')
        sys.stdout.flush()
        
        print('positive: {0}, negative: {1}'.format(np.sum(self.labels == 1), np.sum(self.labels == -1)))

    def calc_mean(self, csvAllFilePaths, standardJointNumber=0, standardFrame=0):
        dataLists = []
        for index, csvfilepath in enumerate(csvAllFilePaths):
            with open(csvfilepath, 'r') as f:
                reader = csv.reader(f)
                videopath = next(reader)[0]
                videoname = videopath.split('/')
                videoname = videoname[-1]

                header = next(reader)

                time_row = [row for row in reader]
                dataLists.append(Data(videopath=videopath, width=header[1], height=header[3],
                                      frame_num=header[5], fps=header[7], time_rows=time_row))

        dataLists = np.array(dataLists)
        X = []
        Y = []
        for data_ in dataLists:
            X.append(data_.x[standardJointNumber][standardFrame])
            Y.append(data_.y[standardJointNumber][standardFrame])
        X = np.array(X)
        Y = np.array(Y)

        alignInfo = {'x': np.nanmean(X), 'y': np.nanmean(Y), 'joint': standardJointNumber, 'frame': standardFrame}
        return alignInfo  

    def exportFeatureVec2csv(self, data='feature'):
        positiveFile = os.path.basename(self.info['positiveCsvFileName']).split('.')[0]
        negativeFile = os.path.basename(self.info['negativeCsvFileName']).split('.')[0]
        exportPath = 'data/{0}_{1}_{2}_{3}_{4}.csv'.format(self.experience, self.dicimate, self.method, positiveFile, negativeFile)

        print('exporting to \"{0}\"'.format(exportPath))

        sys.stdout.write('\r [{0}{1}]:{2:d}%'.format('#' * 0, ' ' * 20, 0))
        sys.stdout.flush()
        with open(exportPath, 'w') as f:
            if data == 'all':
                # bag id, bag label, person id, csvpath, feature 1, ..., feature N
                for bagId, (bag, label, person, csvFilePath) in enumerate(zip(self.bags, self.labels, self.persons, self.csvFilePaths)):
                    percent = (bagId + 1) / len(self.labels)
                    sys.stdout.write('\r [{0}{1}]:{2:d}%'
                                     .format('#' * int(percent * 20), ' ' * (20 - int(percent * 20)),
                                             int(percent * 100)))
                    sys.stdout.flush()
                    for instance in bag:
                        features = ''
                        # print(instance.shape)
                        # 64x64=(4096)

                        for feature in instance:
                            features += '{0},'.format(feature)
                        row = '{0},{1},{2},{3},{4},\n'.format(bagId, label, person, csvFilePath, features)
                        f.write(row)
            elif data == 'feature':
                # bag id, bag label, feature 1, ..., feature N
                for index, (bag, label) in enumerate(zip(self.bags, self.labels)):
                    percent = (index + 1) / len(self.labels)
                    sys.stdout.write('\r [{0}{1}]:{2:d}%'
                                     .format('#' * int(percent * 20), ' ' * (20 - int(percent * 20)),
                                             int(percent * 100)))
                    sys.stdout.flush()
                    for instance in bag:
                        features = ''
                        # print(instance.shape)
                        # 64x64=(4096)

                        for feature in instance:
                            features += '{0},'.format(feature)
                        row = '{0},{1},{2},\n'.format(index, label, features)
                        f.write(row)
            elif data == 'person':
                # bag id, person id
                for bagId, person in enumerate(self.persons):
                    percent = (bagId + 1) / len(self.labels)
                    sys.stdout.write('\r [{0}{1}]:{2:d}%'
                                     .format('#' * int(percent * 20), ' ' * (20 - int(percent * 20)),
                                             int(percent * 100)))
                    sys.stdout.flush()
                    f.write('{0},{1},\n'.format(bagId, person))
            else:
                raise ValueError("{0} is invalid data name".format(data))
        print('\nfinished exporting csv file')

    def importCsv2Feature(self, positiveCsvFileName, negativeCsvFileName, dicimate, data='feature', featureDims=4096, positiveLabel=1.0, negativeLabel=-1.0):
        positiveFile = os.path.basename(positiveCsvFileName).split('.')[0]
        negativeFile = os.path.basename(negativeCsvFileName).split('.')[0]
        importPath = 'data/{0}_{1}_{2}_{3}_{4}.csv'.format(self.experience, dicimate, self.method, positiveFile, negativeFile)

        print('importing from \"{0}\"'.format(importPath))

        sys.stdout.write('\r [{0}{1}]:{2:d}%'.format('#' * 0, ' ' * 20, 0))
        sys.stdout.flush()
        with open(importPath, 'r') as f:
            self.dicimate = dicimate
            self.info = {'positiveCsvFileName': positiveCsvFileName, 'negativeCsvFileName': negativeCsvFileName}

            bagId = 0
            temporalBag = []
            lines = f.readlines()

            if data == 'all':
                self.bags = []
                self.labels = []
                self.persons = []
                self.csvFilePaths = []
                self.positives = {}
                self.negatives = {}

                # bag id, bag label, person id, csvpath, feature 1, ..., feature N
                for index, line in enumerate(lines):
                    percent = (index + 1) / len(lines)
                    sys.stdout.write('\r [{0}{1}]:{2:d}%'
                                     .format('#' * int(percent * 20), ' ' * (20 - int(percent * 20)),
                                             int(percent * 100)))
                    sys.stdout.flush()

                    data = line.split(',')
                    bagid = float(data[0])
                    baglabel = float(data[1])
                    person = data[2]
                    csvfilepath = data[3]
                    features = np.array(data[4: 4 + featureDims], dtype=np.float)
                    #####mistaken code!!!!!######
                    if bagId != bagid:
                        self.bags.append(temporalBag)
                        self.labels.append(baglabel)
                        self.persons.append(person)
                        self.csvFilePaths.append(csvfilepath)
                        infoDic = {'hand': person[0], 'person': person[1:]}
                        if baglabel == positiveLabel:
                            self.positives[csvfilepath] = infoDic
                        elif baglabel == negativeLabel:
                            self.negatives[csvfilepath] = infoDic
                        else:
                            print('\nWarning: unknown label was detected {0}\n'.format(baglabel))
                        temporalBag = []
                        bagId += 1

                    else:
                        temporalBag.append(features)
                self.labels = np.array(self.labels)
                print('\npositive: {0}, negative: {1}'.format(np.sum(self.labels == 1), np.sum(self.labels == -1)))

            elif data == 'feature':
                self.bags = []
                self.labels = []
                # bag id, bag label, feature 1, ..., feature N
                for index, line in enumerate(lines):
                    percent = (index + 1) / len(lines)
                    sys.stdout.write('\r [{0}{1}]:{2:d}%'
                                     .format('#' * int(percent * 20), ' ' * (20 - int(percent * 20)),
                                             int(percent * 100)))
                    sys.stdout.flush()

                    data = line.split(',')
                    bagid = float(data[0])
                    baglabel = float(data[1])
                    features = np.array(data[2: 2 + featureDims], dtype=np.float)

                    if bagId == bagid:
                        temporalBag.append(features)
                    else:
                        self.bags.append(temporalBag)
                        self.labels.append(baglabel)
                        temporalBag = []
                        bagId += 1
                self.labels = np.array(self.labels)
                print('\npositive: {0}, negative: {1}'.format(np.sum(self.labels == 1), np.sum(self.labels == -1)))

            elif data == 'person':
                self.persons = []
                # bag id, person id
                for index, line in enumerate(lines):
                    percent = (index + 1) / len(lines)
                    sys.stdout.write('\r [{0}{1}]:{2:d}%'
                                     .format('#' * int(percent * 20), ' ' * (20 - int(percent * 20)),
                                             int(percent * 100)))
                    sys.stdout.flush()

                    data = line.split(',')
                    bagid = float(data[0])
                    person = data[1]
                    self.persons.append(person)

            else:
                raise ValueError("{0} is invalid data name".format(data))
        print('\nfinished exporting csv file')

