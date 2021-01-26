import glob
from .data import Data
import csv
import numpy as np
import sys
import random
import os
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV, cross_validate
from parameter_tune import tune
import copy
import threading
import cv2
import platform
import pandas as pd
import cv2
if platform.system() == 'Darwin':
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from vis_data import visualization
from .openpose import OpenPoseBase
from .pitch import Pitch

def mkdir(resultSuperDirPath):
    if not os.path.isdir('{0}'.format(resultSuperDirPath)):
        os.makedirs('{0}'.format(resultSuperDirPath))

    if not os.path.isdir('{0}/func_max_image'.format(resultSuperDirPath)):
        os.makedirs('{0}/func_max_image'.format(resultSuperDirPath))

    if not os.path.isdir('{0}/nonzero_image'.format(resultSuperDirPath)):
        os.makedirs('{0}/nonzero_image'.format(resultSuperDirPath))

    if not os.path.isdir('{0}/bag_video'.format(resultSuperDirPath)):
        os.makedirs('{0}/bag_video'.format(resultSuperDirPath))

    if not os.path.isdir('{0}/leave-one-out'.format(resultSuperDirPath)):
        os.makedirs('{0}/leave-one-out'.format(resultSuperDirPath))

    if not os.path.isdir('{0}/leave-one-person-out'.format(resultSuperDirPath)):
        os.makedirs('{0}/leave-one-person-out'.format(resultSuperDirPath))

    if not os.path.isdir('{0}/video'.format(resultSuperDirPath)):
        os.makedirs('{0}/video'.format(resultSuperDirPath))

def myScore(estimator, x, y):
    yPred = np.sign(estimator.predict(x, instancePrediction=False))
    acc = accuracy_score(y, yPred)
    return acc


class MILBase(OpenPoseBase):
    def __init__(self, method, experience, dirName, estimatorName, runenv, bonetype='BODY_25', debug=False):
        """
        :param method: method name. ['misvm', 'MISVM', 'MILES']
        :param experience: experience name.
        :param dirName: result directory name.
        :param estimatorName: estimator name to save.
        :param debug: Bool, whether to print log for debug
        """
        super().__init__(bonetype, runenv, debug)
        self.method = method
        self.experience = experience
        self.dirName = dirName

        self.estimatorName = estimatorName

        self.info = {}
        self.positives = {}
        self.negatives = {}

        self.pitches = []

    @property
    def negativeCsvFileName(self):
        return self.info['negativeCsvFileName']
    @negativeCsvFileName.setter
    def negativeCsvFileName(self, filename):
        self.info['negativeCsvFileName'] = filename
    @property
    def negativeCsvFileNames(self):
        return list(self.negatives.keys())

    @property
    def positiveCsvFileName(self):
        return self.info['positiveCsvFileName']
    @positiveCsvFileName.setter
    def positiveCsvFileName(self, filename):
        self.info['positiveCsvFileName'] = filename
    @property
    def positiveCsvFileNames(self):
        return list(self.positives.keys())

    @property
    def labels(self):
        return np.array([pitch.label for pitch in self.pitches])
    @property
    def bags(self):
        """
        return list of ndarray
        """
        return [pitch.bag for pitch in self.pitches]
    @property
    def persons(self):
        return [pitch.person for pitch in self.pitches]
    @property
    def csvFilePaths(self):
        return [pitch.csvpath for pitch in self.pitches]

    def get_label(self, videoname):
        if videoname in self.positiveCsvFileNames:
            return 1

        elif videoname in self.negativeCsvFileNames:
            return -1

        else:
            return 0

class MIL(MILBase):
    

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

    def dataVisualization(self, vis_method='t-sne', permode='person'):
        if self.method in ['img']:
            # mode
            imgsDict = {}

            imgsize = int(np.sqrt(self.bags[0].shape[1]))
            if permode == 'label':
                #labels = ['pos', 'neg']
                posargs = np.where(self.labels == 1)[0]
                imgsDict['pos'] = []
                for posarg in posargs:
                    imgsDict['pos'].append(
                        self.bags[posarg].reshape(self.bags[posarg].shape[0], imgsize, imgsize))

                negargs = np.where(self.labels == -1)[0]
                imgsDict['neg'] = []
                for negarg in negargs:
                    imgsDict['neg'].append(
                        self.bags[negarg].reshape(self.bags[negarg].shape[0], imgsize, imgsize))


            elif permode == 'person':
                personList = list(set(self.persons))
                imgsDict = {label: [] for label in personList}
                for i, person in enumerate(self.persons):
                    imgsDict[person].append(self.bags[i])

            else:
                raise NameError('{0} is invalid permode'.format(permode))


            visualization.visualization(imgsDict=imgsDict, method=vis_method, experience=self.experience, permode=permode,
                          positiveCsvFileName=self.info['positiveCsvFileName'], negativeCsvFileName=self.info['negativeCsvFileName'])
        else:
            raise NameError('{0} is undefined method in this function'.format(self.method))

    def train(self, estimator, resultSuperDirPath, sampleNumPerLabel=0, customIndices=None):
        mkdir(resultSuperDirPath)

        print("accuracy will be for train data")
        if customIndices is None:
            if sampleNumPerLabel == 0:  # all
                indices = [i for i in range(len(self.bags))]
            else:
                print("random sampling...")
                posiindex = np.where(np.array(self.labels) == 1)[0]
                negaindex = np.where(np.array(self.labels) == -1)[0]
                posiindex = random.sample(posiindex, sampleNumPerLabel)
                negaindex = random.sample(negaindex, sampleNumPerLabel)

                indices = random.sample(np.concatenate((posiindex, negaindex)), 2 * sampleNumPerLabel)
                print("Note that train data indices are \n{0}".format(indices))
        else:
            indices = customIndices

        bags = [self.bags[index] for index in indices]
        labels = [self.labels[index] for index in indices]

        estimator.fit(bags, labels)
        joblib.dump(estimator, os.path.join(resultSuperDirPath, '{0}.pkl.cmp'.format(self.estimatorName)), compress=True)
        predictions, instance_labels = estimator.predict(bags, instancePrediction=True)

        with open('{0}/trainResult.txt'.format(resultSuperDirPath), 'w') as f:
            print(labels)
            print(np.sign(predictions))
            print("Accuracy: {0}".format(np.average(labels == np.sign(predictions)) * 100))
            print(indices)

            print('correct labels are \n{0}'.format(labels), file=f)
            print('predicted labels are \n{0}'.format(np.sign(predictions)), file=f)
            print("Accuracy: {0}".format(np.average(labels == np.sign(predictions)) * 100), file=f)
            print(self.info, file=f)
            print('train indices are \n{0}'.format(indices), file=f)

    def searchGamma(self, ini, fin, step, randomSampledTrainRatio=0.8):
        if len(self.bags) == 0:
            raise NotImplementedError("this method must call setData")

        randomSampledTrainNum = int(randomSampledTrainRatio * len(self.bags))

        indices = random.sample([j for j in range(len(self.bags))], randomSampledTrainNum)
        train_bags = [self.bags[index] for index in indices]
        test_bags = [self.bags[index] for index in range(len(self.bags)) if index not in indices]

        text = 'bag number: {0}\n'.format(len(self.bags))
        text_, K, best_g = tune(ini, fin, step, train_bags, test_bags)
        text += text_

        newpath = './result/{0}/{1}/g{2}/'.format(self.experience, self.dirName, best_g)
        if not os.path.isdir(newpath):
            os.makedirs('{0}'.format(newpath))

        with open('{0}/gamma.txt'.format(newpath), 'w') as f:
            f.write(text)
            print(K, file=f)
            print('train bag number: {0}, test bag number: {1}'.format(randomSampledTrainNum, len(self.bags) - randomSampledTrainNum), file=f)
            print(self.info, file=f)

    def gridSearch(self, estimator, params_grid, cv, threadNum=8):
        gscv = GridSearchCV(estimator, params_grid, cv=cv, scoring=myScore, n_jobs=threadNum)
        gscv.fit(self.bags, self.labels)
        print(gscv.cv_results_)
        print('\n\nbest score is')
        print(gscv.best_params_)

        best_g = gscv.best_params_['gamma']
        if not os.path.isdir('result/{0}/{1}/g{2}'.format(self.experience, self.dirName, best_g)):
            os.makedirs('result/{0}/{1}/g{2}'.format(self.experience, self.dirName, best_g))

        with open('result/{0}/{1}/g{2}/gridsearch.txt'.format(self.experience, self.dirName, best_g), 'w') as f:
            print('{0}\n'.format(gscv.cv_results_), file=f)
            print('best parameters', file=f)
            print('{0}'.format(gscv.best_params_), file=f)
            print(self.info, file=f)

    def crossValidation(self, estimator, cv, resultSuperDirPath, threadNum=8):
        mkdir(resultSuperDirPath)

        score = cross_validate(estimator, self.bags, self.labels, scoring=myScore, cv=cv, n_jobs=threadNum)
        with open(os.path.join(resultSuperDirPath, 'crossVal-{0}.txt'.format(cv)), 'w') as f:
            print(score, file=f)
            print(self.info, file=f)

        print(score)

    def leaveOneOut(self, estimator, resultSuperDirPath, n_jobs=8, trainAcc=False):
        mkdir(resultSuperDirPath)

        predictedLabelsDict = {}
        corresctLabelsDict = {}
        trainAccuracies = []
        for index in range(0, len(self.bags), n_jobs):
            threadList = []

            def job(threadIndex):
                testIndex = threadIndex
                trainIndices = np.concatenate([np.arange(0, testIndex), np.arange(testIndex + 1, len(self.bags))])

                trainBags = [bag for index, bag in enumerate(self.bags) if index in trainIndices]
                trainLabels = np.array(self.labels)[trainIndices]
                testBag = [self.bags[testIndex]]
                testLabel = [self.labels[testIndex]]

                estimator_ = copy.deepcopy(estimator)
                estimator_.fit(trainBags, trainLabels)

                predicts = estimator_.predict(testBag, instancePrediction=False)

                predictedLabelsDict[str(threadIndex)] = np.sign(predicts)
                corresctLabelsDict[str(threadIndex)] = testLabel

                joblib.dump(estimator_, os.path.join(resultSuperDirPath, 'leave-one-out', '{0}-{1}.pkl.cmp'.format(self.estimatorName, threadIndex)),
                            compress=True)
                if trainAcc:
                    predictedTrains = estimator_.predict(trainBags, instancePrediction=False)
                    trainAccuracies.append(np.average(np.sign(predictedTrains) == trainLabels) * 100)

            lastThreadIndex = index + n_jobs
            if lastThreadIndex > len(self.bags):
                lastThreadIndex = len(self.bags)

            for threadIndex in range(index, lastThreadIndex):
                thread = threading.Thread(target=job, name=str(threadIndex), args=([threadIndex]))
                thread.daemon = True
                thread.start()
                threadList.append(thread)

            for thread_ in threadList:
                thread_.join()

            del threadList

        # sort
        predictedLabels = []
        corresctLabels = []
        for i in range(len(self.bags)):
            predictedLabels.append(predictedLabelsDict[str(i)])
            corresctLabels.append(corresctLabelsDict[str(i)])
        predictedLabels = np.squeeze(np.array(predictedLabels))
        corresctLabels = np.squeeze(np.array(corresctLabels))

        acc = (np.average(predictedLabels == corresctLabels) * 100)
        with open(os.path.join(resultSuperDirPath, 'leave-one-out.txt'), 'w') as f:
            print('correct labels are\n{0}'.format(corresctLabels), file=f)
            print('predicted labels are\n{0}'.format(predictedLabels), file=f)
            print('Accuracy: {0}'.format(acc))
            print('Accuracy: {0}'.format(acc, file=f))
            if trainAcc:
                print('accuracies for train\n{0}'.format(trainAccuracies))
                print('accuracies for train\n{0}'.format(trainAccuracies), file=f)
            print(self.info, file=f)

    def read_loo(self, resultSuperDirPath, reload=False):
        estimatorPaths = sorted(glob.glob(resultSuperDirPath + '/leave-one-out/*.pkl.cmp'))
        estimators = []

        p = 0
        for estimatorPath in estimatorPaths:
            estimators.append(joblib.load(estimatorPath))
            if reload:
                estimator = estimators[-1]
                filename = os.path.basename(estimatorPath)
                index = int(filename.split('.')[0].split('-')[1])

                predictedLabel = estimator.predict([self.bags[index]], instancePrediction=False)
                if np.sign(predictedLabel) == self.labels[index]:
                    p += 1

        if reload:
            print('accuracy: {0}'.format(float(p) * 100/len(self.labels)))

        return estimators

    # resultVis = {}
    def leaveOnePersonOut(self, estimator, resultSuperDirPath, n_jobs=8, trainAcc=False, resultVis=None):
        personList = list(set(self.persons))

        mkdir(resultSuperDirPath)

        predictedLabelsDict = {}
        corresctLabelsDict = {}
        trainAccuracies = {}
        resultVisualization = {}
        for index in range(0, len(personList), n_jobs):
            threadList = []

            def job(threadIndex):
                testIndices = np.array([i for i, person in enumerate(self.persons) if person == personList[threadIndex]])
                if testIndices.size == 0:
                    predictedLabelsDict[str(threadIndex)] = []
                    corresctLabelsDict[str(threadIndex)] = []
                    if trainAcc:
                        trainAccuracies[str(threadIndex)] = "Not calculated"
                    return
                trainIndices = np.setdiff1d(np.arange(0, len(self.bags)), testIndices)

                trainBags = [bag for index, bag in enumerate(self.bags) if index in trainIndices]
                trainLabels = np.array(self.labels)[trainIndices]
                testBags = [bag for index, bag in enumerate(self.bags) if index in testIndices]
                testLabels =  np.array(self.labels)[testIndices]

                estimator_ = copy.deepcopy(estimator)
                estimator_.fit(trainBags, trainLabels)

                predicts, inst_preds = estimator_.predict(testBags, instancePrediction=True)

                predictedLabelsDict[str(threadIndex)] = np.sign(predicts)
                corresctLabelsDict[str(threadIndex)] = testLabels

                joblib.dump(estimator_, os.path.join(resultSuperDirPath, 'leave-one-person-out',
                                                    '{0}-{1}.pkl.cmp'.format(self.estimatorName, personList[threadIndex])),
                            compress=True)

                if trainAcc:
                    predictedTrains = estimator_.predict(trainBags, instancePrediction=False)
                    trainAccuracies[str(threadIndex)] = np.average(np.sign(predictedTrains) == trainLabels) * 100

                if resultVis is not None and isinstance(resultVis, str):
                    ini = 0
                    time_series_data = []
                    for bag_i in testIndices:
                        fin = ini + len(self.bags[bag_i])
                        time_series_data.append(np.array(eval('{0}[ini:fin]'.format(resultVis))))
                        ini = fin
                    resultVisualization[str(threadIndex)] = {'tsd': time_series_data, 'indices': testIndices}
                elif resultVis is not None:
                    raise ValueError('resultVis must be str, \'predict\',\'estimator_.w_\', etc.')

            lastThreadIndex = index + n_jobs
            if lastThreadIndex > len(personList):
                lastThreadIndex = len(personList)

            for threadIndex in range(index, lastThreadIndex):
                thread = threading.Thread(target=job, name=str(threadIndex), args=([threadIndex]))
                thread.daemon = True
                thread.start()
                threadList.append(thread)

            for thread_ in threadList:
                thread_.join()

            del threadList

        # sort
        predictedLabels = []
        corresctLabels = []
        for i in range(len(personList)):
            predictedLabels.extend(predictedLabelsDict[str(i)])
            corresctLabels.extend(corresctLabelsDict[str(i)])

        predictedLabels = np.array(predictedLabels)
        corresctLabels = np.array(corresctLabels)

        acc = (np.average(predictedLabels == corresctLabels) * 100)
        with open(os.path.join(resultSuperDirPath, 'leave-one-person-out.txt'), 'w') as f:
            print('correct labels are\n{0}'.format(corresctLabels), file=f)
            print('predicted labels are\n{0}'.format(predictedLabels), file=f)
            print('Accuracy: {0}'.format(acc), file=f)
            print('Accuracy: {0}'.format(acc))
            if trainAcc:
                print('accuracies for train\n{0}'.format(
                    {personList[int(i)]: trainAccuracy for i, trainAccuracy in trainAccuracies.items()}), file=f)
                print('accuracies for train\n{0}'.format({personList[int(i)]: trainAccuracy for i, trainAccuracy in trainAccuracies.items()}))

            print(self.info, file=f)

        if resultVis is not None:
            time_series_data = []
            indices = []
            for visData in resultVisualization.values():
                time_series_data.extend(visData['tsd'])
                indices.extend(visData['indices'])
            self.visualization(time_series_data, indices, resultSuperDirPath=resultSuperDirPath)

    def read_LOPO(self, resultSuperDirPath, reload=False):
        estimatorPaths = sorted(glob.glob(resultSuperDirPath + '/leave-one-person-out/*.pkl.cmp'))
        estimators = []

        p = 0
        for estimatorPath in estimatorPaths:
            estimators.append(joblib.load(estimatorPath))
            if reload:
                estimator = estimators[-1]
                filename = os.path.basename(estimatorPath)
                filedPerson = int(filename.split('.')[0].split('-')[1])

                testIndices = np.array([i for i, person in enumerate(self.persons) if person == filedPerson])

                predictedLabel = estimator.predict([bag for index, bag in enumerate(self.bags) if index in testIndices], instancePrediction=False)
                p += np.sum(np.sign(predictedLabel) == np.array(self.labels)[testIndices])

        if reload:
            print('accuracy: {0}'.format(float(p) * 100/len(self.labels)))

        return estimators

    """
    example: pluralParametersImplement(estimators, pathes,
                                        extraArgs=[{}, {sampleNumPerLabel:3}, {}], n_jobs=8)
    """
    def pluralParametersTrain(self, estimators, pathes, extraArgs=None, n_jobs=8): # implementations
        if extraArgs is None:
            extraArgs = [{} for i in range(len(pathes))]
        for index in range(0, len(estimators), n_jobs):
            threadList = []

            def job(threadIndex):
                estimator_ = estimators[threadIndex]
                path = pathes[threadIndex]
                args = extraArgs[threadIndex]
                self.train(estimator=estimator_, resultSuperDirPath=path, **args)


            lastThreadIndex = index + n_jobs
            if lastThreadIndex > len(estimators):
                lastThreadIndex = len(estimators)

            for threadIndex in range(index, lastThreadIndex):
                thread = threading.Thread(target=job, name=str(threadIndex), args=([threadIndex]))
                thread.daemon = True
                thread.start()
                threadList.append(thread)

            for thread_ in threadList:
                thread_.join()

            del threadList

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

    def visualization(self, time_series_data, indices, resultSuperDirPath, mode='plot', frameSize=(800, 600)):
        mkdir(resultSuperDirPath)
        ini = 0

        for index, bag_i in enumerate(indices):
            sys.stdout.write('\r{0}/{1}'.format(index + 1, len(indices)))
            fin = ini + len(self.bags[bag_i])

            with open(self.csvFilePaths[bag_i], 'r') as f:
                videopath = f.readline().split(',')[0]

                videoname = videopath.split('/')[-1][:-4] + '.mp4'
                video = cv2.VideoCapture(videopath)
                height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_max = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_rate = video.get(cv2.CAP_PROP_FPS)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(os.path.join(resultSuperDirPath, 'video', videoname), fourcc, frame_rate, frameSize)

                fig = plt.figure()
                # (x, y)
                graphImgSize = (int(frameSize[0]), int(frameSize[1]/2))
                def updateFrame(frame):
                    if frame != 0:
                        plt.cla()

                    xmax = time_series_data[index].size - 1
                    ymin, ymax = np.min(time_series_data[index]) - 0.1, np.max(time_series_data[index]) + 0.1
                    plt.xlim([0, xmax])
                    plt.ylim([ymin, ymax])
                    if mode == 'plot':
                        plt.hlines(y=[0], xmin=0, xmax=xmax, colors='black', linestyles='--')
                        plt.plot(time_series_data[index], '-o')

                        x = int(frame / self.dicimate)
                        plt.vlines(x=[x], ymin=ymin, ymax=ymax, colors='black', linewidths=3)
                        plt.plot([x], time_series_data[index][x], 'o', color='red')
                    elif mode == 'bar':
                        plt.bar(np.arange(time_series_data[index].size), time_series_data[index], align='center', color='black')

                        x = int(frame / self.dicimate)
                        plt.bar(x, time_series_data[index][x], align='center', color='red')
                    else:
                        raise ValueError('{0} is invalid mode'.format(mode))

                for frame in range(frame_max):
                    percent = int((frame + 1.0) * 100 / frame_max)
                    sys.stdout.write(
                        '\r{0}/{1}:writing {2}... |{3}| {4}% finished'.format(index + 1, len(indices), videoname,
                                                                              '#' * int(percent * 0.2) + '-' * (20 - int(percent * 0.2)),
                                                                              percent))
                    sys.stdout.flush()

                    ret, videoImg = video.read()
                    updateFrame(frame)

                    # convert canvas to image
                    fig.canvas.draw()

                    width, height = fig.get_size_inches() * fig.get_dpi()
                    graphImg = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8', sep='').reshape(int(height),
                                                                                                       int(width), 3)
                    # img is rgb, convert to opencv's default bgr
                    graphImg = cv2.resize(cv2.cvtColor(graphImg, cv2.COLOR_RGB2BGR), graphImgSize)

                    outputImg = cv2.vconcat((graphImg, cv2.resize(videoImg, (int(frameSize[0]), int(frameSize[1]/2)))))
                    writer.write(outputImg)

                writer.release()
                video.release()


            ini = fin

        print('\nfinished')
