from operator import le
import numpy as np
import random, os, copy, threading, glob, sys, cv2, platform
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import confusion_matrix
if platform.system() == 'Darwin':
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from .utils import tune, myScore, visualization, frame_detector, plot_confusion_matrix, labels2str, str2labels
from .objects.base.mil import MILBase, mkdir

class MILTrainMixin(MILBase):
    def train(self, estimator, paramDir, sampleNumPerLabel=0, customIndices=None):
        """
        :param paramDir: str, e.g. g0.001/C50
        """
        resultSuperDirPath = os.path.join(self.resultDir, paramDir)
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
            print(labels2str(labels))
            print(labels2str(np.sign(predictions)))
            print("Accuracy: {0}".format(np.average(labels == np.sign(predictions)) * 100))
            print(indices)

            print('correct labels are \n{0}'.format(labels2str(labels)), file=f)
            print('predicted labels are \n{0}'.format(labels2str(np.sign(predictions))), file=f)
            print("Accuracy: {0}".format(np.average(labels == np.sign(predictions)) * 100), file=f)
            print(self.info, file=f)
            print('train indices are \n{0}'.format(indices), file=f)

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

class MILEvalMixin(MILBase):
    def _resultSuperDirPath(self, paramDir):
        return os.path.join(self.resultDir, paramDir)

    def crossValidation(self, estimator, cv, paramDir, threadNum=8):
        """
        :param paramDir: str, e.g. g0.001/C50
        """
        resultSuperDirPath = self._resultSuperDirPath(paramDir)
        mkdir(resultSuperDirPath)

        score = cross_validate(estimator, self.bags, self.labels, scoring=myScore, cv=cv, n_jobs=threadNum)
        with open(os.path.join(resultSuperDirPath, 'crossVal-{0}.txt'.format(cv)), 'w') as f:
            print(score, file=f)
            print(self.info, file=f)

        print(score)

    def leaveOneOut(self, estimator, paramDir, n_jobs=8, trainAcc=False):
        """
        :param paramDir: str, e.g. g0.001/C50
        """
        resultSuperDirPath = self._resultSuperDirPath(paramDir)
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
            print('correct labels are\n{0}'.format(labels2str(corresctLabels)), file=f)
            print('predicted labels are\n{0}'.format(labels2str(predictedLabels)), file=f)
            print('Accuracy: {0}'.format(acc))
            print('Accuracy: {0}'.format(acc, file=f))
            if trainAcc:
                print('accuracies for train\n{0}'.format(trainAccuracies))
                print('accuracies for train\n{0}'.format(trainAccuracies), file=f)
            print(self.info, file=f)

    def read_loo(self, paramDir, reload=False):
        """
        :param paramDir: str, e.g. g0.001/C50
        """
        resultSuperDirPath = self._resultSuperDirPath(paramDir)
        estimatorPaths = sorted(glob.glob(os.path.join(resultSuperDirPath, 'leave-one-out', '*.pkl.cmp')))
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
    def leaveOnePersonOut(self, estimator, paramDir, n_jobs=8, trainAcc=False, resultVis=None):
        """
        :param paramDir: str, e.g. g0.001/C50
        """
        resultSuperDirPath = self._resultSuperDirPath(paramDir)
        mkdir(resultSuperDirPath)

        personList = list(set(self.persons))

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
            print('correct labels are\n{0}'.format(labels2str(corresctLabels)), file=f)
            print('predicted labels are\n{0}'.format(labels2str(predictedLabels)), file=f)
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
            # go to MILVisMixin method
            self.visualization(time_series_data, indices, resultSuperDirPath=resultSuperDirPath)

    def read_LOPO(self, paramDir, reload=False):
        """
        :param paramDir: str, e.g. g0.001/C50
        """
        resultSuperDirPath = self._resultSuperDirPath(paramDir)
        estimatorPaths = sorted(glob.glob(os.path.join(resultSuperDirPath, 'leave-one-person-out', '*.pkl.cmp')))
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

    def check_identification_func_max(self, paramDir, estimator, indices=None, manualData=None):
        """
        :param estimator: estimator or str. str represents estimator's filename(e.g. misvm.pkl.cmp).
        :param manualData: None or dict. dict's ket must be 'bags', 'labels', 'csvFilePaths'
        """
        path = self._resultSuperDirPath(paramDir)
        if isinstance(estimator, str):
            estimator = joblib.load(os.path.join(path, estimator))

        # check indices
        if indices is None:
            indices = [i for i in range(len(self.bags))]
        
        # check manual data
        if manualData:
            bags = manualData.get('bags')
            labels = manualData.get('labels')
            csvFilePaths = manualData.get('csvFilePaths')
        else:
            bags = self.bags
            labels = self.labels
            csvFilePaths = self.csvFilePaths
                    
        # select bags and labels by indices
        bags = [self.bags[index] for index in indices]
        labels = [self.labels[index] for index in indices]
        
        #print classifier.get_params()
        bag_predictions, instance_predictions = estimator.predict(bags, instancePrediction=True)

        print(labels2str(labels))
        print(labels2str(np.sign(bag_predictions)))
        print("Accuracy: {0}".format(np.average(labels == np.sign(bag_predictions)) * 100))

        sys.setrecursionlimit(5000)

        ini = 0
        bag_predictions = np.sign(bag_predictions)
        for index, bag_i in enumerate(indices):
            fin = ini + len(bags[bag_i])
            ident_func_result = np.array(instance_predictions[ini:fin])

            max = np.max(ident_func_result)
            if max <= 0:
                ini = fin
                continue
            with open(csvFilePaths[bag_i], 'r') as f:
                videopath = f.readline().split(',')[0]

                videoname = videopath.split('/')[-1][:-4]
                print("\nvideoname: {0}".format(videoname))
            #print("csvfile: {0}".format(csvnamelist[bag_i]))
            print("prediction labels: {0}".format(bag_predictions[index]))
            print("right labels: {0}".format(labels[index]))

            print("the maximum of identification func: {0}".format(max))
            frame = np.argmax(ident_func_result) * self.dicimate
            print("maximum frame: {0}".format(frame))
            frame_detector(csvFilePaths[bag_i], frame, path)
            #bag2video(csvnamelist[bag_i], nontimelist[bag_i], path)
            ini = fin

    def parse_txt(self, paramDir, filename):
        """
        :param paramDir: str, e.g. g0.001/C50
        :param filename: str
        Note that parsed txt must be;
        ======
        
            correct labels are 
            [-1, -1, 1, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1]
            predicted labels are 
            [-1. -1.  1. -1.  1. -1. -1.  1.  1. -1. -1.  1.  1.  1.  1.  1. -1. -1.
            1.  1. -1.  1. -1.  1.  1.  1.  1.  1.  1. -1.]
            Accuracy: 90.0
            ~~~~~

        ======   
        """
        path = os.path.join(self._resultSuperDirPath(paramDir), filename)
        with open(path, 'r') as f:
            lines = f.readlines()
            # true labels were written in 2 line
            labels = str2labels(lines[1])
            
            # predicted labels were written in 4 line
            predicted_labels = str2labels(lines[3])

            return labels, predicted_labels

class MILVisMixin(MILBase):
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


            visualization(imgsDict=imgsDict, method=vis_method, experience=self.experience, permode=permode,
                          positiveCsvFileName=self.info['positiveCsvFileName'], negativeCsvFileName=self.info['negativeCsvFileName'])
        else:
            raise NameError('{0} is undefined method in this function'.format(self.method))

    def confusion_matrix(self, predicted_labels, labels=None, title='Confusion matrix', target_names=['positive', 'negative'],
                        cmap=None, normalize=True):
        """
        :param target_names: given classification classes such as [0, 1, 2]
                the class names, for example: ['high', 'medium', 'low']

        :param title:        the text to display at the top of the matrix

        :param cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                see http://matplotlib.org/examples/color/colormaps_reference.html
                plt.get_cmap('jet') or plt.cm.Blues

        :param normalize:    If False, plot the raw numbers
                If True, plot the proportions
        """
        if labels is None:
            labels = self.labels

        if len(predicted_labels) != len(labels):
            raise ValueError('Unbalanced length between labels:{} and predicted_labels:{}'.format(len(labels), len(predicted_labels)))
        # create confusion matrix
        cm = confusion_matrix(labels, predicted_labels)
        
        # plot
        plot_confusion_matrix(self.runenv, cm, target_names, title, cmap, normalize)
