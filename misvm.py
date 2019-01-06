from __future__ import print_function
from openpose import OpenPose
import numpy as np
import misvm
from sklearn.externals import joblib
from frame_detector import frame_detector, bag2video, plusvideo
# activate milpy35
from mil import MIL

method = 'combination'
dir_name = 'combination-misvm'
kernel = 'rbf'
gamma = 0.0012
C = 4500
experience = '2018'
positiveCsvFileName='hard-video.csv'
negativeCsvFileName='easy-video.csv'
path = './result/{0}/{1}/g{2}/c{3}'.format(experience, dir_name, gamma, C)
dicimate = 4

mil = MIL(method=method, experience=experience, dirName=dir_name, estimatorName='misvm')
mil.setData(positiveCsvFileName=positiveCsvFileName, negativeCsvFileName=negativeCsvFileName,
            saveMotionTmplate=False, dicimate=dicimate, videoExtension='mp4', csvExtension='csv')
#mil.importCsv2Feature(positiveCsvFileName, negativeCsvFileName, dicimate, data='all')


def main():# read hard and easy
    #estimator = misvm.miSVM(kernel=kernel, gamma=gamma, C=C, verbose=True, max_iters=100)
    estimator = misvm.miPSVM(feature=kernel, gamma=gamma, C=C, verbose=True, max_iters=100, n_components=36*3, sparse=['P'])
    mil.train(estimator=estimator, resultSuperDirPath=path)

def check_identification_func_max():
    bags, labels, csvnamelist = mil.bags, mil.labels, mil.csvFilePaths

    #classifier = joblib.load('result/flag/flag-parameter-c10000.cmp')
    classifier = joblib.load(path + '/misvm.pkl.cmp')
    indexes = [i for i in range(len(bags))]
    #indexes = [4, 25, 35, 30, 14, 13, 12, 37, 48, 16, 39, 24, 34, 6, 49, 5, 18, 38, 11, 28, 40, 23, 21, 41, 8, 10, 26, 43, 47, 19]
    #indexes = [index for index in range(len(bags)) if index not in indexes]
    bags = [bags[index] for index in indexes]
    labels = [labels[index] for index in indexes]

    #print classifier.get_params()
    bag_predictions, instance_predictions = classifier.predict(bags, instancePrediction=True)
    print(labels)
    print(np.sign(bag_predictions))
    print("Accuracy: {0}".format(np.average(labels == np.sign(bag_predictions)) * 100))

    import sys
    sys.setrecursionlimit(5000)

    ini = 0
    bag_predictions = np.sign(bag_predictions)
    for index, bag_i in enumerate(indexes):
        fin = ini + len(bags[bag_i])
        ident_func_result = np.array(instance_predictions[ini:fin])

        max = np.max(ident_func_result)
        if max <= 0:
            ini = fin
            continue
        with open(csvnamelist[bag_i], 'r') as f:
            videopath = f.readline().split(',')[0]

            videoname = videopath.split('/')[-1][:-4]
            print("videoname: {0}".format(videoname))
        #print("csvfile: {0}".format(csvnamelist[bag_i]))
        print("prediction labels: {0}".format(bag_predictions[index]))
        print("right labels: {0}".format(labels[index]))

        print("the maximum of identification func: {0}".format(max))
        frame = np.argmax(ident_func_result) * dicimate
        print("maximum frame: {0}".format(frame))
        frame_detector(csvnamelist[bag_i], frame, path)
        #bag2video(csvnamelist[bag_i], nontimelist[bag_i], path)
        ini = fin



def search_hyperparameter(ini, fin, step, randomSampledTrainRatio):
    mil.searchGamma(ini=ini, fin=fin, step=step, randomSampledTrainRatio=randomSampledTrainRatio)

def gridsearch(params_grid, cv=2):
    estimator = misvm.miSVM(max_iters=250)
    mil.gridSearch(estimator=estimator, params_grid=params_grid, cv=cv)


def cross_validation():
    estimator = misvm.miSVM(kernel=kernel, gamma=gamma, C=C, verbose=True, max_iters=200)
    mil.crossValidation(estimator, 5, path)

def leave_one_out(n_jobs=8):
    estimator = misvm.miSVM(kernel=kernel, gamma=gamma, C=C, verbose=True, max_iters=200)
    mil.leaveOneOut(estimator=estimator, resultSuperDirPath=path, n_jobs=n_jobs, trainAcc=True)

def leave_one_person_out(n_jobs=8, resultVis=False):
    if resultVis:
        resultvis = 'inst_preds'
    else:
        resultvis = None
    estimator = misvm.miSVM(kernel=kernel, gamma=gamma, C=C, verbose=True, max_iters=200)
    mil.leaveOnePersonOut(estimator=estimator, resultSuperDirPath=path, n_jobs=n_jobs, trainAcc=True, resultVis=resultvis)

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()



def get_from_openpose():
    op = OpenPose()
    #op.get_from_openpose(videosdir='/home/junkado/Desktop/keio/hard/focusright', extension=".mp4")
    op.manual_videofile("/home/junkado/Desktop/keio/hard/allcutvideo/C1234.mp4")

    #editLists = [8,9,10,11,12,13,14,16,17,23,24,25,26,27,30,50,51,65,99,100,101,102,103,125,129,130,131,132,133,134,135,137,138,139,140,141,143,145,160,163]
    #editLists = [170,261,263,264,266,269,270,271,272,273,274,275,276,277,306,307,308,309,310,311,312,314,315,316,317,318,319,320,323,325,326,327,328,329,338,339,340,341,342,343]
    #videopaths = ["/home/junkado/Desktop/keio/hard/focusright/{0}.mp4".format(editfile) for editfile in editLists]
    #op.manual_videofiles(videopaths)

def exportFeatureVec2Csv():
    #mil.exportFeatureVec2csv(data='feature')
    mil.exportFeatureVec2csv(data='all')

def visualization():
    bags = mil.bags
    classifier = joblib.load(path + '/misvm.pkl.cmp')
    indices = [i for i in range(len(bags))]
    bag_predictions, instance_predictions = classifier.predict(bags, instancePrediction=True)

    ini = 0
    ident_func_results = []
    for index, bag_i in enumerate(indices):
        fin = ini + len(bags[bag_i])
        ident_func_results.append(np.array(instance_predictions[ini:fin]))
        ini = fin
    """
    ident_func_results.append(np.array([-0.78560122, 0.54519545, 0.46807868, 0.46715913, 0.26363045, 0.27786555,
                                        0.22690806, 0.20750884, 0.32449584, 0.37900016, 0.23489776, 0.15342022,
                                        0.4148295, 0.25238769, 0.29778564, 0.22747711, 0.21663552, 0.07551101,
                                        0.19331927, 0.18972733, 0.2916057, 0.31533071, 0.28310115, 0.19699904
                                           , -0.44702401, -0.23014429, 0.09828476, -0.13944687, -0.07006605, -0.10182374
                                           , -0.18137682, -0.245083, -0.57225477, -0.55978502, -0.84435834, -0.8639321
                                           , -0.89150618, -0.97406168, -0.89231222, -0.44298549, -0.39935993, -0.311711
                                           , -0.42653582, -0.51900241, -0.50230192, -0.36062895, -0.3329459, -0.26808575
                                           , -0.23018796, -0.21894479, -0.4197332, -0.40067229, -0.55829579, -0.55243558
                                           , -0.44397456, -0.59697793, -0.57133463, -0.63345068, -0.42979022, -0.2246336
                                           , -0.36737076, -0.23052741, -0.06819818, -0.42510965, -1.00368161,
                                        -1.03595145
                                           , -1.00060042, -0.96606247, -1.02635246, -1.041099, -0.95649054, -0.95343488
                                           , -0.77730283, -0.6717092, -0.45264443, -0.44387568, -0.26860944, -0.06294351
                                           , -0.20784132, -0.28675038, 0.00213939, -0.01451041, 0.03790828, 0.44598379
                                           , 0.32833122, -0.22523026, -0.78366198, -0.77177513, -1.00599781,
                                        -1.50473151]))
    """
    mil.visualization(ident_func_results, indices, resultSuperDirPath=path)

if __name__ == '__main__':
    #search_hyperparameter(ini=0.001, fin=0.002, step=0.0001, randomSampledTrainRatio=0.8)
    #gridsearch(params_grid=[{'gamma': [0.0012], 'C': [10, 50, 100, 500, 1000, 5000, 10000], 'kernel': ['rbf']}])
    main()
    #check_identification_func_max()
    #cross_validation()
    #get_from_openpose()
    #exportFeatureVec2Csv()
    #visualization()
    #leave_one_out(n_jobs=14)
    #leave_one_person_out(n_jobs=10, resultVis=True)
    """
    # use thread
    estimators = []
    pathes = []
    CC = [1000, 3000, 3500, 4000, 5000]
    for cc in CC:
        estimators.append(misvm.miSVM(kernel=kernel, gamma=gamma, C=cc, verbose=True, max_iters=100))
        pathes.append('./result/{0}/{1}/g{2}/c{3}'.format(experience, dir_name, gamma, cc))
    mil.pluralParametersTrain(estimators, pathes=pathes, n_jobs=10)
    """
    """
    # no thread
    CC = [1000, 3000, 3500, 4000, 5000]
    for cc in CC:
        path = './result/{0}/{1}/g{2}/c{3}'.format(experience, dir_name, gamma, cc)
        C = cc
        #leave_one_person_out()
        #result_leave_one_person_out()
        check_identification_func_max()
    """

