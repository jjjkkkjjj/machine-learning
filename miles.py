from __future__ import print_function
import numpy as np
from MIL.miles import MILES
from sklearn.externals import joblib
# activate milespy35
from mil import MIL
import cv2

method = 'img'
dir_name = 'normedimg-miles'
kernel = 'rbf'
lamb = 0.7
mu = 0.5
gamma = 0.0012
C = 5000 # pointless
experience = '2018'
positiveCsvFileName='hard-video.csv'
negativeCsvFileName='easy-video.csv'
path = './result/{0}/{1}/g{2}/mu{3}lamb{4}'.format(experience, dir_name, gamma, mu, lamb)
dicimate = 4

mil = MIL(method=method, experience=experience, dirName=dir_name, estimatorName='MILES')
mil.setData(positiveCsvFileName=positiveCsvFileName, negativeCsvFileName=negativeCsvFileName,
            saveMotionTmplate=False, dicimate=dicimate, videoExtension='mp4', csvExtension='csv')

def main():# read hard and easy
    estimator = MILES(lamb=lamb, mu=mu, similarity=kernel, gamma=gamma, C=C, verbose=True)
    mil.train(estimator=estimator, resultSuperDirPath=path)


def check_important_feature_frame():
    bags, labels, csvnamelist = mil.bags, mil.labels, mil.csvFilePaths

    # sort [p,p,p,p,n,n,n,n,n,n,n,n,n]
    Pbags_indices_of_bags = np.where(np.array(labels) == 1)[0]  # .shape[0] means l+
    Nbags_indices_of_bags = np.where(np.array(labels) == -1)[0]  # .shape means l-

    newbags, newlabels, newcsvnamelist = [], [], []
    for indexPositive in Pbags_indices_of_bags:
        newbags.append(bags[indexPositive])
        newlabels.append(1)
        newcsvnamelist.append(csvnamelist[indexPositive])
    for indexNegative in Nbags_indices_of_bags:
        newbags.append(bags[indexNegative])
        newlabels.append(-1)
        newcsvnamelist.append(csvnamelist[indexNegative])
    newlabels = np.array(newlabels)

    #indexes = [i for i in range(len(bags))]
    #bags = [bags[index] for index in indexes]
    #labels = [labels[index] for index in indexes]

    #classifier = joblib.load('result/flag/flag-parameter-c10000.cmp')
    classifier = joblib.load(path + '/MILES.pkl.cmp')
    """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.ylabel('weight')
    plt.plot(np.arange(classifier.w_.size), classifier.w_)
    #print (classifier.w_)
    plt.show()
    """
    ini = 0
    for bag_i, bag in enumerate(newbags):
        fin = ini + len(bag)

        weights = classifier.w_[ini:fin]
        indices_nonzeroFrame = np.where(np.abs(weights) > classifier.tol)[0] * dicimate
        with open(newcsvnamelist[bag_i], 'r') as f:
            videopath = f.readline().split(',')[0]

            videoname = videopath.split('/')[-1][:-4]
            print("videoname: {0}".format(videoname))

        print("nonzero frames: {0}".format(indices_nonzeroFrame))

        with open(newcsvnamelist[bag_i], 'r') as f:
            video_path = f.readline().split(',')[0]

            video = cv2.VideoCapture(video_path)

            for frame in indices_nonzeroFrame:
                video.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)

                ret, img = video.read()
                # this may occur due to dicimate
                # example: len(bag)=61, dicimate=4, videoframe=240
                # in other words, if ret is False, last frame should be extracted
                if not ret:
                    video.set(cv2.CAP_PROP_POS_FRAMES, int(video.get(cv2.CAP_PROP_FRAME_COUNT)) - 2)
                    ret, img = video.read()
                videoname = path.split('/')[-1][:-4]

                cv2.imwrite("{0}/nonzero_image/{1}_{2}.jpg".format(path, videoname, weights[int(frame/dicimate)]), img)

            video.release()
        #bag2video(csvnamelist[bag_i], nontimelist[bag_i], path)
        ini = fin

def search_hyperparameter(ini, fin, step, randomSampledTrainRatio):
    mil.searchGamma(ini=ini, fin=fin, step=step, randomSampledTrainRatio=randomSampledTrainRatio)


def gridsearch(params_grid, cv=2):
    estimator = MILES()
    mil.gridSearch(estimator=estimator, params_grid=params_grid, cv=cv)


def cross_validation():
    estimator = MILES(lamb=lamb, mu=mu, similarity=kernel, gamma=gamma, C=C, verbose=True)
    mil.crossValidation(estimator, 5, path)

def leave_one_out(n_jobs=8):
    estimator = MILES(lamb=lamb, mu=mu, similarity=kernel, gamma=gamma, C=C, verbose=True)
    mil.leaveOneOut(estimator=estimator, resultSuperDirPath=path, n_jobs=n_jobs, trainAcc=True)

def leave_one_person_out(n_jobs=8, resultVis=False):
    if resultVis:
        resultvis = 'estimator_.w_'
    else:
        resultvis = None
    estimator = MILES(lamb=lamb, mu=mu, similarity=kernel, gamma=gamma, C=C, verbose=True)
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

def visualization():
    bags = mil.bags
    classifier = joblib.load(path + '/MILES.pkl.cmp')
    indices = [i for i in range(len(bags))]

    ini = 0
    weights = []
    for index, bag_i in enumerate(indices):
        fin = ini + len(bags[bag_i])
        weights.append(np.array(classifier.w_[ini: fin]))
        ini = fin

    mil.visualization(weights, indices, mode='bar', resultSuperDirPath=path)

if __name__ == '__main__':
    #main()
    #search_hyperparameter(savemotiontmp=False, ini=0.005, fin=0.01, step=0.0001)
    #gridsearch(savemotiontmp=False, params_grid=[
    #    {'gamma': [0.0012], 'mu': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #     'lamb': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'similarity': ['rbf']}])
    #check_important_feature_frame()
    #cross_validation()
    #visualization()
    #leave_one_out(n_jobs=12)
    #leave_one_person_out(n_jobs=10)

    # no need 'global'

    # use thread
    """
    estimators = []
    pathes = []
    L = [0.5, 0.55, 0.6, 0.65, 0.7]
    for l in L:
        lamb = l
        estimators.append(MILES(lamb=lamb, mu=mu, similarity=kernel, gamma=gamma, C=C, verbose=True))
        pathes.append('./result/{0}/{1}/g{2}/mu{3}lamb{4}'.format(experience, dir_name, gamma, mu, lamb))
    mil.pluralParametersTrain(estimators, pathes=pathes, n_jobs=10)

    # no thread
    """
    """
    lamb = 0.5
    path = './result/{0}/{1}/g{2}/mu{3}lamb{4}'.format(experience, dir_name, gamma, mu, lamb)
    clf = joblib.load(path + '/MILES.pkl.cmp')
    bags, labels, csvnamelist = mil.bags, mil.labels, mil.csvFilePaths
    Pbags_indices_of_bags = np.where(np.array(labels) == 1)[0]  # .shape[0] means l+
    Nbags_indices_of_bags = np.where(np.array(labels) == -1)[0]  # .shape means l-

    newbags, newlabels, newcsvnamelist = [], [], []
    for indexPositive in Pbags_indices_of_bags:
        newbags.append(bags[indexPositive])
        newlabels.append(1)
        newcsvnamelist.append(csvnamelist[indexPositive])
    for indexNegative in Nbags_indices_of_bags:
        newbags.append(bags[indexNegative])
        newlabels.append(-1)
        newcsvnamelist.append(csvnamelist[indexNegative])
    newlabels = np.array(newlabels)

    predicted = clf.predict(newbags, instancePrediction=False)
    print(np.average(np.sign(predicted) == np.array(newlabels))*100)
    """

    L = [0.2, 0.4, 0.6, 0.8]
    for l in L:
        lamb = l
        path = './result/{0}/{1}/g{2}/mu{3}lamb{4}'.format(experience, dir_name, gamma, mu, lamb)
        leave_one_person_out(n_jobs=10, resultVis=False)
        #main()
        check_important_feature_frame()
