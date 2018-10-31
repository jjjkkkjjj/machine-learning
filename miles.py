from __future__ import print_function
from data import Data
import os
import glob
import csv
import numpy as np
from MIL.miles import MILES
import random
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
#from sklearn.grid_search import GridSearchCV
from parameter_tune import tune
from frame_detector import frame_detector, bag2video, plusvideo
# activate milespy35
import threading
import cv2

method = 'img'
dir_name = 'img-miles'
kernel = 'rbf'
lamb = 0.2
mu = 0.7
gamma = 0.0013
C = 5000
sample_num_per_label = 0
experience = '2018'
path = './result/{0}/{1}/g{2}/mu{3}lamb{4}'.format(experience, dir_name, gamma, mu, lamb)
dicimate = 4
person = []

def mean_for_align(data_list, standard_joint_number=0, standard_frame=0):
    data_list = np.array(data_list)
    X = []
    Y = []
    for data_ in data_list:
        X.append(data_.x[standard_joint_number][standard_frame])
        Y.append(data_.y[standard_joint_number][standard_frame])
    X = np.array(X)
    Y = np.array(Y)

    return {'x':np.nanmean(X), 'y':np.nanmean(Y), 'joint':standard_joint_number, 'frame':standard_frame}

def read_data(notnantime=False, method='real', savemotiontemp=False):
    global person
    person = []
    # read hard
    with open('2d-data/video_dir_info.txt', 'rt') as f:
        videosdir = f.readline()

    with open('video/{0}/hard-video.csv'.format(experience), 'r') as f:
        hard_csvfiles_ = f.read().split('\n')[:-1]

    with open('video/{0}/easy-video.csv'.format(experience), 'r') as f:
        easy_csvfiles_ = f.read().split('\n')[:-1]

    #hard_csvfiles = ['C' + file.split(',')[0] + '.MP4' for file in hard_csvfiles_]
    hard_csvfiles = [file.split(',')[0] + '.mp4' for file in hard_csvfiles_]
    hard_hand = [file.split(',')[1][0] for file in hard_csvfiles_]
    hard_person = [file.split(',')[1][1] for file in hard_csvfiles_]
    #easy_csvfiles = ['C' + file.split(',')[0] + '.MP4' for file in easy_csvfiles_]
    easy_csvfiles = [file.split(',')[0] + '.mp4' for file in easy_csvfiles_]
    easy_hand = [file.split(',')[1][0] for file in easy_csvfiles_]
    easy_person = [file.split(',')[1][1] for file in easy_csvfiles_]

    csvfiles = sorted(glob.glob('2d-data/*.csv'))
    #csvfiles = sorted(glob.glob('old-data/__2d-data-noncut/*.csv'))
    # mean
    if method in ['align', 'dirvec', 'img']:
        data_list = []
        for index, csvfile in enumerate(csvfiles):
            with open(csvfile, 'rt') as f:
                reader = csv.reader(f)
                videopath = next(reader)[0]
                videoname = videopath.split('/')
                videoname = videoname[-1]

                header = next(reader)

                time_row = [row for row in reader]

                data_list.append(Data(videopath, header[1], header[3], header[5], header[7], time_row))

        mean = mean_for_align(data_list)
        del data_list

    bags = []  # tuple [instance(time), feature vector]
    labels = []
    csvname_list = []

    for index, csvfile in enumerate(csvfiles):
        with open(csvfile, 'rt') as f:
            reader = csv.reader(f)
            videopath = next(reader)[0]
            videoname = videopath.split('/')
            videoname = videoname[-1]

            header = next(reader)

            time_row = [row for row in reader]

            """
            x, y, c, times = data.elim_nan()

            if videoname in hard_csvfiles:
                bags.append(np.array([[x[int(i / 2)][time] if i % 2 == 0 else y[int(i / 2)][time] for i in range(len(x) * 2)] for time in times]))
                labels.append(1)

            elif videoname in easy_csvfiles:
                bags.append(np.array([[x[int(i / 2)][time] if i % 2 == 0 else y[int(i / 2)][time] for i in range(len(x) * 2)] for time in times]))
                labels.append(-1)
            """
            #data.nonantimes(update=True)
            if method == 'real':
                if videoname in hard_csvfiles:
                    labels.append(1)

                elif videoname in easy_csvfiles:
                    labels.append(-1)

                else:
                    continue

                data = Data(videopath, header[1], header[3], header[5], header[7], time_row)
                if not data.norm(save=False):
                    labels = labels[:-1]
                    continue
                data.interpolate('linear', True, False)
                flags = data.nanflags()
                bag = data.bag(None, 'x', 'y', nanflag=flags)
                bags.append(bag)
                csvname_list.append(csvfile)

            elif method == 'binary':
                if videoname in hard_csvfiles:
                    labels.append(1)

                elif videoname in easy_csvfiles:
                    labels.append(-1)

                else:
                    continue
                data = Data(videopath, header[1], header[3], header[5], header[7], time_row)
                if not data.norm(save=False):
                    labels = labels[:-1]
                    continue
                data.interpolate('linear', True, False)
                flags = data.nanflags()
                binary = data.binary()
                bag = data.bag(None, binary=binary, nanflag=flags)
                bags.append(bag)
                csvname_list.append(csvfile)

            elif method == 'mirror':
                if videoname in hard_csvfiles:
                    labels.append(1)
                    data = Data(videopath, header[1], header[3], header[5], header[7], time_row,
                                hand=hard_hand[hard_csvfiles.index(videoname)])

                elif videoname in easy_csvfiles:
                    labels.append(-1)
                    data = Data(videopath, header[1], header[3], header[5], header[7], time_row,
                                hand=easy_hand[easy_csvfiles.index(videoname)])

                else:
                    continue

                if not data.norm(save=False):
                    labels = labels[:-1]
                    continue
                data.interpolate('linear', True, False)
                data.mirror(mirrorFor='l')
                flags = data.nanflags()
                bag = data.bag(None, 'x', 'y', nanflag=flags)
                bags.append(bag)
                csvname_list.append(csvfile)

            elif method == 'align':
                if videoname in hard_csvfiles:
                    labels.append(1)
                    data = Data(videopath, header[1], header[3], header[5], header[7], time_row,
                                hand=hard_hand[hard_csvfiles.index(videoname)], dicimate=5)

                elif videoname in easy_csvfiles:
                    labels.append(-1)
                    data = Data(videopath, header[1], header[3], header[5], header[7], time_row,
                                hand=easy_hand[easy_csvfiles.index(videoname)], dicimate=5)

                else:
                    continue

                if not data.norm(save=False, mean_for_alignment=mean):
                    labels = labels[:-1]
                    continue
                data.interpolate('linear', True, False)
                data.mirror(mirrorFor='l')
                flags = data.nanflags()
                bag = data.bag([16, 17], 'x', 'y', nanflag=flags)
                #bag = data.bag([16, 17], 'x', 'y')
                bags.append(bag)
                csvname_list.append(csvfile)

            elif method == 'dirvec':
                if videoname in hard_csvfiles:
                    labels.append(1)
                    data = Data(videopath, header[1], header[3], header[5], header[7], time_row,
                                hand=hard_hand[hard_csvfiles.index(videoname)])

                elif videoname in easy_csvfiles:
                    labels.append(-1)
                    data = Data(videopath, header[1], header[3], header[5], header[7], time_row,
                                hand=easy_hand[easy_csvfiles.index(videoname)])

                else:
                    continue

                if not data.norm(save=False, mean_for_alignment=mean):
                    labels = labels[:-1]
                    continue

                data.interpolate('linear', update=True, save=False)
                data.mirror(mirrorFor='l')
                flags = data.nanflags(nonnanflag=-1, nanflag=1)
                directionx, directiony, _ = data.direction_vector(elim_outlier=False, save=False, filter=False)

                #features = data.interpolate_dir('linear', True, dirx=directionx, diry=directiony, length=length)
                #features['nanflag'] = flags

                #bag = data.bag(features)
                bag = data.bag([14, 16], dirx=directionx, diry=directiony, nanflags=flags)
                bags.append(bag)
                csvname_list.append(csvfile)

            elif method == 'img':
                if videoname in hard_csvfiles:
                    labels.append(1)
                    data = Data(videopath, header[1], header[3], header[5], header[7], time_row,
                                hand=hard_hand[hard_csvfiles.index(videoname)])
                    person.append(data.hand+hard_person[hard_csvfiles.index(videoname)])

                elif videoname in easy_csvfiles:
                    labels.append(-1)
                    data = Data(videopath, header[1], header[3], header[5], header[7], time_row,
                                hand=easy_hand[easy_csvfiles.index(videoname)])
                    person.append(data.hand+easy_person[easy_csvfiles.index(videoname)])

                else:
                    continue

                if not data.norm(save=False, mean_for_alignment=mean):
                    labels = labels[:-1]
                    person = person[:-1]
                    #print(videoname)
                    continue

                data.mirror(mirrorFor='l')
                imgs = data.joint2img(1, save=False)

                imgs = data.motion_history(imgs, dicimate, save=savemotiontemp)
                features = data.img2featurevector(imgs)
                bag = data.bag(None, **features)
                bags.append(bag)
                csvname_list.append(csvfile)
            else:
                raise ValueError('{0} is invalid method'.format(method))

    labels = np.array(labels, dtype=float)

    if notnantime:
        return bags, labels, csvname_list
    else:
        return bags, labels

def my_scorer(estimator, x, y):
    yPred = np.sign(estimator.predict(x, instancePrediction=False))
    a = accuracy_score(y, yPred)
    return a

def main():# read hard and easy
    bags, labels = read_data(method=method, savemotiontemp=False)

    print('positive:{0}\nnegative:{1}'.format(np.sum(np.array(labels)==1), np.sum(np.array(labels)==-1)))

    print(np.sum(np.array([len(bag) for bag in bags]))) # instance number
    #exit()
    mkdir()

    if sample_num_per_label == 0:# all
        indexes = [i for i in range(len(bags))]
    else:
        posiindex = np.where(np.array(labels) == 1)[0]
        negaindex = np.where(np.array(labels) == -1)[0]
        posiindex = random.sample(posiindex, sample_num_per_label)
        negaindex = random.sample(negaindex, sample_num_per_label)

        indexes = random.sample(np.concatenate((posiindex, negaindex)), 2*sample_num_per_label)
    bags = [bags[index] for index in indexes]
    labels = [labels[index] for index in indexes]

    #mkdir()
    classifier = MILES(lamb=lamb, mu=mu, similarity=kernel, gamma=gamma, C=C, verbose=True)
    #classifier = joblib.load('MILES.pkl.cmp')
    classifier.fit(bags, labels)

    joblib.dump(classifier, path + '/MILES.pkl.cmp', compress=True)
    predictions, instance_labels = classifier.predict(bags, instancePrediction=True)
    with open('{0}/parameter.txt'.format(path), 'a') as f:
        print(labels)
        print(np.sign(predictions))
        print("Accuracy: {0}".format(np.average(labels == np.sign(predictions)) * 100))
        print(indexes)
        print(labels, file=f)
        print(np.sign(predictions), file=f)
        print("Accuracy: {0}".format(np.average(labels == np.sign(predictions)) * 100), file=f)
        print(indexes, file=f)

def mkdir():
    import os
    if not os.path.isdir('{0}'.format(path)):
        os.makedirs('{0}'.format(path))
        os.makedirs('{0}/nonzero_image'.format(path))
        os.makedirs('{0}/func_plus'.format(path))
        os.makedirs('{0}/bag_video'.format(path))
        os.makedirs('{0}/leave-one-person-out'.format(path))
        os.makedirs('{0}/leave-one-person-out_result'.format(path))



def check_important_feature_frame():
    bags, labels, csvnamelist = read_data(method=method, notnantime=True, savemotiontemp=False)

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
    for bag_i, bag in enumerate(bags):
        fin = ini + len(bag)

        weights = classifier.w_[ini:fin]
        indices_nonzeroFrame = np.where(np.abs(weights) > classifier.tol)[0] * dicimate
        with open(csvnamelist[bag_i], 'r') as f:
            videopath = f.readline().split(',')[0]

            videoname = videopath.split('/')[-1][:-4]
            print("videoname: {0}".format(videoname))

        print("nonzero frames: {0}".format(indices_nonzeroFrame))

        with open(csvnamelist[bag_i], 'r') as f:
            video_path = f.readline().split(',')[0]

            video = cv2.VideoCapture(video_path)

            for frame in indices_nonzeroFrame:
                video.set(cv2.CAP_PROP_POS_FRAMES, frame)

                ret, img = video.read()

                videoname = path.split('/')[-1][:-4]

                cv2.imwrite("{0}/nonzero_image/{1}_{2}.jpg".format(path, videoname, weights[int(frame/dicimate)]), img)

            video.release()
        #bag2video(csvnamelist[bag_i], nontimelist[bag_i], path)
        ini = fin

def search_hyperparameter(savemotiontmp, ini, fin, step):
    global gamma, path
    bags, labels = read_data(method=method, savemotiontemp=savemotiontmp)

    print(len(bags))
    indexes = random.sample([j for j in range(len(bags))], 30)
    train_bags = [bags[index] for index in range(len(bags)) if index not in indexes]
    train_labels = [labels[index] for index in range(len(bags)) if index not in indexes]
    test_bags = [bags[index] for index in indexes]
    test_labels = [labels[index] for index in indexes]

    text = '{0}\n'.format(len(bags))
    text_, K, best_g = tune(ini, fin, step, train_bags, test_bags)
    text += text_

    gamma = best_g
    path = './result/{0}/{1}/g{2}/'.format(experience, dir_name, gamma)
    if not os.path.isdir(path):
        os.makedirs('{0}'.format(path))
    with open('{0}/parameter.txt'.format(path), 'w') as f:
        f.write(text)
        print(K, file=f)
    exit()
    """
    clf = MILES.MILES(max_iters=100, verbose=True)
    tuned_parameters = [
        # {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        #{'C': [1, 10], 'kernel': ['rbf'], 'gamma': [0.04]},
         {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': [0.001, 0.0001]},
        # {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.001, 0.0001]}
    ]

    grid_search = GridSearchCV(clf, tuned_parameters, cv=5, scoring=my_scorer, verbose=True)
    grid_search.fit(bags, labels)
    print("score")
    print grid_search.grid_scores_
    print "best"
    print grid_search.best_params_
    """

# no need!
def cross_validation():
    bags, labels = read_data(method=method)
    # check if exsisting directory
    import os
    if not os.path.isdir(path):
        raise IOError('{0} is not directory'.format(path))
    else:
        with open('{0}/cross_validation_info.txt'.format(path), 'w') as f:
            f.write('kernel:{0}\ngamma:{1}\nC:{2}\n'.format(kernel, gamma, C))

    for i in range(5):
        posiindex = np.where(np.array(labels) == 1)[0]
        negaindex = np.where(np.array(labels) == -1)[0]
        posiindex = random.sample(posiindex, sample_num_per_label)
        negaindex = random.sample(negaindex, sample_num_per_label)

        with open('{0}/cross_validation_info.txt'.format(path), 'a') as f:
            indexes = random.sample(np.concatenate((posiindex, negaindex)), 2 * sample_num_per_label)
            print("iter: {0}".format(i), file=f)
            print("train index:", file=f)
            print(indexes, file=f)
            # indexes = random.sample([j for j in range(len(bags))], 10)
            train_bags = [bags[index] for index in range(len(bags)) if index not in indexes]
            train_labels = [labels[index] for index in range(len(bags)) if index not in indexes]
            test_bags = [bags[index] for index in indexes]
            test_labels = [labels[index] for index in indexes]

            classifier = MILES(lamb=lamb, mu=mu, similarity=kernel, gamma=gamma, C=C, verbose=True)
            classifier.fit(train_bags, train_labels)
            predictions = classifier.predict(test_bags)
            joblib.dump(classifier, path + '/MILES{0}.pkl.cmp'.format(i), compress=True)
            # print(test_labels)
            # print(np.sign(predictions))
            # print("Accuracy: {0}".format(np.average(test_labels == np.sign(predictions)) * 100))

            print(test_labels, file=f)
            print(np.sign(predictions), file=f)
            print("Accuracy: {0}".format(np.average(test_labels == np.sign(predictions)) * 100), file=f)

def leave_one_put(check=False, threadnum=5):
    bags, labels, csvnamelist = read_data(method=method, notnantime=True, savemotiontemp=False)

    import os
    if not os.path.isdir(path + '/leave-one-out'):
        if not os.path.isdir(path):
            os.mkdir(path)
        os.mkdir(path + '/leave-one-out')

    if check:
        Path = path + '/leave-one-out_result/'
        if not os.path.isdir(Path + '/func_max_image/'):
            os.mkdir(Path + '/func_max_image/')

        classifier = joblib.load(Path + '/MILES.pkl.cmp')
        indexes = [i for i in range(len(bags))]

        bags = [bags[index] for index in indexes]
        labels = [labels[index] for index in indexes]

        # print classifier.get_params()
        bag_predictions, instance_predictions = classifier.predict(bags, instancePrediction=True)
        with open('{0}/leave-one-out_result/result.txt'.format(path), 'w') as f:
            print(labels)
            print(labels, file=f)
            print(np.sign(bag_predictions))
            print(np.sign(bag_predictions), file=f)
            print("Accuracy: {0}".format(np.average(labels == np.sign(bag_predictions)) * 100))
            print("Accuracy: {0}".format(np.average(labels == np.sign(bag_predictions)) * 100), file=f)

        ini = 0
        instance_index_list = []
        max = []
        for bag_i, bag in zip(indexes, bags):
            fin = ini + len(bag)
            ident_func_result = np.array(instance_predictions[ini:fin])
            # show graph
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.plot(np.arange(fin - ini), instance_predictions[ini:fin], '-')
            # plt.show()
            with open(csvnamelist[bag_i], 'r') as f:
                videopath = f.readline().split(',')[0]

                videoname = videopath.split('/')[-1][:-4]
                plt.savefig(Path + '/func_max_image/' + videoname + '-func.png')

            max.append(np.max(ident_func_result))
            instance_index_list.append(np.argmax(ident_func_result))
            # plusvideo(csvnamelist[bag_i], ident_func_result, path)
            ini = fin

        bag_predictions = np.sign(bag_predictions)
        for index, bag_i in enumerate(indexes):
            with open(csvnamelist[bag_i], 'r') as f:
                videopath = f.readline().split(',')[0]

                videoname = videopath.split('/')[-1][:-4]
                print("videoname: {0}".format(videoname))
            # print("csvfile: {0}".format(csvnamelist[bag_i]))
            print("prediction labels: {0}".format(bag_predictions[index]))
            print("right labels: {0}".format(labels[index]))
            print("the maximum of identification func: {0}".format(max[index]))
            frame = instance_index_list[index] * dicimate
            print("maximum frame: {0}".format(frame))
            frame_detector(csvnamelist[bag_i], frame, Path)
            # bag2video(csvnamelist[bag_i], nontimelist[bag_i], path)

    else:
        for cnt, i in enumerate(range(0, len(bags), threadnum)):
            threadlist = []

            def work(threadcount):
                train_bags = [bags[j] for j in range(len(bags)) if i + threadcount != j]
                train_labels = [labels[j] for j in range(len(labels)) if i + threadcount != j]

                print('{0}/{1}'.format(i + threadcount + 1, len(bags)))
                if not os.path.isdir(path + '/leave-one-out/' + str(i + threadcount)):
                    os.mkdir(path + '/leave-one-out/' + str(i + threadcount))
                classifier = MILES(lamb=lamb, mu=mu, similarity=kernel, gamma=gamma, C=C, verbose=True)
                classifier.fit(train_bags, train_labels)
                predictions = classifier.predict(train_bags)
                joblib.dump(classifier, path + '/leave-one-out/{0}/MILES.pkl.cmp'.format(i + threadcount),
                            compress=True)
                # print(test_labels)
                # print(np.sign(predictions))
                # print("Accuracy: {0}".format(np.average(test_labels == np.sign(predictions)) * 100))
                with open('{0}/leave-one-out/{1}/output.txt'.format(path, i + threadcount), 'w') as f:
                    print(train_labels, file=f)
                    print(np.sign(predictions), file=f)
                    print("Accuracy: {0}".format(np.average(train_labels == np.sign(predictions)) * 100), file=f)

            if len(bags) - cnt * threadnum < threadnum:
                for thcnt in range(len(bags) - cnt * threadnum):
                    thread = threading.Thread(target=work, name=str(i + thcnt), args=([thcnt]))
                    threadlist.append(thread)
            else:
                for thcnt in range(threadnum):
                    thread = threading.Thread(target=work, name=str(i + thcnt), args=([thcnt]))
                    threadlist.append(thread)

            for thread in threadlist:
                thread.daemon = True
                thread.start()

            for thread in threadlist:
                thread.join()

            del threadlist

        result_leave_one_out(bags, labels)

def leave_one_person_out(threadnum=8, makenewfile=False):
    bags, labels, csvnamelist = read_data(method=method, notnantime=True, savemotiontemp=False)

    personlist = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'l1', 'l2', 'l3', 'l4']

    if makenewfile is not None:
        mkdir()

    for cnt, i in enumerate(range(0, len(personlist), threadnum)):
        threadlist = []

        def work(threadcount, leaveI):
            train_bags = [bags[j] for j in range(len(bags)) if not j in leaveI]
            train_labels = [labels[j] for j in range(len(labels)) if not j in leaveI]

            print('{0}/{1}'.format(i + threadcount + 1, len(personlist)))
            if not os.path.isdir(path + '/leave-one-person-out/' + str(i + threadcount)):
                os.mkdir(path + '/leave-one-person-out/' + str(i + threadcount))
            classifier = MILES(lamb=lamb, mu=mu, similarity=kernel, gamma=gamma, C=C, verbose=True)
            classifier.fit(train_bags, train_labels)
            predictions = classifier.predict(train_bags, instancePrediction=False)
            joblib.dump(classifier, path + '/leave-one-person-out/{0}/MILES.pkl.cmp'.format(i + threadcount),
                        compress=True)
            # print(test_labels)
            # print(np.sign(predictions))
            # print("Accuracy: {0}".format(np.average(test_labels == np.sign(predictions)) * 100))
            with open('{0}/leave-one-person-out/{1}/output.txt'.format(path, i + threadcount), 'w') as f:
                print(train_labels, file=f)
                print(np.sign(predictions), file=f)
                print("Accuracy: {0}".format(np.average(train_labels == np.sign(predictions)) * 100), file=f)

        if len(personlist) - cnt * threadnum < threadnum:
            for thcnt in range(len(personlist) - cnt * threadnum):
                leaveindexes = [j for j, p in enumerate(person) if p == personlist[i + thcnt]]
                thread = threading.Thread(target=work, name=str(i + thcnt), args=([thcnt, leaveindexes]))
                threadlist.append(thread)
        else:
            for thcnt in range(threadnum):
                leaveindexes = [j for j, p in enumerate(person) if p == personlist[i + thcnt]]
                thread = threading.Thread(target=work, name=str(i + thcnt), args=([thcnt, leaveindexes]))
                threadlist.append(thread)

        for thread in threadlist:
            thread.daemon = True
            thread.start()

        for thread in threadlist:
            thread.join()

        del threadlist

    result_leave_one_person_out(bags, labels)

def result_leave_one_person_out(bags=None, labels=None):
    if bags is None:
        bags, labels = read_data(method=method, savemotiontemp=False)

    personlist = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'l1', 'l2', 'l3', 'l4']

    mkdir()

    dirs = sorted(glob.glob(path + '/leave-one-person-out/*/'))
    #from sklearn.metrics import confusion_matrix
    #import matplotlib.pyplot as plt
    #import pandas as pd
    #import seaborn as sn

    true_lebel = []
    predict = []
    pos = int(0)
    pn = 0
    for dir in dirs:
        num = int(dir.split('/')[-2])

        classifier = joblib.load(dir + 'MILES.pkl.cmp')
        Index =  [j for j, p in enumerate(person) if p == personlist[num]]

        if len(Index) == 0:
            continue
        test_bags = [bags[j] for j in range(len(bags)) if j in Index]
        test_labels = [labels[j] for j in range(len(labels)) if j in Index]

        bag_predictions, instance_predictions = classifier.predict(test_bags, instancePrediction=True)
        tmp = np.sum(np.array(test_labels == np.sign(bag_predictions))).item()
        pos += tmp
        pn += len(Index)

        true_lebel.extend(test_labels)
        predict.extend(np.sign(bag_predictions).tolist())
    #print(len(true_lebel), len(predict))
    #cmx_data = confusion_matrix(true_lebel, predict)
    #plot_confusion_matrix(cmx_data, ['positive', 'negative'], normalize=False)
    """
    df_cmx = pd.DataFrame(cmx_data, index=['positive', 'negative'], columns=['positive', 'nagative'])
    plt.clf()
    plt.xlabel('Predicted', fontsize=26)
    plt.ylabel('True', fontsize=26)
    plt.title('Confusion Matrix', fontsize=26)
    plt.xticks([0, 1], ['positive', 'negative'], rotation=45, fontsize=26)
    plt.yticks([0, 1], ['positive', 'negative'], fontsize=26)
    plt.rcParams['font.size'] = 22
    #df_cmx.index.name = 'True'
    #df_cmx.columns.name = 'Predicted'

    fig, ax = plt.subplots(figsize=(10, 7))
    sn.heatmap(df_cmx, annot=True)
    plt.show()
    """
    with open('{0}/leave-one-person-out_result/result.txt'.format(path), 'w') as f:
        print('accuracy: {0}'.format(pos * 100.0 / pn))
        print('accuracy: {0}'.format(pos*100.0/pn), file=f)

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

def result_leave_one_out(bags=None, labels=None):
    if bags is None:
        bags, labels = read_data(method=method, savemotiontemp=False)

    if not os.path.exists('{0}/leave-one-out_result'.format(path)):
        os.mkdir('{0}/leave-one-out_result'.format(path))

    dirs = sorted(glob.glob(path + '/leave-one-out/*/'))
    p = 0
    for dir in dirs:
        num = int(dir.split('/')[-2])
        classifier = joblib.load(dir + 'MILES.pkl.cmp')
        bag_predictions, instance_predictions = classifier.predict([bags[num]], instancePrediction=True)

        if np.sign(bag_predictions)[0] == float(labels[num]):
            p += 1
    with open('{0}/leave-one-out_result/result.txt'.format(path), 'w') as f:
        print('accuracy: {0}'.format(p * 100.0 / len(dirs)))
        print('accuracy: {0}'.format(p*100.0/len(dirs)), file=f)

def gridsearch(savemotiontmp, params_grid):
    bags, labels = read_data(method=method, savemotiontemp=savemotiontmp)
    print(len(bags))
    classifier = MILES()

    gscv = GridSearchCV(classifier, params_grid, cv=2, scoring=my_scorer)
    gscv.fit(bags, labels)
    print(gscv.cv_results_)
    print('\n\nbest score is')
    print(gscv.best_params_)
    with open('result/{0}/gridsearch.txt'.format(dir_name), 'w') as f:
        print('{0}\n'.format(gscv.cv_results_), file=f)
        print('best parameters', file=f)
        print('{0}'.format(gscv.best_params_), file=f)
    #text = '{0}\n'.format(len(bags))
    #text_, K = tune(ini, fin, step, train_bags, test_bags)
    #text += text_
    #mkdir(text, K)
    exit()


if __name__ == '__main__':
    #main()
    search_hyperparameter(savemotiontmp=False, ini=0.0009, fin=0.002, step=0.0001)
    #gridsearch(savemotiontmp=False, params_grid=[{'gamma':[0.0013], 'mu':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'lamb':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'similarity':['rbf']}])
    #check_important_feature_frame()
    #cross_validation()
    #leave_one_put(check=False, threadnum=8)
    #leave_one_person_out(makenewfile=True)

    # i dont need 'global'
    """
    L = [0.23, 0.24]
    for l in L:
        lamb = l
        path = './result/{0}/{1}/g{2}/mu{3}lamb{4}'.format(experience, dir_name, gamma, mu, lamb)
        leave_one_person_out(makenewfile=True)
    """
    #result_leave_one_out()
    #pythonresult_leave_one_person_out()
