import platform
if platform.system() == 'Darwin':
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np
from sklearn.manifold import TSNE
import glob
import os, sys
import cv2

"""
filepath = '../bag/joint2img/motempl/*.mp4'
method = 't-sne' # hist or t-sne
experience = '2018'
positiveCsvFileName='hard-video.csv'
negativeCsvFileName='easy-video.csv'
extension = '.mp4'
permode = 'person' # person or label
dirpath = 'result/'
"""

def visualization(imgsDict=None, **kwargs):
    filepath = '../bag/joint2img/motempl/*.mp4'
    method = 't-sne'  # hist or t-sne
    experience = '2018'
    positiveCsvFileName = 'hard-video.csv'
    negativeCsvFileName = 'easy-video.csv'
    extension = '.mp4'
    permode = 'label'  # person or label
    dirpath = 'result/'

    for key in list(kwargs.keys()):
        exec('{0} = kwargs.pop(key, {0})'.format(key))

    # make result directory
    dirpath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           dirpath, method)
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)

    if imgsDict is None:
        pathes = sorted(glob.glob(filepath))
        # print(filepathes)

        # assign label
        labels = []

        f = open(os.path.join('../video/', experience, positiveCsvFileName), 'r')
        poslines = f.readlines()
        for line in poslines:
            labels.append(str(line.split(',')[1]).strip())
        f.close()

        f = open(os.path.join('../video/', experience, negativeCsvFileName), 'r')
        neglines = f.readlines()
        for line in neglines:
            labels.append(str(line.split(',')[1]).strip())
        f.close()

        # mode
        if permode == 'label':
            labels = ['pos', 'neg']
            labelFiles = {label: [] for label in labels}
            for line in poslines:
                labelFiles['pos'].append(line.split(',')[0] + extension)
            for line in neglines:
                labelFiles['neg'].append(line.split(',')[0] + extension)
        elif permode == 'person':
            labels = list(set(labels))
            labelFiles = {label: [] for label in labels}
            for line in poslines:
                labelFiles[str(line.split(',')[1]).strip()].append(line.split(',')[0] + extension)
            for line in neglines:
                labelFiles[str(line.split(',')[1]).strip()].append(line.split(',')[0] + extension)
        else:
            raise NameError('{0} is invalid permode'.format(permode))

        imgsDict = {label: [] for label in labels}
        sys.stdout.write('\rreading files... 0%')
        sys.stdout.flush()
        for i, path in enumerate(pathes):
            # read video files
            video = cv2.VideoCapture(path)
            frameNum = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            imgs = []
            for frame in range(frameNum):
                ret, img = video.read()
                imgs.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
            video.release()

            # check label
            check(path, imgsDict, imgs, labelFiles)
            sys.stdout.write('\rreading files... {0}%'.format(int(100 * (i + 1) / len(pathes))))
            sys.stdout.flush()

        sys.stdout.write('\rfinished reading files\n')
        sys.stdout.flush()


    if method == 'hist':
        histogram(imgsDict, dirpath)
    elif method == 't-sne':
        tsne(imgsDict, dirpath)
    else:
        raise NameError("{0} is invalid method name".format(method))

    return

def check(path, imgsDict, imgs, labelFiles):
    labels = list(labelFiles.keys())

    # check label of filepath
    path_ = os.path.basename(path)
    for label in labels:
        if path_ in labelFiles[label]:
            imgsDict[label].append(imgs)
            return
    raise ValueError("{0} is not contained in {1}".format(path, labels))


def histogram(imgsDict, dirpath, step=10):
    for i, (label, Imgs) in enumerate(imgsDict.items()):
        sys.stdout.write('\rcalculate {0}\'s histogram, will save to {1} ...{2}/{3}'.format(label, dirpath, i + 1, len(imgsDict)))
        sys.stdout.flush()

        bins = np.arange(0, 255, step)
        hists = np.zeros(bins.size - 1)
        instanceNum = 0
        for imgs in Imgs:
            hist, bins = np.histogram(imgs, bins=bins)
            hists += hist
            instanceNum += len(imgs)
        hists /= instanceNum

        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.clf()
        plt.title('histogram per an instance in {0} bags'.format(label))
        plt.xlabel('pixcel value')
        plt.ylabel('Number')
        plt.bar(center, hists, align='center', width=width)
        for index, value in enumerate(hists):
            plt.text(index*step + 0.5*step, value + 3, str(int(value)))

        plt.savefig(os.path.join(dirpath, label + '.png'))

    sys.stdout.write('\rsaved histogram to {0}\n'.format(dirpath))
    sys.stdout.flush()

def tsne(imgsDict, dirpath):
    instanceNum = [0] * len(imgsDict)
    X = []
    colors = {label: hsv_to_rgb([i*1.0/len(imgsDict), 1.0, 1.0]) for i, label in enumerate(imgsDict.keys())}
    colorList = []
    for i, (label, Imgs) in enumerate(imgsDict.items()):
        X.append(np.vstack(Imgs))

        colorList.extend([colors[label]] * X[i].shape[0])
        instanceNum[i] += X[i].shape[0]

    X = np.vstack(X)
    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])

    sys.stdout.write(
        '\rcalculate t-sne values, will save to {0}'.format(dirpath))
    sys.stdout.flush()
    perplexities = [5, 30, 50]
    for i, perplexity in enumerate(perplexities):
        sys.stdout.write(
            '\rcalculate perplexity = {0}, will save to {1} ...{2}/{3}'.format(perplexity, dirpath, i + 1, len(perplexities)))
        sys.stdout.flush()
        tsne_ = TSNE(n_components=2, random_state=0, perplexity=perplexity)

        X_reduced = tsne_.fit_transform(X)
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=colorList)
        plt.savefig(os.path.join(dirpath, str(perplexity) + '.png'))

    sys.stdout.write('\rsaved t-sne to {0}\n'.format(dirpath))
    sys.stdout.flush()


if __name__ == '__main__':
    visualization()