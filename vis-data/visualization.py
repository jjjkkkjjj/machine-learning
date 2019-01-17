import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import glob
import cv2

def visualization(filepath):
    filepathes = sorted(glob.glob(filepath))
    for filepath in filepathes:



if __name__ == '__main__':
    visualization('../bag/joint2img/motempl/*.mp4')