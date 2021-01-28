import cv2, os
import numpy as np
from sklearn.metrics import accuracy_score

def show_video(runenv, path):
    if runenv == 'terminal':
        video = cv2.VideoCapture(path)
        while video.isOpened():
            ret, img = video.read()
            cv2.imshow('check', img)
            k = cv2.waitKey(10)
            if k == ord('q'):
                break
                #exit()
        video.release()
        cv2.destroyAllWindows()

    elif runenv == 'jupyter':
        import ipywidgets as wd
        from IPython.display import Video, display
        out = wd.Output(layout={'border': '1px solid black'})
        with out:
            display(Video(path, embed=True, mimetype='mp4'))


def check_and_create_dir(*dirs):
    path = os.path.join(*dirs)
    if not os.path.exists(path):
        os.makedirs(path)
    

def myScore(estimator, x, y):
    yPred = np.sign(estimator.predict(x, instancePrediction=False))
    acc = accuracy_score(y, yPred)
    return acc