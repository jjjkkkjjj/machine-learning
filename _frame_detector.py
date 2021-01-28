import cv2
import numpy as np

def frame_detector(csvfile, frame, dirpath):
    with open(csvfile, 'r') as f:
        path = f.readline().split(',')[0]

        video = cv2.VideoCapture(path)

        video.set(cv2.CAP_PROP_POS_FRAMES, frame)

        ret, img = video.read()

        videoname = path.split('/')[-1][:-4]

        cv2.imwrite("{0}/func_max_image/".format(dirpath) + videoname + "-max.jpg", img)

        video.release()


def bag2video(csvfile, nontimelist, dirpath):
    with open(csvfile, 'r') as f:
        path = f.readline().split(',')[0]

        videoname = path.split('/')[-1][:-4]

        reader = cv2.VideoCapture(path)

        fps = reader.get(cv2.CAP_PROP_FPS)
        height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        writer = cv2.VideoWriter("{0}/bag_video/{1}nononly.mp4".format(dirpath, videoname), fourcc, fps, (width, height))

        for time in nontimelist:
            reader.set(cv2.CAP_PROP_POS_FRAMES, time)
            ret, img = reader.read()

            writer.write(img)

        writer.release()
        reader.release()

def plusvideo(csvfile, instancepredictions, dirpath):
    with open(csvfile, 'r') as f:
        timelist = np.where(np.array(instancepredictions) > 0)[0]
        if timelist.shape[0] == 0:
            return

        path = f.readline().split(',')[0]

        videoname = path.split('/')[-1][:-4]

        reader = cv2.VideoCapture(path)

        fps = reader.get(cv2.CAP_PROP_FPS)
        height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        writer = cv2.VideoWriter("{0}/func_plus/{1}plus.mp4".format(dirpath, videoname), fourcc, fps, (width, height))
        print("saved to {0}/func_plus/{1}plus.mp4".format(dirpath, videoname))

        for time in timelist:
            reader.set(cv2.CAP_PROP_POS_FRAMES, time)
            ret, img = reader.read()

            writer.write(img)

        writer.release()
        reader.release()