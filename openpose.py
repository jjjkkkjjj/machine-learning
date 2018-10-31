import sys
import subprocess

import os
import glob
import json
import cv2
import numpy as np
import shutil

Lines = [[1, 0],[1, 2],[2, 3],[3, 4],[1, 5],[5, 6],[6, 7],[1, 8],[8, 9],[9, 10],[1, 11],[11, 12],
         [12, 13], [0, 14],[14, 16],[0, 15],[15, 17]]

class OpenPose:
    def __init__(self): pass

    def get_from_openpose(self, videosdir=None, extension=".MP4"):
        if videosdir is None:
            videosdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'video/')
        videosdir = os.path.join(videosdir, '*{0}'.format(extension))

        self.__videoslist = sorted(glob.glob(videosdir))
        # print(self.__videoslist)

        print("start to get data from openpose....")
        with open('2d-data/video_dir_info.txt', 'w') as f:
            f.write(videosdir)
        videosdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temporal_files/')

        for num, videopath in enumerate(self.__videoslist):
            print("processing... {0}/{1}".format(num, len(self.__videoslist)))

            # delete all json files
            shutil.rmtree('./temporal_files')
            os.mkdir('./temporal_files')
            with open('./.gitkeep', 'w') as gitkeep:
                gitkeep.write("")
            args = ["bash", "./get-from-openpose.sh", videopath, videosdir]
            out = subprocess.check_output(args)

            # print(out)
            self.__videopath = videopath

            video = cv2.VideoCapture(self.__videopath)
            self.__height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.__width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.__frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.__frame_rate = video.get(cv2.CAP_PROP_FPS)
            video.release()

            self.__json2csv(self.__videopath.split('/')[-1])
        print("finish getting data!")
        print("saved csv data in 2d-data directory")

    def __json2csv(self, videoname):
        jsonfiles = sorted(glob.glob('temporal_files/*.json'))
        Data = []
        # set previous frame
        self.__previous_num = 0
        for num, jsonfile in enumerate(jsonfiles):
            with open(jsonfile, 'r') as fr:
                # people_dicts['people'][people num][data][joint index]
                people_dicts = json.load(fr)
                # if openpose detects more than 2 people, call __select_pitcher
                if num == 0:
                    self.__previous_num = 0
                    if len(people_dicts['people']) > 1:
                        Data.append(self.__select_pitcher_initial_frame(people_dicts))
                    else:
                        Data.append(people_dicts['people'][0]['pose_keypoints'])
                # num != 0
                elif len(people_dicts['people']) == 1:
                    self.__previous_num = num
                    Data.append(people_dicts['people'][0]['pose_keypoints'])
                else:
                    #print (num)
                    Data.append(self.__select_pitcher(people_dicts, Data[self.__previous_num - 1][0], Data[self.__previous_num - 1][1]))

        self.__check_video(Data, videoname)
        if len(Data) == 0 or len(Data) != self.__frame_num:
            print('{} is invalid video'.format(self.__videopath))
            return
        # make csv
        with open('2d-data/{0}.csv'.format(videoname.split('.')[0]), 'w') as fw:
            # header
            fw.write('{},\n'.format(self.__videopath))
            fw.write('width,{0},height,{1},frame_num,{2},fps,{3},\n'.format(self.__width, self.__height, self.__frame_num, self.__frame_rate))

            row_str = 'time,'
            for joint_ini_index in range(0, len(Data[0]), 3):
                row_str += 'x{0},y{0},c{0},'.format(int(joint_ini_index/3))
            row_str += '\n'
            fw.write(row_str)

            for frame in range(self.__frame_num):
                row_str = '{},'.format(frame)
                for joint_ini_index in range(0, len(Data[frame]), 3):
                    row_str += '{0},{1},{2},'.format(int(Data[frame][joint_ini_index]), int(Data[frame][joint_ini_index + 1]), float(Data[frame][joint_ini_index + 2]))
                row_str += '\n'
                fw.write(row_str)
        return


    def __check_video(self, Data, videoname):
        fourcc = cv2.VideoWriter_fourcc(*'MPEG')

        w = 600
        h = 400

        out = cv2.VideoWriter('2d-video/{0}'.format(videoname), fourcc, self.__frame_rate, (h, w))

        for frame in range(self.__frame_num):
            img = np.zeros((self.__height, self.__width, 3), np.uint8)

            for joint_ini_index in range(0, len(Data[frame]), 3):
                cv2.circle(img, (int(Data[frame][joint_ini_index]), int(Data[frame][joint_ini_index + 1])), 4, (255, 255, 255), -1)
            for line in Lines:
                cv2.line(img, (int(Data[frame][line[0]*3]), int(Data[frame][line[0]*3 + 1])),
                         (int(Data[frame][line[1]*3]), int(Data[frame][line[1]*3 + 1])), (255, 255, 255), 2)

            img = cv2.resize(img, (h, w))
            out.write(img)
            #cv2.imshow('check', img)
            k = cv2.waitKey(10)
            if k == ord('q'):
                break
                #exit()

        out.release()


    def __select_pitcher(self, people_dicts, head_x_prev_frame, head_y_prev_frame):
        head_x = []
        head_y = []
        for people_dict in people_dicts['people']:
            # get head position
            head_x.append(people_dict['pose_keypoints'][0])
            head_y.append(people_dict['pose_keypoints'][1])

        distance = np.array([pow((head_x[i] - head_x_prev_frame), 2) + pow((head_y[i] - head_y_prev_frame), 2) for i in range(len(head_x))])
        if len(distance) != 0:
            self.__previous_num += 1
            return people_dicts['people'][np.argmin(distance)]['pose_keypoints']
        else:
            return [0 for i in range(18*3)]

    def __select_pitcher_initial_frame(self, people_dicts):
        # select person is in center
        head_x = []
        head_y = []
        center_x = self.__width / 2
        center_y = self.__height / 2

        for people_dict in people_dicts['people']:
            # get head position
            head_x.append(people_dict['pose_keypoints'][0])
            head_y.append(people_dict['pose_keypoints'][1])

        distance = np.array([pow((head_x[i] - center_x), 2) + pow((head_y[i] - center_y), 2) for i in range(len(head_x))])

        return people_dicts['people'][np.argmin(distance)]['pose_keypoints']

    def __manual_select_pitcher_initial_frame(self, people_dicts):
        # user can select initial person id
        img = np.zeros((self.__height, self.__width, 3), np.uint8)
        for i,people_dict in enumerate(people_dicts['people']):
            # write person id
            cv2.putText(img, str(i), (int(people_dict['pose_keypoints'][0]) - 5,
                                           int(people_dict['pose_keypoints'][1]) - 5),
                        cv2.FONT_HERSHEY_PLAIN, 15, (255, 255, 255))
            # write joint point
            for joint in range(0, len(people_dict['pose_keypoints']), 3):
                x = int(people_dict['pose_keypoints'][joint])
                y = int(people_dict['pose_keypoints'][joint + 1])
                if x == 0 and y == 0:
                    continue
                cv2.circle(img, (x, y), 5, (255, 255, 255), -1)

        cv2.imshow("select id", img)
        button = cv2.waitKey()
        button = button - 48
        cv2.destroyAllWindows()

        return people_dicts['people'][button]['pose_keypoints']

    def manual_videofile(self, videopath):
        videosdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temporal_files/')
        videoname = videopath.split('/')[-1]
        # delete all json files
        shutil.rmtree('./temporal_files')
        os.mkdir('./temporal_files')
        args = ["bash", "./get-from-openpose.sh", videopath, videosdir]
        out = subprocess.check_output(args)
        # print(out)
        self.__videopath = videopath
        video = cv2.VideoCapture(self.__videopath)
        self.__height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.__width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.__frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.__frame_rate = video.get(cv2.CAP_PROP_FPS)
        video.release()

        # json2csv

        jsonfiles = sorted(glob.glob('temporal_files/*.json'))
        Data = []
        # set previous frame
        self.__previous_num = 0
        for num, jsonfile in enumerate(jsonfiles):
            with open(jsonfile, 'r') as fr:
                # people_dicts['people'][people num][data][joint index]
                people_dicts = json.load(fr)
                # if openpose detects more than 2 people, call __select_pitcher
                if num == 0:
                    self.__previous_num = 0
                    if len(people_dicts['people']) > 1:
                        Data.append(self.__manual_select_pitcher_initial_frame(people_dicts))
                    else:
                        Data.append(people_dicts['people'][0]['pose_keypoints'])
                # num != 0
                elif len(people_dicts['people']) == 1:
                    self.__previous_num = num
                    Data.append(people_dicts['people'][0]['pose_keypoints'])
                else:
                    # print (num)
                    Data.append(self.__select_pitcher(people_dicts, Data[self.__previous_num - 1][0],
                                                      Data[self.__previous_num - 1][1]))

        self.__check_video(Data, videoname)
        if len(Data) == 0 or len(Data) != self.__frame_num:
            print('{} is invalid video'.format(self.__videopath))
            return
        # make csv
        with open('2d-data/{0}.csv'.format(videoname.split('.')[0]), 'w') as fw:
            # header
            fw.write('{},\n'.format(self.__videopath))
            fw.write(
                'width,{0},height,{1},frame_num,{2},fps,{3},\n'.format(self.__width, self.__height, self.__frame_num,
                                                                       self.__frame_rate))

            row_str = 'time,'
            for joint_ini_index in range(0, len(Data[0]), 3):
                row_str += 'x{0},y{0},c{0},'.format(int(joint_ini_index / 3))
            row_str += '\n'
            fw.write(row_str)

            for frame in range(self.__frame_num):
                row_str = '{},'.format(frame)
                for joint_ini_index in range(0, len(Data[frame]), 3):
                    row_str += '{0},{1},{2},'.format(int(Data[frame][joint_ini_index]),
                                                     int(Data[frame][joint_ini_index + 1]),
                                                     float(Data[frame][joint_ini_index + 2]))
                row_str += '\n'
                fw.write(row_str)
        return

    def manual_videofiles(self, videopaths):
        videosdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temporal_files/')

        PEOPLE_DICT = {}
        VIDEO_DATA = {}
        print("using openpose now...\n")
        for num, videopath in enumerate(videopaths):
            sys.stdout.write("\rprocessing... {0}/{1}".format(num, len(videopaths)))
            sys.stdout.flush()
            videoname = videopath.split('/')[-1]
            # delete all json files
            shutil.rmtree('./temporal_files')
            os.mkdir('./temporal_files')
            args = ["bash", "./get-from-openpose.sh", videopath, videosdir]
            out = subprocess.check_output(args)
            # print(out)

            video_data = {}
            video = cv2.VideoCapture(videopath)
            video_data['videoname'] = videoname
            video_data['height'] = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_data['width'] = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_data['frame_num'] = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            video_data['frame_rate'] = video.get(cv2.CAP_PROP_FPS)
            VIDEO_DATA[videopath] = video_data
            video.release()

            jsonfiles = sorted(glob.glob('temporal_files/*.json'))
            people_dicts = []
            for frame, jsonfile in enumerate(jsonfiles):
                with open(jsonfile, 'r') as fr:
                    # people_dicts['people'][people num][data][joint index]
                    people_dicts.append(json.load(fr))
            PEOPLE_DICT[videopath] = people_dicts

        print("finished estimating joint coordinetes!")
        print("select person you want")

        for videopath in videopaths:
            self.__videopath = videopath
            videoname = VIDEO_DATA[videopath]['videoname']
            self.__height = VIDEO_DATA[videopath]['height']
            self.__width = VIDEO_DATA[videopath]['width']
            self.__frame_num = VIDEO_DATA[videopath]['frame_num']
            self.__frame_rate = VIDEO_DATA[videopath]['frame_rate']

            Data = []
            print("select person you want in {0}".format(videoname))
            for num in range(len(PEOPLE_DICT[videopath])):
                people_dicts = PEOPLE_DICT[videopath][num]
                if num == 0:
                    self.__previous_num = 0
                    if len(people_dicts['people']) > 1:
                        Data.append(self.__manual_select_pitcher_initial_frame(people_dicts))
                    else:
                        Data.append(people_dicts['people'][0]['pose_keypoints'])
                # num != 0
                elif len(people_dicts['people']) == 1:
                    self.__previous_num = num
                    Data.append(people_dicts['people'][0]['pose_keypoints'])
                else:
                    # print (num)
                    Data.append(self.__select_pitcher(people_dicts, Data[self.__previous_num - 1][0],
                                                      Data[self.__previous_num - 1][1]))

            self.__check_video(Data, videoname)
            if len(Data) == 0 or len(Data) != self.__frame_num:
                print('{} is invalid video'.format(self.__videopath))
                return
            # make csv
            with open('2d-data/{0}.csv'.format(videoname.split('.')[0]), 'w') as fw:
                # header
                fw.write('{},\n'.format(self.__videopath))
                fw.write(
                    'width,{0},height,{1},frame_num,{2},fps,{3},\n'.format(self.__width, self.__height,
                                                                           self.__frame_num,
                                                                           self.__frame_rate))

                row_str = 'time,'
                for joint_ini_index in range(0, len(Data[0]), 3):
                    row_str += 'x{0},y{0},c{0},'.format(int(joint_ini_index / 3))
                row_str += '\n'
                fw.write(row_str)

                for frame in range(self.__frame_num):
                    row_str = '{},'.format(frame)
                    for joint_ini_index in range(0, len(Data[frame]), 3):
                        row_str += '{0},{1},{2},'.format(int(Data[frame][joint_ini_index]),
                                                         int(Data[frame][joint_ini_index + 1]),
                                                         float(Data[frame][joint_ini_index + 2]))
                    row_str += '\n'
                    fw.write(row_str)