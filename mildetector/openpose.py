import sys
import subprocess

import os, logging
import glob
import json
import cv2
import numpy as np
import shutil

# see https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
bone_lines_COCO = [
    [1, 0],[1, 2],[2, 3],[3, 4],[1, 5],[5, 6],[6, 7],
    [1, 8],[8, 9],[9, 10],
    [1, 11],[11, 12],[12, 13],
    [0, 14],[14, 16],[0, 15],[15, 17]
]

bone_lines_BODY_25 = [
    [1, 0],[1, 2],[2, 3],[3, 4],[1, 5],[5, 6],[6, 7],[1, 8],
    [8, 9],[9, 10],[10,11],[11,24],[11,22],[22,23],
    [8, 12],[12, 13],[13, 14],[14, 21],[14, 19],[19,20],
    [0, 15], [15, 17],[0, 16],[16, 18]
]

bonetypes_list = ['BODY_25', 'COCO']

class OpenPose:
    def __init__(self, runenv='terminal', binpath=None, debug=False, bonetype='BODY_25'): 
        if runenv == 'jupyter' and binpath is None:
            logging.warning('binpath is not set. Jupyter cannot load openpose.bin location by default.')
        if binpath:
            os.environ['PATH'] += ':' + binpath
        self.runenv = runenv
        self.rootdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        self.videoinfo = VideoInfo(videopath=None)
        self.debug = debug
        if self.debug:
            logging.basicConfig()
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.basicConfig()
            logging.getLogger().setLevel(logging.INFO)
            
        self.bonetype = bonetype
        
        if self.bonetype not in bonetypes_list:
            raise ValueError('Invalid bonetype. Must be {} but got {}'.format(bonetypes_list, self.bonetype))
        if self.bonetype == 'COCO':
            raise ValueError("Unsupported COCO")
        
    
    @property
    def tmpdir(self):
        return os.path.join(self.rootdir, 'temporal_files')
    
    @property
    def isParsed(self):
        return self.videoinfo.isParsed
    
    @property
    def videopath(self):
        if not self.isParsed:
            raise AssertionError('Call parse_video method first!')
        return self.videoinfo.videopath
    @property
    def videoname(self):
        return os.path.basename(self.videopath)
    @property
    def height(self):
        if not self.isParsed:
            raise AssertionError('Call parse_video method first!')
        return self.videoinfo.height
    @property
    def width(self):
        if not self.isParsed:
            raise AssertionError('Call parse_video method first!')
        return self.videoinfo.width
    @property
    def frame_num(self):
        if not self.isParsed:
            raise AssertionError('Call parse_video method first!')
        return self.videoinfo.frame_num
    @property
    def frame_rate(self):
        if not self.isParsed:
            raise AssertionError('Call parse_video method first!')
        return self.videoinfo.frame_rate
    
    @property
    def bone_lines(self):
        if self.bonetype == 'BODY_25':
            return bone_lines_BODY_25
        elif self.bonetype == 'COCO':
            return bone_lines_COCO
    @property
    def joint_num(self):
        if self.bonetype == 'BODY_25':
            return 25
        elif self.bonetype == 'COCO':
            return 18

    def create_bonevideo(self, Data, videoname, size=(600, 400)):
        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        w, h = size
        
        out_videopath = os.path.join(self.rootdir, '2d-video', videoname)
        out = cv2.VideoWriter(out_videopath, fourcc, self.frame_rate, (h, w))

        for frame in range(self.frame_num):
            img = np.zeros((self.height, self.width, 3), np.uint8)

            for joint_ini_index in range(0, len(Data[frame]), 3):
                cv2.circle(img, (int(Data[frame][joint_ini_index]), int(Data[frame][joint_ini_index + 1])), 4, (255, 255, 255), -1)
            for line in self.bone_lines:
                cv2.line(img, (int(Data[frame][line[0]*3]), int(Data[frame][line[0]*3 + 1])),
                         (int(Data[frame][line[1]*3]), int(Data[frame][line[1]*3 + 1])), (255, 255, 255), 2)

            img = cv2.resize(img, (h, w))
            out.write(img)

        out.release()

        # debug
        if self.debug:
            if self.runenv == 'terminal':
                video = cv2.VideoCapture(out_videopath)
                while video.isOpened():
                    cv2.imshow('check', img)
                    k = cv2.waitKey(10)
                    if k == ord('q'):
                        break
                        #exit()
                video.release()

            elif self.runenv == 'jupyter':
                import ipywidgets as wd
                from IPython.display import Video, display
                out = wd.Output(layout={'border': '1px solid black'})
                with out:
                    display(Video(out_videopath, embed=True, mimetype='mp4'))
                

    

    def __select_pitcher_initial_frame(self, people_dicts, autoselect):
        if autoselect:
            # select person is in center
            head_x = []
            head_y = []
            center_x = self.__width / 2
            center_y = self.__height / 2

            for people_dict in people_dicts['people']:
                # get head position
                head_x.append(people_dict['pose_keypoints_2d'][0])
                head_y.append(people_dict['pose_keypoints_2d'][1])

            distance = np.array([pow((head_x[i] - center_x), 2) + pow((head_y[i] - center_y), 2) for i in range(len(head_x))])

            return people_dicts['people'][np.argmin(distance)]['pose_keypoints_2d']
        else:
            # user can select initial person id
            img = np.zeros((self.height, self.width, 3), np.uint8)
            for i,people_dict in enumerate(people_dicts['people']):
                # write person id
                cv2.putText(img, str(i), (int(people_dict['pose_keypoints_2d'][0]) - 5,
                                               int(people_dict['pose_keypoints_2d'][1]) - 5),
                            cv2.FONT_HERSHEY_PLAIN, 15, (255, 255, 255))
                # write joint point
                for joint in range(0, len(people_dict['pose_keypoints_2d']), 3):
                    x = int(people_dict['pose_keypoints_2d'][joint])
                    y = int(people_dict['pose_keypoints_2d'][joint + 1])
                    if x == 0 and y == 0:
                        continue
                    cv2.circle(img, (x, y), 5, (255, 255, 255), -1)

            video = cv2.VideoCapture(self.videopath)
            ret, realimg = video.read()
            video.release()

            img = cv2.resize(img, (int(self.height/2), int(self.width/2)))
            realimg = cv2.resize(realimg, (int(self.height/2), int(self.width/2)))
            
            # terminal
            if self.runenv == 'terminal':
                cv2.imshow("select id", cv2.hconcat([img, realimg]))
                button = cv2.waitKey()
                button = button - 48
                cv2.destroyAllWindows()
            else:
                import ipywidgets as wd
                from IPython.display import display
                from PIL import Image
                show_img = cv2.cvtColor(cv2.hconcat([img, realimg]), cv2.COLOR_BGR2RGB)
                out = wd.Output(layout={'border': '1px solid black'})
                print('Select id')
                with out:
                    display(Image.fromarray(show_img))
                
                button = int(input())

            return people_dicts['people'][button]['pose_keypoints_2d']
    
    
    def __select_pitcher(self, people_dicts, head_x_prev_frame, head_y_prev_frame, previous_num):
            head_x = []
            head_y = []
            for people_dict in people_dicts['people']:
                # get head position
                head_x.append(people_dict['pose_keypoints_2d'][0])
                head_y.append(people_dict['pose_keypoints_2d'][1])

            distance = np.array([pow((head_x[i] - head_x_prev_frame), 2) + pow((head_y[i] - head_y_prev_frame), 2) for i in range(len(head_x))])
            if len(distance) != 0:
                previous_num += 1
                return people_dicts['people'][np.argmin(distance)]['pose_keypoints_2d'], previous_num
            else:
                
                return [0 for i in range(self.joint_num*3)], previous_num
    
    def parse_video(self, videopath):
        self.videoinfo.parse_video(videopath)
        
        
    def run_openpose(self):
        if not self.isParsed:
            raise AssertionError('Call parse_video method first!')
            
        # delete all json files in temporal directory first
        shutil.rmtree(self.tmpdir)
        os.mkdir(self.tmpdir)
        # create .gitkeep
        with open(os.path.join(self.tmpdir, '.gitkeep'), 'w') as gitkeep:
            gitkeep.write("")
        
        # run openpose
        args = ["bash", os.path.join(self.rootdir, "mildetector", "get-from-openpose.sh"), self.videopath, self.tmpdir]
        popen = subprocess.Popen(args, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, universal_newlines=True)
        
        for stdout_line in iter(popen.stdout.readline, ""):
            logging.debug(stdout_line)
            
        popen.stdout.close()
        return_code = popen.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, args)
    
    
    
    # convert json into csv    
    def json2csv(self, jsondata=None, autoselect=True):
        if not self.isParsed:
            raise AssertionError('Call parse_video method first!')
        PEOPLE_DICTS = []
        if jsondata is None:    
            jsonfiles = sorted(glob.glob(os.path.join(self.tmpdir, '*.json')))

            for num, jsonfile in enumerate(jsonfiles):
                with open(jsonfile, 'r') as fr:
                    PEOPLE_DICTS.append(json.load(fr))
        else:
            PEOPLE_DICTS = jsondata
            
        #assert len(jsonfiles) == self.frame_num, "Invalid jsonfile numbers: must be {0}, but got {1}".format(self.frame_num, len(jsonfiles))
        if len(jsonfiles) == 0:
            raise ValueError('No jsonfiles!')
        if len(jsonfiles) != self.frame_num:
            logging.warning("Empty frame was detected. Update frame number from {} to {}".format(self.frame_num, len(jsonfiles)))
            self.videoinfo.frame_num = len(jsonfiles)
        
        
        Data = []
        # set previous frame
        previous_num = 0
        for num, people_dicts in enumerate(PEOPLE_DICTS):
            if num == 0: # first frame
                previous_num = 0
                if len(people_dicts['people']) > 1: # select person
                    data = self.__select_pitcher_initial_frame(people_dicts, autoselect)
                else:
                    data = people_dicts['people'][0]['pose_keypoints_2d']
            # num != 0, i.e, not first frame
            elif len(people_dicts['people']) == 1:
                previous_num = num
                data = people_dicts['people'][0]['pose_keypoints_2d']
            else:
                # print (num)
                data, previous_num = self.__select_pitcher(people_dicts, Data[previous_num - 1][0],
                                                  Data[previous_num - 1][1], previous_num)
            
            Data.append(data)
        
        assert len(Data) == self.frame_num, "Invalid Data number: must be {} but got {}".format(self.frame_num, len(Data))
        self.create_bonevideo(Data, self.videoname)
         
        self.export_csv(Data)
        
        
    def export_csv(self, Data):
        # make csv
        videofilename = self.videoname.split('.')[0]
        csvpath = os.path.join(self.rootdir, '2d-data', '{}.csv'.format(videofilename))
        with open(csvpath, 'w') as fw:
            # header
            fw.write('{},{}\n'.format(self.videopath, ','*75))
            fw.write('width,{0},height,{1},frame_num,{2},fps,{3},{4}\n'.format(self.width, self.height, self.frame_num, self.frame_rate, ','*68))

            row_str = 'time,'
            for joint_ini_index in range(0, len(Data[0]), 3):
                row_str += 'x{0},y{0},c{0},'.format(int(joint_ini_index / 3))
            row_str += '\n'
            fw.write(row_str)

            for frame in range(self.frame_num):
                row_str = '{},'.format(frame)
                for joint_ini_index in range(0, len(Data[frame]), 3):
                    row_str += '{0},{1},{2},'.format(int(Data[frame][joint_ini_index]),
                                                     int(Data[frame][joint_ini_index + 1]),
                                                     float(Data[frame][joint_ini_index + 2]))
                row_str += '\n'
                fw.write(row_str)
    
    ################## export function ###################
    
    def auto_export_from_dir(self, videosdir=None, extension=".MP4"):
        if videosdir is None:
            videosdir = os.path.join(self.rootdir, 'video/')
        videosdir = os.path.join(videosdir, '*{0}'.format(extension))

        videoslist = sorted(glob.glob(videosdir))
        # print(videoslist)

        print("start to get data from openpose....")
        with open(os.path.join(self.rootdir, '2d-data', 'video_dir_info.txt'), 'w') as f:
            f.write(videosdir)

        for num, videopath in enumerate(videoslist):
            sys.stdout.write("\rprocessing... {0}/{1}".format(num, len(videoslist)))
            sys.stdout.flush()

            self.manual_export_from_file(videopath)
            
        print("\nfinish getting data!")
        print("saved csv data in 2d-data directory")
    
    def manual_export_from_file(self, videopath):
        self.parse_video(videopath)
        self.run_openpose()
        self.json2csv(autoselect=False)

    def manual_export_from_files(self, videopaths):
        # store json data and video info to select person manually after running openpose
        PEOPLE_DICT = {}
        VIDEO_DATA = {}
        
        print("using openpose now...\n")
        for num, videopath in enumerate(videopaths):
            sys.stdout.write("\rprocessing... {0}/{1}".format(num, len(videopaths)))
            sys.stdout.flush()
            videoname = videopath.split('/')[-1]
            
            self.parse_video(videopath)
            self.run_openpose()
            VIDEO_DATA[videopath] = self.videoinfo.export_dict


            jsonfiles = sorted(glob.glob(os.path.join(self.tmpdir, '*.json')))
            people_dicts = []
            for frame, jsonfile in enumerate(jsonfiles):
                with open(jsonfile, 'r') as fr:
                    # people_dicts['people'][people num][data][joint index]
                    people_dicts.append(json.load(fr))
            PEOPLE_DICT[videopath] = people_dicts

        print("\nfinished estimating joint coordinetes!")
        print("select person you want to export")

        for videopath in videopaths:
            print("select person you want to export in {0}".format(videoname))
            
            self.videoinfo.parse_from_dict(VIDEO_DATA[videopath])
            self.json2csv(jsondata=PEOPLE_DICT[videopath], autoselect=False)
            
            
                    
class VideoInfo:
    def __init__(self, videopath):
        self.isParsed = False
        
        self.videopath = videopath
        self.height = 0
        self.width = 0
        self.frame_num = 0
        self.frame_rate = 0.0
        
        if videopath:
            self.parse_video(videopath)
    
    @property
    def videoname(self):
        return os.path.basename(self.videopath)
        
    def parse_video(self, videopath):
        # parse video information
        self.videopath = videopath
        video = cv2.VideoCapture(self.videopath)
        self.height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = video.get(cv2.CAP_PROP_FPS)
        video.release()
        
        self.isParsed = True
        
    def parse_from_dict(self, video_dict):
        # parse video information
        self.videopath = video_dict['videoname']
        self.height = video_dict['height']
        self.width = video_dict['width']
        self.frame_num = video_dict['frame_num']
        self.frame_rate = video_dict['frame_rate'] 
        
        self.isParsed = True
    
    def export_dict(self):
        ret = {}
        ret['videoname'] = self.videoname
        ret['height'] = self.height
        ret['width'] = self.width
        ret['frame_num'] = self.frame_num
        ret['frame_rate'] = self.frame_rate
        return ret