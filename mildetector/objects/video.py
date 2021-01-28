import cv2

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
