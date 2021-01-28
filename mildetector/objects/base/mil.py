import numpy as np
import os

from .openpose import OpenPoseBase

def mkdir(resultSuperDirPath):
    from ...utils import check_and_create_dir

    check_and_create_dir(resultSuperDirPath)
    check_and_create_dir(resultSuperDirPath, 'func_max_image')
    check_and_create_dir(resultSuperDirPath, 'nonzero_image')
    check_and_create_dir(resultSuperDirPath, 'bag_video')
    check_and_create_dir(resultSuperDirPath, 'leave-one-out')
    check_and_create_dir(resultSuperDirPath, 'leave-one-person-out')
    check_and_create_dir(resultSuperDirPath, 'video')



class MILBase(OpenPoseBase):
    def __init__(self, method, experience, dirName, estimatorName, runenv, bonetype='BODY_25', debug=False):
        """
        :param method: method name. ['misvm', 'MISVM', 'MILES']
        :param experience: experience name.
        :param dirName: result directory name.
        :param estimatorName: estimator name to save.
        :param debug: Bool, whether to print log for debug
        """
        super().__init__(bonetype, runenv, debug)
        self.method = method
        self.experience = experience
        self.dirName = dirName

        self.estimatorName = estimatorName

        self.info = {}
        self.positives = {}
        self.negatives = {}

        self.pitches = []

    @property
    def negativeCsvFileName(self):
        return self.info['negativeCsvFileName']
    @negativeCsvFileName.setter
    def negativeCsvFileName(self, filename):
        self.info['negativeCsvFileName'] = filename
    @property
    def negativeCsvFileNames(self):
        return list(self.negatives.keys())

    @property
    def positiveCsvFileName(self):
        return self.info['positiveCsvFileName']
    @positiveCsvFileName.setter
    def positiveCsvFileName(self, filename):
        self.info['positiveCsvFileName'] = filename
    @property
    def positiveCsvFileNames(self):
        return list(self.positives.keys())

    @property
    def labels(self):
        return np.array([pitch.label for pitch in self.pitches])
    @property
    def bags(self):
        """
        return list of ndarray
        """
        return [pitch.bag for pitch in self.pitches]
    @property
    def persons(self):
        return [pitch.person for pitch in self.pitches]
    @property
    def csvFilePaths(self):
        return [pitch.csvpath for pitch in self.pitches]

    @property
    def resultDir(self):
        return os.path.join(self.rootdir, 'result', self.experience, self.dirName)

    def get_label(self, videoname):
        if videoname in self.positiveCsvFileNames:
            return 1

        elif videoname in self.negativeCsvFileNames:
            return -1

        else:
            return 0
