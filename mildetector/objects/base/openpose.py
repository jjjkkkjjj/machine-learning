from .base import Base

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

class OpenPoseBase(Base):
    def __init__(self, bonetype, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bonetype = check_bonetype(bonetype)

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

def check_bonetype(bonetype):
    if bonetype not in bonetypes_list:
        raise ValueError('Invalid bonetype. Must be {} but got {}'.format(bonetypes_list, bonetype))
    if bonetype == 'COCO':
        raise ValueError("Unsupported COCO")
    return bonetype