import time

from .data import Data


class Pitch:
    def __init__(self, label, person, csvpath, data: Data):
        self.label = label
        self.csvpath = csvpath
        self.person = person
        self.data = data

        self.bag = None

    def preprocess(self, method, mean=None, **kwargs):
        if not self.data.norm(save=False, mean_for_alignment=mean):
            self.label = 0
            return

        if method == 'real':
            self.bag = bag_real(self.data, **kwargs)

        elif method == 'binary':
            self.bag = bag_binary(self.data, **kwargs)

        elif method == 'mirror':
            self.bag = bag_mirror(self.data, **kwargs)

        elif method == 'align':
            self.bag = bag_align(self.data, **kwargs)

        elif method == 'dirvec':
            self.bag = bag_dirvec(self.data, **kwargs)

        elif method == 'img':
            self.bag = bag_img(self.data, **kwargs)

        elif method == 'combination':
            self.bag = bag_combinatation(self.data, **kwargs)
        
        else:
            raise ValueError('{0} is invalid method'.format(self.method))

def bag_real(data, interpkwargs={'method': 'linear', 'update': True, 'save': False}):
    data.interpolate(**interpkwargs)
    flags = data.nanflags()
    return data.bag(None, 'x', 'y', nanflag=flags)

def bag_binary(data, interpkwargs={'method': 'linear', 'update': True, 'save': False},
                binarykwrags={'size': (50, 100), 'sigma': 15.0}):
    data.interpolate(**interpkwargs)
    flags = data.nanflags()
    binary = data.binary(**binarykwrags)
    return data.bag(None, binary=binary, nanflag=flags)

def bag_mirror(data, interpkwargs={'method': 'linear', 'update': True, 'save': False}, mirrorFor='l'):
    data.interpolate(**interpkwargs)
    data.mirror(mirrorFor=mirrorFor)
    flags = data.nanflags()
    return data.bag(None, 'x', 'y', nanflag=flags)

def bag_align(data, interpkwargs={'method': 'linear', 'update': True, 'save': False}, mirrorFor='l'):
    data.interpolate(**interpkwargs)
    data.mirror(mirrorFor=mirrorFor)
    flags = data.nanflags()
    return data.bag([16, 17], 'x', 'y', nanflag=flags)
    # bag = data.bag([16, 17], 'x', 'y')

def bag_dirvec(data, interpkwargs={'method': 'linear', 'update': True, 'save': False}, mirrorFor='l',
                nanflagkwargs={'nonnanflag': -1, 'nanflag': 1}, dirveckwargs={'elim_outlier': False, 'save': False, 'filter': False}):
    data.interpolate(**interpkwargs)
    data.mirror(mirrorFor=mirrorFor)
    flags = data.nanflags(**nanflagkwargs)
    directionx, directiony, _ = data.direction_vector(**dirveckwargs)

    # features = data.interpolate_dir('linear', True, dirx=directionx, diry=directiony, length=length)
    # features['nanflag'] = flags

    # bag = data.bag(features)
    return data.bag([14, 16], dirx=directionx, diry=directiony, nanflags=flags)

def bag_img(data, mirrorFor='l', joint2imgkwargs={'width': 64, 'height': 64, 'save': False}, mhkwargs={'duration': 0.1, 'save': True},
            img2featurekwargs={'dwidth': 16, 'dheight': 16, 'norm': False}):
    data.mirror(mirrorFor='l')
    imgs = data.joint2img(**joint2imgkwargs)

    imgs = data.motion_history(imgs, **mhkwargs)
    features = data.img2featurevector(imgs, **img2featurekwargs)
    return data.bag(None, norm=False, **features)

def bag_combinatation(data, mirrorFor='l', combinataionkwargs={'selectedJoinyNum': 2, 'sparse': False}):
    data.mirror(mirrorFor=mirrorFor)
    features = data.combination(**combinataionkwargs)
    #bag = data.bag_sparse(**features)
    return data.bag(None, **features)