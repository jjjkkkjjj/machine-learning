

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
            self.data.interpolate('linear', True, False)
            flags = self.data.nanflags()
            self.bag = self.data.bag(None, 'x', 'y', nanflag=flags)

        elif method == 'binary':
            self.data.interpolate('linear', True, False)
            flags = self.data.nanflags()
            binary = self.data.binary()
            self.bag = self.data.bag(None, binary=binary, nanflag=flags)
            
        elif method == 'mirror':
            self.data.interpolate('linear', True, False)
            self.data.mirror(mirrorFor='l')
            flags = self.data.nanflags()
            self.bag = self.data.bag(None, 'x', 'y', nanflag=flags)

        elif method == 'align':
            self.data.interpolate('linear', True, False)
            self.data.mirror(mirrorFor='l')
            flags = self.data.nanflags()
            self.bag = self.data.bag([16, 17], 'x', 'y', nanflag=flags)
            # bag = data.bag([16, 17], 'x', 'y')

        elif method == 'dirvec':
            self.data.interpolate('linear', update=True, save=False)
            self.data.mirror(mirrorFor='l')
            flags = self.data.nanflags(nonnanflag=-1, nanflag=1)
            directionx, directiony, _ = self.data.direction_vector(elim_outlier=False, save=False, filter=False)

            # features = data.interpolate_dir('linear', True, dirx=directionx, diry=directiony, length=length)
            # features['nanflag'] = flags

            # bag = data.bag(features)
            self.bag = self.data.bag([14, 16], dirx=directionx, diry=directiony, nanflags=flags)

        elif method == 'img':
            self.data.mirror(mirrorFor='l')
            imgs = self.data.joint2img(1, save=False)
            
            dicimate = kwargs.pop('dicimate')
            saveMotionTmplate = kwargs.pop('saveMotionTmplate')

            imgs = self.data.motion_history(imgs, dicimate, save=saveMotionTmplate)
            features = self.data.img2featurevector(imgs, norm=False)
            self.bag = self.data.bag(None, norm=False, **features)

        elif method == 'combination':
            self.data.mirror(mirrorFor='l')
            features = self.data.combination(2, self.dicimate, sparse=False)
            #bag = data.bag_sparse(**features)
            self.bag = self.data.bag(None, **features)
        
        else:
            raise ValueError('{0} is invalid method'.format(self.method))