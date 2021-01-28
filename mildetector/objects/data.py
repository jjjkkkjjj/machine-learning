from mildetector.utils import show_video
import numpy as np
from scipy.interpolate import CubicSpline as cs
from scipy.interpolate import interp1d
import sys
import cv2
from scipy import fftpack
import scipy.sparse as sp
import itertools
import os, time
from .base import OpenPoseBase
from ..utils import check_and_create_dir

def checkExistDir():
    check_and_create_dir('bag', 'joint2img', 'motempl')
    check_and_create_dir('bag', 'norm')


class Data(OpenPoseBase):
    def __init__(self, videopath, width, height, frame_num, fps, time_rows, hand=None, dicimate=None, bonetype='BODY_25', runenv='terminal', debug=False):
        super().__init__(bonetype, runenv, debug)
        self.videopath = videopath
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.frame_num = int(frame_num)
        self.hand = hand


        self.joint_name = [time_rows[0][i] for i in range(1, len(time_rows[0])) if time_rows[0][i] != '']
        self.x = []
        self.y = []
        self.c = []
        self.nan = []

        for index in range(0, len(self.joint_name), 3):
            self.x.append([time_rows[time + 1][index + 1] for time in range(len(time_rows) - 1) ])
            self.y.append([time_rows[time + 1][index + 1 + 1] for time in range(len(time_rows) - 1)])
            self.c.append([time_rows[time + 1][index + 1 + 2] for time in range(len(time_rows) - 1)])

        # [joint][time]
        self.x = np.array(self.x).astype(np.int)
        self.x = np.where(self.x != 0, self.x, np.nan)
        self.y = np.array(self.y).astype(np.int)
        self.y = np.where(self.y != 0, self.y, np.nan)
        self.c = np.array(self.c).astype(np.float)

        if dicimate is not None:
            self.x = self.x[:, ::dicimate]
            self.y = self.y[:, ::dicimate]
            self.c = self.c[:, ::dicimate]
            self.frame_num = int(self.x.shape[1])
        self.dicimate = dicimate

        for x in self.x:
            self.nan.append(np.where(np.isnan(x))[0])

        checkExistDir()


    def norm(self, base_time=0, method='default', save=False, mean_for_alignment=None):
        if method == '':
            print("not define")
            exit()
        else:
            if mean_for_alignment is not None:
                diff = (self.x[mean_for_alignment['joint'], mean_for_alignment['frame']] - mean_for_alignment['x'],
                        self.y[mean_for_alignment['joint'], mean_for_alignment['frame']] - mean_for_alignment['y'])

                self.x -= diff[0]
                self.y -= diff[1]
                #print(self.x[mean_for_alignment['joint'], mean_for_alignment['frame']],
                #       self.y[mean_for_alignment['joint'], mean_for_alignment['frame']])

            # 0 head, 10: right uncle, 13: left uncle
            headx = self.x[0][base_time]
            heady = self.y[0][base_time]
            unclex = self.x[10][base_time]
            uncley = self.y[10][base_time]
            if np.isnan(uncley):
                uncley = self.y[13][base_time]
            baseheight = (uncley - heady)*1.3
            basewidth = baseheight*0.3

            # position vector to new origin
            try:
                o_vec = (int(headx - basewidth*0.5), int(heady - (uncley - heady)*0.3*0.5))
            except ValueError:
                return False
            self.x = self.x.astype('float')
            self.y = self.y.astype('float')

            self.x = self.x - o_vec[0]
            self.x /= basewidth
            self.y = self.y - o_vec[1]
            self.y /= baseheight

            # range will be changed into 0 - 1
            min = np.nanmin(self.x)
            self.x -= min
            max = np.nanmax(self.x)
            self.x /= max

            min = np.nanmin(self.y)
            self.y -= min
            max = np.nanmax(self.y)
            self.y /= max

            if save:
                self.save(path='bag/norm/{0}'.format(self.videopath.split('/')[-1]), plot=(self.x, self.y))
            return True

    def __dellist(self, items, indexes):
        if len(indexes) == 0:
            return items, [i for i in range(len(items))]
        else:
            return [item for index, item in enumerate(items) if index not in indexes]

    # return times of real number
    # if update is true, then there is no return value
    def nonantimes(self, update=False):
        nan_joint_index, nan_time = np.where(np.isnan(self.x))

        times = np.arange(self.frame_num, dtype='int')
        for joint_index in range(len(self.joint_name) / 3):
            time_indexes = np.where(nan_joint_index == joint_index)[0]
            del_time = [nan_time[i] for i in time_indexes]

            times = np.setdiff1d(times, np.array(del_time)).astype('int')

        if update:
            self.x = self.x[:, times]
            self.y = self.y[:, times]
            self.c = self.c[:, times]
            self.frame_num = times.shape[0]
        else:
            return times

    # return feature vectors
    # if update is true, then there is no return value
    def interpolate(self, method='spline', update=True, save=False):
        newx = []
        newy = []
        newc = []

        # eliminate initial and last nan for interpolation
        # print(self.frame_num)
        # self.__adjust_time()
        # print(self.frame_num)
        # interpolation indivisually
        if method == 'spline':
            for x, y, c in zip(self.x, self.y, self.c):
                time = np.where(~np.isnan(x))[0]
                if time.shape[0] == 0:
                    newx.append(np.zeros(x.shape))
                    newy.append(np.zeros(y.shape))
                    newc.append(np.zeros(c.shape))
                    continue
                elif time.shape[0] == 1:
                    _newx, _newy, _newc = np.zeros(x.shape), np.zeros(y.shape), np.zeros(c.shape)
                    _newx[time] = x[time]
                    _newy[time] = y[time]
                    _newc[time] = c[time]
                    newx.append(_newx)
                    newy.append(_newy)
                    newc.append(_newc)
                    continue

                spline_x = cs(time, x[time])
                spline_y = cs(time, y[time])
                spline_c = cs(time, c[time])

                time = [i for i in range(len(x))]

                tmp = spline_x(time)
                newx.append(np.where(tmp < 0, 0, np.where(tmp > 1, 1, tmp)))
                tmp = spline_y(time)
                newy.append(np.where(tmp < 0, 0, np.where(tmp > 1, 1, tmp)))
                tmp = spline_c(time)
                newc.append(np.where(tmp < 0, 0, np.where(tmp > 1, 1, tmp)))

        elif method == 'linear':
            for x, y, c in zip(self.x, self.y, self.c):
                time = np.where(~np.isnan(x))[0]
                if time.shape[0] == 0:
                    newx.append(np.zeros(x.shape))
                    newy.append(np.zeros(y.shape))
                    newc.append(np.zeros(c.shape))
                    continue
                elif time.shape[0] == 1:
                    _newx, _newy, _newc = np.zeros(x.shape), np.zeros(y.shape), np.zeros(c.shape)
                    _newx[time] = x[time]
                    _newy[time] = y[time]
                    _newc[time] = c[time]
                    newx.append(_newx)
                    newy.append(_newy)
                    newc.append(_newc)
                    continue

                interp1d_x = interp1d(time, x[time], fill_value='extrapolate')
                interp1d_y = interp1d(time, y[time], fill_value='extrapolate')
                interp1d_c = interp1d(time, c[time], fill_value='extrapolate')

                time = [i for i in range(len(x))]

                tmp = interp1d_x(time)
                newx.append(np.where(tmp < 0, 0, np.where(tmp > 1, 1, tmp)))
                tmp = interp1d_y(time)
                newy.append(np.where(tmp < 0, 0, np.where(tmp > 1, 1, tmp)))
                tmp = interp1d_c(time)
                newc.append(np.where(tmp < 0, 0, np.where(tmp > 1, 1, tmp)))

        else:
            print("warning: {0} is not defined as interpolation method".format(method))

        newx = np.array(newx)
        newy = np.array(newy)
        newc = np.array(newc)

        if save:
            self.save(path='bag/interpolate-{0}/{1}'.format(method, self.videopath.split('/')[-1]), plot=(newx, newy))

        if update:
            self.x = newx
            self.y = newy
            self.c = newc
        else:
            return newx, newy, newc

    def interpolate_dir(self, method='spline', save=False, **new_feature):
        new = {}
        # eliminate initial and last nan for interpolation
        #print(self.frame_num)
        #self.__adjust_time()
        #print(self.frame_num)
        # interpolation indivisually
        if method == 'spline':
            for key, feature in new_feature.items():
                if self.frame_num == np.array(feature).shape[0]:
                    new[key] = []
                    for var in feature:
                        time = np.where(~np.isnan(var))[0]
                        if time.shape[0] == 0:
                            new[key].append(np.zeros(var.shape))
                            continue

                        spline = cs(time, var[time])

                        time = [i for i in range(len(var))]

                        tmp = spline(time)
                        new[key].append(np.where(tmp < 0, 0, np.where(tmp > 1, 1, tmp)))
                else:
                    raise TypeError('{0}\'s shape which is {1} must be ({2},...)'.format(key, np.array(feature).shape,
                                                                                         self.frame_num))

        elif method == 'linear':
            for key, feature in new_feature.items():
                if self.frame_num == np.array(feature).shape[0]:
                    new[key] = []
                    for var in feature:
                        time = np.where(~np.isnan(var))[0]
                        if time.shape[0] == 0:
                            new[key].append(np.zeros(var.shape))
                            continue

                        interp1d_ = interp1d(time, var[time], fill_value="extrapolate")

                        time = [i for i in range(len(var))]

                        tmp = interp1d_(time)
                        if key == 'length':
                            new[key].append(np.where(tmp <= 0, 0.001, np.where(tmp > 1, 1, tmp)))
                        else:
                            new[key].append(np.where(tmp < -1, -1, np.where(tmp > 1, 1, tmp)))
                else:
                    raise TypeError('{0}\'s shape which is {1} must be ({2},...)'.format(key, np.array(feature).shape, self.frame_num))

        else:
            print("warning: {0} is not defined as interpolation method".format(method))

        for key in new.keys():
            new[key] = np.array(new[key])
        try:
            l = np.sqrt(new['dirx'] * new['dirx'] + new['diry'] * new['diry'])
            new['dirx'] /= l
            new['diry'] /= l
        except KeyError:
            pass
        #for var in args:
        #    exec 'self.{0} = new[{0}]'.format(var) in {}

        if save:
            x = [[] for i in range(18)]
            y = [[] for i in range(18)]

            # [between joints][time]
            newx = new['dirx'].T
            newy = new['diry'].T
            length = new['length'].T
            # here is restoring angle, length into x,y
            #Lines = [[1, 0],[1, 2],[2, 3],[3, 4],[1, 5],[5, 6],[6, 7],[1, 8],[8, 9],[9, 10],[1, 11],[11, 12],
            #[12, 13], [0, 14],[14, 16],[0, 15],[15, 17]]
            x[1] = np.zeros(self.frame_num)
            y[1] = np.zeros(self.frame_num)

            for joint_index, bet_index in zip([0, 2, 5, 8, 11], [0, 1, 4, 7, 10]):
                x[joint_index] = newx[bet_index] * length[bet_index]
                y[joint_index] = newy[bet_index] * length[bet_index]

            jointlists = [[1, 2, 3, 4],
                          [1, 5, 6, 7],
                          [1, 8, 9, 10],
                          [1, 11, 12, 13],
                          [0, 14, 16],
                          [0, 15, 17]]
            betlists = [[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                        [10, 11, 12],
                        [13, 14],
                        [15, 16]]
            for jointlist, betlist in zip(jointlists, betlists):
                for index, bet_index in zip(range(1, len(jointlist)), betlist):
                    x[jointlist[index]] = newx[bet_index] * length[bet_index] + x[jointlist[index - 1]]
                    y[jointlist[index]] = newy[bet_index] * length[bet_index] + y[jointlist[index - 1]]

            self.save(path='bag/interpolate-{0}/{1}'.format(method, self.videopath.split('/')[-1]),
                    plot=(np.array(x), np.array(y)))


        return new
        """
        if update:
            self.x = newx
            self.y = newy
            self.c = newc
        else:
            return newx, newy, newc
        """

    # return bag [time][feature]
    # feature's shape is (time, )
    def bag(self, delete_index, *args, norm=False, **new_feature):
        feature_vectors = {}

        for var in args:
            feature_vectors[var] = eval('self.{0}.T'.format(var))
            if delete_index is not None:
                feature_vectors[var] = np.delete(feature_vectors[var], delete_index, axis=1)

            # if error is occurred, raise attributeerror
        for key, feature in new_feature.items():
            if self.frame_num == np.array(feature).shape[0]:
                feature_vectors[key] = feature

                if delete_index is not None:
                    feature_vectors[key] = np.delete(feature_vectors[key], delete_index, axis=1)

            else:
                if args is None:
                    raise TypeError('{0}\'s shape which is {1} must be ({2},...)'.format(key, np.array(feature).shape,
                                                                                         self.frame_num))
                else:
                    feature_vectors[key] = feature


        Bag = np.hstack(tuple([feature for feature in feature_vectors.values()]))
        if norm:
            Bag = Bag / np.max(np.linalg.norm(Bag, axis=1))
        """
        exit()
        return
        for time in range(self.frame_num):
            for key, feature in feature_vectors.items():
                a = 0
                
        
        Bag = np.array([[self.x[int(i / 2), time] if i % 2 == 0 else self.y[int(i / 2), time]
                         for i in range(self.x.shape[0] * 2)]
                        for time in range(self.x.shape[1])])
                        """

        return Bag

    def bag_sparse(self, **new_feature):
        feature_vectors = {}

        for key, feature in new_feature.items():
            if isinstance(feature, sp.coo_matrix):
                feature_vectors[key] = feature
            else:
                raise TypeError('feature type must be sparse matrix(coo)')

        Bag = sp.hstack(tuple([feature for feature in feature_vectors.values()]))

        return Bag

    def binary(self, size=(50, 100), sigma=15.0):
        binary_features = [] #[joint][time]


        X = np.tile(np.arange(size[0]), (self.frame_num, 1))#[time][position=50]
        Y = np.tile(np.arange(size[1]), (self.frame_num, 1))#[time][position=100]

        for x, y in zip(self.x, self.y):
            x = (x*size[0]).astype('int')
            y = (y*size[1]).astype('int')

            x_part = np.power(X - x[:, np.newaxis], 2)
            y_part = np.power(Y - y[:, np.newaxis], 2)
            charac = x_part[:, :, np.newaxis] + y_part[:, np.newaxis, :]

            binary_features.append(np.exp(-charac/ (2*np.power(sigma, 2))))

        # [time][joint][positionx=50][positiony=100]
        return np.array(binary_features).transpose((1, 0, 2, 3))

    # return flags list expressed whether it is nan or not
    def nanflags(self, nonnanflag=0, nanflag=1):
        flags = [] # [joint][time]
        for x, nan in zip(self.x, self.nan):
            flag = np.ones(x.shape[0]) * nonnanflag
            flag[nan] = nanflag
            flags.append(flag)
        # [time][joint]
        return np.array(flags).T

    def __adjust_time(self, nan=None):

        initial_complete_time = []
        finish_complete_time = []

        for x in self.x:
            if nan is None:
                time = np.where(~np.isnan(x))[0]
            else:
                time = np.where(x == nan)[0]

            try:
                initial_complete_time.append(time[0])
                finish_complete_time.append(time[-1])
            except IndexError:
                raise IndexError("invalid joint")

        init_time = np.max(np.array(initial_complete_time))
        fin_time = np.min(np.array(finish_complete_time))

        self.frame_num = fin_time - init_time

        #transpose = lambda matrix: map(list, zip(*matrix))
        #self.x = transpose(transpose(self.x)[init_time:])
        self.x = list(np.array(self.x).T[init_time:fin_time].T)
        self.y = list(np.array(self.y).T[init_time:fin_time].T)
        self.c = list(np.array(self.c).T[init_time:fin_time].T)

    def mirror(self, mirrorFor='l'):
        if self.hand == mirrorFor:
            xrange = (np.nanmin(self.x), np.nanmax(self.x))
            yrange = (np.nanmin(self.y), np.nanmax(self.y))

            center = ((xrange[1]-xrange[0])/2, (yrange[1]-yrange[0])/2) # basically it is 0.5, 0.5

            self.x = self.x + 2*(center[0] - self.x)

            # check
            """
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'.decode('utf-8')
            fig = plt.figure()

            def __update(time):
                if time != 0:
                    plt.cla()

                plt.ylim([np.nanmax(self.y), np.nanmin(self.y)])
                plt.xlim([np.nanmin(self.x), np.nanmax(self.x)])
                plt.scatter(self.x[:, time], self.y[:, time])
                for line in Lines:
                    plt.plot([self.x[line[0], time], self.x[line[1], time]],
                             [self.y[line[0], time], self.y[line[1], time]])
                    plt.title('frame {0}'.format(time))

            ani = FuncAnimation(fig, __update, frames=self.frame_num, interval=10)
            plt.show()
            """
        else:
            return

    def direction_vector(self, elim_outlier=False, save=False, filter=False):
        # direction: line[0] -> line[1]
        directionx = []
        directiony = []
        length = []
        # range?
        for line in self.bone_lines:
            x = (self.x[line[1]] - self.x[line[0]]) # [time]
            y = (self.y[line[1]] - self.y[line[0]]) # [time]
            l = np.sqrt(x*x + y*y)
            l[l == 0] = 1
            # eliminate outlier
            # i don't know if this process is needed
            if elim_outlier:
                p75, p25 = np.nanpercentile(l, [75, 25])
                IQR = (p75 - p25)
                bo = np.logical_or.reduce((l > p75 + IQR * 1.5, l < p25 - IQR * 1.5))

                x[bo] = np.nan
                y[bo] = np.nan
                l[bo] = np.nan

            if filter:
                num = 5
                b = np.ones(num) / num

                x = np.convolve(x, b, mode='same')
                y = np.convolve(y, b, mode='same')
                angle = np.arctan2(y, x)
                import matplotlib.pyplot as plt
                fig = plt.figure()
                #plt.plot(np.arange(self.frame_num), angle, '-')
                plt.plot(np.arange(self.frame_num), x, '-')
                plt.plot(np.arange(self.frame_num), y, '-')
                plt.legend()
                plt.show()


            directionx.append(x/l)
            directiony.append(y/l)
            length.append(l)
            #length.append(l/np.nanmax(l))

            #print np.sum(np.isnan(l))

        directionx = np.array(directionx)
        directiony = np.array(directiony)
        length = np.array(length)

        if save:
            x = [[] for i in range(18)]
            y = [[] for i in range(18)]
            # here is restoring angle, length into x,y
            #Lines = [[1, 0],[1, 2],[2, 3],[3, 4],[1, 5],[5, 6],[6, 7],[1, 8],[8, 9],[9, 10],[1, 11],[11, 12],
            #[12, 13], [0, 14],[14, 16],[0, 15],[15, 17]]
            x[1] = np.zeros(self.frame_num)
            y[1] = np.zeros(self.frame_num)

            for joint_index, bet_index in zip([0, 2, 5, 8, 11], [0, 1, 4, 7, 10]):
                x[joint_index] = directionx[bet_index] * length[bet_index]
                y[joint_index] = directiony[bet_index] * length[bet_index]

            jointlists = [[1, 2, 3, 4],
                          [1, 5, 6, 7],
                          [1, 8, 9, 10],
                          [1, 11, 12, 13],
                          [0, 14, 16],
                          [0, 15, 17]]
            betlists = [[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                        [10, 11, 12],
                        [13, 14],
                        [15, 16]]
            for jointlist, betlist in zip(jointlists, betlists):
                for index, bet_index in zip(range(1, len(jointlist)), betlist):
                    x[jointlist[index]] = directionx[bet_index] * length[bet_index] + x[jointlist[index - 1]]
                    y[jointlist[index]] = directiony[bet_index] * length[bet_index] + y[jointlist[index - 1]]

            self.save(path='bag/dirvec/{0}'.format(self.videopath.split('/')[-1]),
                    plot=(np.array(x), np.array(y)))

        # [time][vector num]
        return np.array(directionx).T, np.array(directiony).T, np.array(length).T

    def save(self, path, plot):

        sys.stdout.write('\rsaving to {0}...'.format(path))
        sys.stdout.flush()
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'.decode('utf-8')
        fig = plt.figure()

        x = plot[0]
        y = plot[1]

        def __update(time):
            if time != 0:
                plt.cla()

            plt.ylim([np.nanmax(y), np.nanmin(y)])
            plt.xlim([np.nanmin(x), np.nanmax(x)])
            plt.scatter(x[:, time], y[:, time])
            for line in self.bone_lines:
                plt.plot([x[line[0], time], x[line[1], time]],
                         [y[line[0], time], y[line[1], time]])
                plt.title('frame {0}'.format(time))

        ani = FuncAnimation(fig, __update, frames=self.frame_num, interval=10)
        plt.show()
        #ani.save(path, writer='ffmpeg')
        sys.stdout.write('\rsaved to {0}\n'.format(path))
        sys.stdout.flush()

    def filter(self, y, cutoff=30):
        from scipy import fftpack
        # fft
        yfft = fftpack.fft()

    def joint2img(self, width=64, height=64, save=False):
        # convert x,y coordinates into pixcel(width x height)
        # [joint][time]
        x = self.x * width
        y = self.y * height

        # [time][joint]
        x = x.T.tolist()
        y = y.T.tolist()
        imgs = []
        for x_, y_ in zip(x, y):
            img = np.zeros((height, width), dtype=np.float32)
            #for x__, y__ in zip(x_, y_):
            #    if not np.isnan(x__):
            #        cv2.circle(img, (int(x__), int(y__)), 1, (255,255,255), -1)
            for line in self.bone_lines:
                if np.isnan(x_[line[0]]) or np.isnan(x_[line[1]]):
                    continue
                else:
                    cv2.line(img, (int(x_[line[0]]), int(y_[line[0]])),
                             (int(x_[line[1]]), int(y_[line[1]])), (255,255,255), 3)
            img = cv2.GaussianBlur(img, (3, 3), 2)
            imgs.append(np.uint8(img))
            #cv2.imshow("a", img)
            #k = cv2.waitKey(30)
            #if k == ord('q'):
            #    break

        if save:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            savepath = os.path.join(self.rootdir, 'bag', 'joint2img', os.path.basename(self.videopath))
            writer = cv2.VideoWriter(savepath, fourcc, 30.0, (height, width))
            for img in imgs:
                writer.write(cv2.cvtColor(np.uint8(img), cv2.COLOR_GRAY2RGB))

            writer.release()
            if self.debug:
                show_video(self.runenv, savepath)

        return imgs

    def motion_history(self, imgs, duration=0.1, save=False):
        img_pre = imgs[0]

        history = np.zeros(img_pre.shape, dtype=np.float32)

        newimgs = []
        videopath = os.path.join(self.rootdir, 'bag', 'joint2img', 'motempl', os.path.basename(self.videopath))
        if save:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(videopath, fourcc, 30.0, img_pre.shape)
            for index, img in enumerate(imgs[1:]):
                color_diff = cv2.absdiff(img, img_pre)

                timestamp = time.clock()
                # note that this function set history as;
                # if color_diff(i, j) != 0:
                #   history(i,j) = ("now")timestamp
                # else:
                #   history(i,j) = 0 (history(i,j) < ("now")timestamp - duration)
                #                = history(i,j) (else)
                cv2.motempl.updateMotionHistory(color_diff, history, timestamp, duration)

                hist = np.array(np.clip((history - (timestamp - duration)) / duration, 0, 1) * 255, np.uint8)
                #hist = cv2.cvtColor(hist, cv2.COLOR_GRAY2BGR)
                newimgs.append(hist)
                img_pre = img.copy()

                writer.write(cv2.cvtColor(hist, cv2.COLOR_GRAY2BGR))

            writer.release()
            if self.debug:
                show_video(self.runenv, videopath)

        else:
            reader = cv2.VideoCapture(videopath)
            frame_num = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

            for i in range(0, frame_num):
                ret, img = reader.read()
                newimgs.append(cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY))
            reader.release()

        self.frame_num = len(newimgs)
        return newimgs

    def img2featurevector(self, imgs, dwidth=16, dheight=16, norm=False):
        if not isinstance(dwidth, int) or not isinstance(dheight, int):
            raise TypeError('dwidth and dheight must be int')

        (height, width) = imgs[0].shape
        if height % dheight != 0 or width % dwidth != 0:
            raise ValueError('dwidth and dheight must be divided by height and width respectedly')

        features_ = []
        wnum = int(width/dwidth)
        hnum = int(height/dheight)

        for img in imgs:
            features_.append(img.reshape((hnum*wnum, dheight*dwidth)).astype(np.float32)/255.)

        # [split][time][dim]
        features_ = np.array(features_).transpose((1,0,2))


        #print(np.max(features_), np.min(features_))
        features_ = np.hstack((feature for feature in features_))
        if norm:
            n = np.linalg.norm(features_, axis=1)[:, np.newaxis]
            n[n == 0] = 1
            features_ =  features_ / n
        features = {"img": features_}
        """
        features = {}
        for split_index, feature in enumerate(features_):
            features[str(int(split_index/dheight)) + str(int(split_index%dwidth))] = feature
        """

        return features

    def combination(self, selectedJoinyNum, sparse=False):
        jointIds = [i for i in range(int(len(self.joint_name)/3))]
        jointComb = list(itertools.combinations(jointIds, selectedJoinyNum))

        features = {}

        # zeros[time][joint(*2)]
        zeros = np.zeros((self.x.shape[1], self.x.shape[0]*2)) # x.shape=[joint][time]
        x, y = zeros, zeros.copy()
        x[:, np.arange(0, len(jointIds)*2, 2)] = self.x.T # even columns
        y[:, np.arange(1, len(jointIds)*2, 2)] = self.y.T # odd columns
        data = x + y

        # instance order is (comb1(t=1),...,comb1(t=T),comb2(t=1),...)
        timeIndices = [i for i in range(self.frame_num)]
        if timeIndices[-1] != self.frame_num - 1:
            timeIndices.append(self.frame_num - 1)
        timeIndices = np.array(timeIndices)
        data = data[timeIndices]

        data[np.isnan(data)] = 0
        if sparse:
            jointComb = np.array(jointComb)
            data_ = np.array([data[:, comb] for comb in np.array(jointComb)]).flatten()
            row_ = np.repeat(np.arange(timeIndices.size*len(jointComb)), selectedJoinyNum, axis=0).flatten()
            col_ = np.repeat(jointComb, timeIndices.size, axis=0).flatten()
            feature = sp.coo_matrix((data_, (row_, col_)), shape=(timeIndices.size*len(jointComb), 18*2))
        else:
            feature = np.zeros((timeIndices.size * len(jointComb), 18 * 2))
            jointComb = np.array(jointComb)
            for n, comb in enumerate(jointComb):
                feature[n * timeIndices.size:(n + 1) * timeIndices.size, comb] = data[:, comb]
            #feature[np.isnan(feature)] = 0.0
        features['combination'] = feature

        return features