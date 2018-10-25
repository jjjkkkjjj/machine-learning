import cv2
import sys
import glob


def frame_cutter(videopath, outputdir):
    video = cv2.VideoCapture(videopath)

    frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    videoname = videopath.split('/')[-1]

    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    writer = cv2.VideoWriter(outputdir + '/' + videoname, fourcc, fps, (width, height))

    sys.stdout.write('\rloading... : 0 %')
    sys.stdout.flush()
    imgs = []
    for i in range(frame_num):
        ret, img = video.read()
        imgs.append(img)
        sys.stdout.write('\rloading... : {0} %'.format(int(i*100/frame_num)))
        sys.stdout.flush()
    print('\nloaded')

    now = 0
    start = 0
    finish = frame_num - 1
    while True:
        cv2.imshow(videoname, imgs[now])

        key = cv2.waitKey()
        if key == ord('n') and frame_num != now + 1:
            now += 1
        elif key == ord('p') and now != 0:
            now += -1
        elif key == ord('t'):
            print('put frame number')
            inp = input()
            if str(inp).isdigit():
                inp = int(inp)
                if inp >= 0 and inp < frame_num:
                    now = inp
                else:
                    print('invalid frame number')
        elif key == ord('i'):
            print('set {0} as initial frame'.format(now))
            start = now
        elif key == ord('l'):
            print('set {0} as last frame'.format(now))
            finish = now
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

    if start == 0 and finish == frame_num - 1:
        print("Warning: your settef frame is same to original video")
        print("skipped this process")
    else:
        sys.stdout.write('\rsaving to {0}... : 0 %'.format(outputdir + "/" + videoname))
        sys.stdout.flush()
        for i, frame in enumerate(range(start, finish)):
            writer.write(imgs[frame])
            sys.stdout.write(
                '\rsaving to {0}... : {1} %'.format(outputdir + "/" + videoname, int(i * 100 / (finish - start))))
            sys.stdout.flush()
        print('\nsaved')
    writer.release()
    video.release()

    return

def all():
    outdir = "/home/junkado/Desktop/keio/hard/cut-video/"
    #with open('hard-video.csv', 'r') as f:
    #    hard_csvfiles_ = f.read().split('\n')[:-1]
    #with open('easy-video.csv', 'r') as f:
    #    easy_csvfiles_ = f.read().split('\n')[:-1]

    #hard_csvfiles = ['C' + file.split(',')[0] + '.MP4' for file in hard_csvfiles_]
    #easy_csvfiles = ['C' + file.split(',')[0] + '.MP4' for file in easy_csvfiles_]

    videofiles = sorted(glob.glob("/home/junkado/Desktop/keio/hard/keio-pitchingvideo/*.MP4"))
    #for hard_csvfile in hard_csvfiles:
    #    frame_cutter("/home/junkado/Desktop/keio/hard/keio-pitchingvideo/" + hard_csvfile, outdir)
    #for easy_csvfile in easy_csvfiles:
    #    frame_cutter("/home/junkado/Desktop/keio/hard/keio-pitchingvideo/" + easy_csvfile, outdir)
    for videofile in videofiles:
        frame_cutter(videofile, outdir)

    return

if __name__ == '__main__':
    #all()
    #exit()
    #args = sys.argv
    #if len(args) == 3:
    #    frame_cutter(args[1], args[2])
    frame_cutter("/home/junkado/Desktop/keio/hard/keio-pitchingvideo/C1105.MP4", "/home/junkado/Desktop/keio/hard/cut-video/")