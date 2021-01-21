import cv2

def show_video(runenv, path):
    if runenv == 'terminal':
        video = cv2.VideoCapture(path)
        while video.isOpened():
            ret, img = video.read()
            cv2.imshow('check', img)
            k = cv2.waitKey(10)
            if k == ord('q'):
                break
                #exit()
        video.release()
        cv2.destroyAllWindows()

    elif runenv == 'jupyter':
        import ipywidgets as wd
        from IPython.display import Video, display
        out = wd.Output(layout={'border': '1px solid black'})
        with out:
            display(Video(path, embed=True, mimetype='mp4'))

