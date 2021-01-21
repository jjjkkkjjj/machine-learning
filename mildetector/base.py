import os, logging

class Base:
    def __init__(self, runenv, debug=False):
        self.debug = debug
        if self.debug:
            logging.basicConfig()
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.basicConfig()
            logging.getLogger().setLevel(logging.INFO)

        self.rootdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        self.runenv = runenv


