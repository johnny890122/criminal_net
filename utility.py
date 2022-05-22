from gensim.models.callbacks import CallbackAny2Vec

def show_arg(args):
    print("="*30)
    print("Show argument...")
    for key, value in vars(args).items():
        print("{:20}: {}".format(key, value))

    print("="*30)

class EpochLogger(CallbackAny2Vec):

    '''Callback to log information about training'''


    def __init__(self):

        self.epoch = 0


    def on_epoch_begin(self, model):
        pass

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1
