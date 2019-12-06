import time
import signal
import os
import sys
import torch


'''
Usage:

    init_checkpoint()

    if exist_checkpoint():
        any_object = load_checkpoint()

    save_checkpoint(any_object)
'''

checkpoint_freq = 1000
MAIN_PID = os.getpid()
HALT_filename = 'HALT'
CHECKPOINT_filename = 'checkpoint.pth.tar'
CHECKPOINT_tempfile = 'checkpoint.temp'
SIGNAL_RECEIVED = False

def SIGTERMHandler(a, b):
    print('received sigterm')
    pass


def signalHandler(a, b):
    global SIGNAL_RECEIVED
    print('Signal received', a, time.time(), flush=True)
    SIGNAL_RECEIVED = True

    ''' If HALT file exists, which means the job is done, exit peacefully.
    '''
    if os.path.isfile(HALT_filename):
        print('Job is done, exiting')
        exit(0)

    return

def init_checkpoint():
    signal.signal(signal.SIGUSR1, signalHandler)
    signal.signal(signal.SIGTERM, SIGTERMHandler)
    print('Signal handler installed', flush=True)

def save_checkpoint(state):
    global CHECKPOINT_filename, CHECKPOINT_tempfile
    torch.save(state, CHECKPOINT_tempfile)
    if os.path.isfile(CHECKPOINT_tempfile):
        os.rename(CHECKPOINT_tempfile, CHECKPOINT_filename)
    print("Checkpoint done")

def save_checkpoint_if_signal(state):
    global SIGNAL_RECEIVED
    if SIGNAL_RECEIVED:
        save_checkpoint(state)

def exist_checkpoint():
    global CHECKPOINT_filename
    return os.path.isfile(CHECKPOINT_filename)

def load_checkpoint():
    global CHECKPOINT_filename
    # optionally resume from a checkpoint
    # if args.resume:
        #if os.path.isfile(args.resume):
    # To make the script simple to understand, we do resume whenever there is
    # a checkpoint file
    if os.path.isfile(CHECKPOINT_filename):
        print(f"=> loading checkpoint {CHECKPOINT_filename}")
        checkpoint = torch.load(CHECKPOINT_filename)
        print("=> loaded checkpoint {CHECKPOINT_filename}")
        return checkpoint
    else:
        raise RuntimeError("=> no checkpoint found at '{CHECKPOINT_filename}'")

