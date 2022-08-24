import librosa
import pyrubberband as pyrb
import madmom
from madmom.features.downbeats import RNNDownBeatProcessor
from madmom.features.downbeats import DBNDownBeatTrackingProcessor
import os
import matplotlib.pyplot as plot
import numpy as np
import soundfile as sf
import json 
from collections import defaultdict
from multiprocessing import Pool
loop_dir= '/path/to/data'
out_dir= '/path/to/output'
os.makedirs(out_dir, exist_ok=True)
valid_loop = defaultdict(int)

def one_bar_segment(file):

    file_path = os.path.join(loop_dir, file)
    try:
        y, sr = librosa.core.load(file_path, sr=None)
    except:
        print('load file failed')
        return
    try:
        act = RNNDownBeatProcessor()(file_path)
        down_beat=proc(act) 
    except:
        print('except happended')
        return
    count = 0
    bar_list = []
    name = file.replace('.wav', '')
        
    for i in range(down_beat.shape[0]):
        if down_beat[i][1] == 1 and i + 4 < down_beat.shape[0] and down_beat[i+4][1] == 1:
            print(down_beat[i: i + 5, :])
            start_time = down_beat[i][0]
            end_time = down_beat[i + 4][0]
            count += 1
            out_path = os.path.join(out_dir, f'{name}_{count}.wav')
            y_one_bar, _ = librosa.core.load(file_path, offset=start_time, duration = end_time - start_time, sr=None)
            y_stretch = pyrb.time_stretch(y_one_bar, sr,  (end_time - start_time) / 2)
            sf.write(out_path, y_stretch, sr)
            print('save file: ',  f'{name}_{count}.wav')

if __name__ == '__main__':
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=4, fps = 100)
    file_list = list(os.listdir(loop_dir))
    with Pool(processes=10) as pool:
        pool.map(one_bar_segment, file_list)
    
    