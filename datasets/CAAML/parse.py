import glob
import subprocess
import os
import tqdm
import csv
import shutil

START = 0
STOP  = 1
LABEL = 2
INST_LABELS = ['l', 'iq', 'ia', 'a']
STU_LABELS  = ['sq', 'sa', 'sp']
FFMPEG = '../../ffmpeg-static/ffmpeg'

# audio/instructor/class/date/annotation
paths = glob.glob('audio/*/*/*/*.txt')
for mpath in tqdm.tqdm(paths, desc='Meetings'):

    mid = '_'.join(mpath.split('/')[1:-1])
    apath = os.path.join(mpath[:mpath.rindex('/')], mid + '.wav')

    if not os.path.exists(os.path.join('segments/', mid)):
        os.makedirs(os.path.join('segments/', mid, 'INST/'))
        os.makedirs(os.path.join('segments/', mid, 'STU/'))

    with open(mpath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            if row[LABEL] in INST_LABELS:
                spath = f'segments/{mid}/INST/'
            elif row[LABEL] in STU_LABELS:
                spath = f'segments/{mid}/STU/'
            else:
                continue

            start_time = row[START].split('.')[0]
            stop_time = row[START].split('.')[0]
            spath = os.path.join(spath, '_'.join([mid, start_time, stop_time]) + '.wav')
            subprocess.call(
                [FFMPEG, '-n', '-i', apath, '-ss', row[START], '-to', row[STOP], '-c', 'copy', spath],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )