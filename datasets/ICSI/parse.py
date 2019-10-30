import glob
import xml.etree.ElementTree as ET
import subprocess
import os
import tqdm

FFMPEG = '../../ffmpeg-static/ffmpeg'

# Iterate through all meetings
paths = glob.glob('transcripts/*.mrt')
for mpath in tqdm.tqdm(paths, desc='Meetings'):

    # Skip preamble
    if 'preambles.mrt' in mpath:
        continue

    # Load tree
    tree = ET.parse(mpath)
    transcript = tree.getroot().findall('Transcript')

    # Each meeeting should have one and only one transcript
    assert len(transcript) == 1, f'ERR: Number of transcripts != 1: {transcript}'
    transcript = transcript[0]

    # Get meeting id from path
    mid = mpath[mpath.rindex('/') + 1:mpath.rindex('.')]
    for segment in tqdm.tqdm(transcript, desc='Segments'):

        # Collect segment info
        try:
            start = segment.attrib['StartTime']
            end   = segment.attrib['EndTime']
            sid   = segment.attrib['Participant']
        except KeyError:
            continue

        # Get audio input path
        apath = os.path.join('./ICSI/audio/', mid, 'chanF.wav')            

        # Get segment write path
        spath = os.path.join('./ICSI/segments/', sid)
        if not os.path.exists(spath):
            os.makedirs(spath)
        spath = os.path.join(spath, f'{mid}_{start}_{end}.wav')

        # Use ffmpeg to extract and write segment audio
        subprocess.call([FFMPEG', '-n', '-i', apath, '-ss', start, '-to', end, '-c', 'copy', spath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
