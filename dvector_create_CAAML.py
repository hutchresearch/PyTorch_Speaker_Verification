"""
Created on Wed Dec 19 14:34:01 2018

@author: Harry

Creates "segment level d vector embeddings" compatible with
https://github.com/google/uis-rnn

Modified on Sun Nov 02

@author: Eric Slyman

+ Support for CAAML dataset
+ Non-stacked label output
+ Verbose processing

"""

import glob
import librosa
import numpy as np
import os
import torch
import tqdm
import pandas
import time
import random
import pickle

from hparam import hparam as hp
from speech_embedder_net import SpeechEmbedder
from VAD_segments import VAD_chunk

STU_LABELS  = ['sq', 'sa', 'sp']
INST_LABELS = ['l', 'iq', 'ia', 'a']
STU_LABEL  = 0
INST_LABEL = 1
NULL_LABEL = 2
VERBOSE = True

def main():
    # Setup vars
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load speech embedder net
    embedder_net = SpeechEmbedder()
    embedder_net.load_state_dict(torch.load(hp.model.model_path))
    embedder_net = embedder_net.to(device)
    embedder_net.eval()

    # Init dvector vars
    all_cluster_ids = []
    all_sequences   = []
    agressiveness   = 2

    paths = glob.glob('./data/CAAML/audio/*/*/*/annotations.txt') + glob.glob('./data/CAAML/audio/*/*/*/annotations1.txt')
    for tpath in paths:

        print('\n============== Processing {} ============== '.format(tpath))

        apath = os.path.join(tpath[:tpath.rindex('/')], '_'.join(tpath.split('/')[4:7]) + '.wav')
        if not os.path.isfile(apath):
            print('No .wav file found, skipping...')
            continue
        
        transcript = timeit(pandas.read_csv, 'Loading transcript')(
            tpath, delimiter='\t', header=None, 
            names=['start', 'stop', 'label'],
            dtype={'start': np.float32, 'stop': np.float32, 'label': str}
        )

        # Segment the audio with VAD
        times, segs = timeit(VAD_chunk, 'Getting VAD chunks')(agressiveness, apath)
        if segs == []:
            print('No segments found, skipping...')
            continue

        # Get STFT frames
        concat_seg, concat_time = concat(segs, times)
        STFT_frames, STFT_times = timeit(get_STFTs, 'Getting STFT frames')(concat_seg, concat_time)
        STFT_frames = np.stack(STFT_frames, axis=2)
        STFT_frames = torch.tensor(np.transpose(STFT_frames, axes=(2,1,0))).to(device)

        # Get speaker embeddings
        embeddings = timeit(get_speaker_embeddings, 'Getting speaker embeddings')(embedder_net, STFT_frames)

        # Align speaker embeddings
        aligned_embeddings, times = timeit(align_embeddings, 'Aligning speaker embeddings')(embeddings.cpu().detach().numpy(), STFT_times)

        cluster_ids = []
        sequences = []
        skipped = 0
        print('Building labels...', end='\r')
        for time, embedding in zip(times, aligned_embeddings):
            # Get all labels this frame crosses
            label_df = transcript[(transcript['start'] <= time[0]) & (transcript['stop'] >= time[1])]

            # Ensure we only evaluate valid frames
            if len(label_df) == 0:
                skipped += 1
                continue

            # Infer proper training/testing label
            label = label_df.iloc[0]['label']
            if label in STU_LABELS:
                label = STU_LABEL
            elif label in INST_LABELS:
                label = INST_LABEL
            else:
                label = NULL_LABEL

            cluster_ids.append(label)
            sequences.append(embedding)

        # Convert list of embeddings to 2d array [seq_length x emb_size]
        cluster_ids = np.asarray(cluster_ids)
        sequences = np.vstack(sequences)

        all_cluster_ids.append(cluster_ids)
        all_sequences.append(sequences) 
        print('Building labels...\tDone ({} skipped)'.format(skipped))

    # Get dset shuffled indicies
    shuffle = np.arange(len(all_cluster_ids))
    np.random.shuffle(shuffle)
    shuffle = np.array_split(shuffle, 10)

    # Save d-vectors
    splits = [('trn', 6), ('dev', 2), ('tst', 2)]
    cur = 0
    for split, nsegs in splits:
        idxs = np.concatenate(shuffle[cur:cur+nsegs])
        cids = [all_cluster_ids[i] for i in idxs]
        seqs = [all_sequences[i] for i in idxs]
        with open('{}_cluster_ids.pkl'.format(split), 'wb') as f:
            pickle.dump(cids, f)
        with open('{}_sequences.pkl'.format(split), 'wb') as f:
            pickle.dump(seqs, f)
        cur += nsegs
        print(len(cids))
    
def timeit(f, msg):
    def wrapped(*args, **kwargs):
        print('{}...'.format(msg), end='\r')
        s = time.time()
        res = f(*args, **kwargs)
        t = time.time() - s
        print('{}...\tDone ({:0.3f} seconds)'.format(msg, t))
        return res
    return wrapped if VERBOSE else f

def get_speaker_embeddings(embedder, frames, embedding_size=256, batch_size=64):
    embeddings = torch.empty(len(frames), embedding_size)
    frames = torch.split(frames, batch_size, dim=0)
    cur = 0
    for frame in frames:
        nxt = cur + len(frame)
        embeddings[cur:nxt] = embedder(frame)
        cur = nxt

    return embeddings

def concat(segs, times):
    # Times are 0.4 second intervals
    # Segs contains amplitudes? detections? @ 16k Hz
    # Concatenate continuous voiced segments
    concat_seg = []
    concat_time = []

    start = times[0][0]
    end = times[0][1]

    seg_concat = segs[0]
    for i in range(0, len(times)-1):
        if times[i][1] == times[i+1][0]:
            seg_concat = np.concatenate((seg_concat, segs[i+1]))
            end = times[i+1][1]
        else:
            concat_seg.append(seg_concat)
            seg_concat = segs[i+1]

            concat_time.append((start, end))
            start = times[i+1][0]
            end = times[i+1][1]
    else:
        concat_seg.append(seg_concat)
        concat_time.append((start, times[-1][1]))

    return concat_seg, concat_time

def get_STFTs(segs, times):
    #Get 240ms STFT windows with 50% overlap
    sr = hp.data.sr
    STFT_frames = []
    STFT_times = []
    for seg, time in zip(segs, times):
        S = librosa.core.stft(
            y=seg, 
            n_fft=hp.data.nfft,
            win_length=int(hp.data.window * sr), 
            hop_length=int(hp.data.hop * sr)
        )
        S = np.abs(S)**2
        mel_basis = librosa.filters.mel(sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
        S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
        for j in range(0, S.shape[1], int(.12/hp.data.hop)):
            if j + 24 < S.shape[1]:
                STFT_frames.append(S[:,j:j+24])
                STFT_times.append((
                    round(time[0] + (j / 100), 2), 
                    round(time[0] + ((j + 24) / 100), 2) 
                ))
            else:
                break
    return STFT_frames, STFT_times

def align_embeddings(embeddings, times):
    partitions = []
    start = 0
    end = 0
    j = 1
    for i, embedding in enumerate(embeddings):
        if (i*.12)+.24 < j*.401:
            end = end + 1
        else:
            partitions.append((start,end))
            start = end
            end = end + 1
            j += 1
    else:
        partitions.append((start,end))
    avg_embeddings = np.zeros((len(partitions),256))
    window = np.zeros((len(partitions), 2))
    for i, partition in enumerate(partitions):
        avg_embeddings[i] = np.average(embeddings[partition[0]:partition[1]],axis=0) 
        window[i] = times[partition[0]][0], times[partition[0]][1]
    return avg_embeddings, window

if __name__ == '__main__':
    main()
