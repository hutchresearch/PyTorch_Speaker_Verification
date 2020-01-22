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
import os
import torch
import pandas
import random
import pickle
import argparse
import sys
import csv
import xml.etree.ElementTree as ET
import numpy                 as np
from hparam              import hparam as hp
from speech_embedder_net import SpeechEmbedder
from VAD_segments        import VAD_chunk
from decorators          import timeit
from abc                 import ABC, abstractmethod

class Dataset(ABC):
    def __init__(self, data_path, save_path):
        self._data_path = glob.glob(data_path)
        self._save_path = save_path

    @property
    def data_path(self):
        return self._data_path

    @property
    def save_path(self):
        return self._save_path

    @abstractmethod
    def get_session(self, path):
        pass

    @abstractmethod
    def get_annotations(self, path):
        pass

    @abstractmethod
    def save(self, all_seqs, all_cids, all_times, sessions):
        pass

class ICSIDataset(Dataset):
    def __init__(self, data_path, save_path):
        super().__init__(data_path, save_path)

    def get_session(self, path):
        return path.split('/')[-2]

    @timeit(msg='Getting annotations')
    def get_annotations(self, path):
        transcript = os.path.dirname(path).replace('audio', 'transcripts') + '.mrt'
        transcript = ET.parse(transcript).getroot().findall('Transcript')
        assert len(transcript) == 1, 'ERR: Number of transcripts != 1: {}'.format(transcript)

        rows = [{
            'start': float(segment.attrib.get('StartTime')),
            'stop' : float(segment.attrib.get('EndTime')),
            'label': segment.attrib.get('Participant')
        } for segment in transcript[0]]

        annotations = pandas.DataFrame(rows, columns=['start', 'stop', 'label'],)
        annotations.dropna(inplace=True)

        return annotations

    def save(self, all_seqs, all_cids, all_times, sessions):
        # Get a shuffled set of indices for each split
        shuffle = np.arange(len(all_cids))
        np.random.shuffle(shuffle)
        shuffle = np.array_split(shuffle, 10)
        splits = [('trn', 9), ('tst', 1)]
  
        entries = [('seqs', all_seqs), ('cids', all_cids), ('times', all_times), 
                    ('sessions', sessionss)]
        cur = 0
        for split, nsegs in splits:
            # Gather indices for this split and save
            idxs = np.concatenate(shuffle[cur:cur+nsegs])
            d = {n: [l[i] for i in idxs] for n, l in entries}

            path = os.path.join(self.save_path, 'icsi_{}.pkl'.format(split))
            with open(path, 'wb') as f:
                pickle.dump(d, f)

            cur += nsegs
            
class CAAMLDataset(Dataset):
    def __init__(self, data_path, save_path, split, *, 
        inst_labels=['l', 'iq', 'ia', 'a'], 
        stu_labels=['sq', 'sa', 'sp']):
        
        super().__init__(data_path, save_path)
        with open(split) as split_csv:
            csv_reader = csv.reader(split_csv, delimiter=',')
            split_dict = {x[0]: x[1] for x in csv_reader}
        
        self._inst_labels = inst_labels
        self._stu_labels  = stu_labels
        self._stu_id      = 0
        self._split       = split_dict

    def get_session(self, path):
        return os.path.basename(path).split('.')[0]

    @timeit(msg='Getting annotations')
    def get_annotations(self, path):
        annot_file = os.path.join(os.path.dirname(path), 'annotations0.txt')

        if not os.path.isfile(annot_file):
            return None

        annotations = pandas.read_csv(
            annot_file, delimiter='\t', header=None, 
            names=['start', 'stop', 'label'],
            dtype={'start': np.float32, 'stop': np.float32, 'label': str}
        )

        inst_id = 'I' + path.split('/')[-4]
        for i, r in annotations.iterrows():
            if r['label'] in self._inst_labels:
                label = inst_id
            elif r['label'] in self._stu_labels:
                label = 'S' + str(self._stu_id).zfill(4)
                self._stu_id += 1
            else:
                label = 'None'
            annotations.loc[i, 'label'] = label

        return annotations

    def save(self, all_seqs, all_cids, all_times, sessions):
        # Build dataset split
        splits = {}
        for seq, cid, time, session in zip(all_seqs, all_cids, all_times, sessions):
            split = self._split[session]
            if split not in splits:
                splits[split] = {'seqs': [], 'cids': [], 'times': [], 'sessions': []}
            splits[split]['seqs'].append(seq)
            splits[split]['cids'].append(cid)
            splits[split]['times'].append(time)
            splits[split]['sessions'].append(session)

        # Write splits to disk
        for split in splits.keys():
            path = os.path.join(self.save_path, 'caaml_{}.pkl'.format(split))
            with open(path, 'wb') as f:
                pickle.dump(splits[split], f) 

def main():
    args = get_args()
    if args.corpus == 'CAAML':
        dataset = CAAMLDataset(args.data_path, args.save_path, args.split)
    elif args.corpus == 'ICSI':
        dataset = ICSIDataset(args.data_path, args.save_path)
    elif args.corpus == 'TIMIT':
        print('Dataset not yet implemented...')
        exit()

    # Load speech embedder net
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedder_net = SpeechEmbedder()
    embedder_net.load_state_dict(torch.load(hp.model.model_path))
    embedder_net = embedder_net.to(device)
    embedder_net.eval()

    all_seqs = []
    all_cids = []
    all_times = []
    sessions = []
    for path in dataset.data_path:
        print('\n============== Processing {} ============== '.format(path))

        # Get session name
        session = dataset.get_session(path)
        if session is None:
            print('ERR: Session not found in any split, skipping...')
            continue
 
        # Get annotations
        annotations = dataset.get_annotations(path)
        if annotations is None:
            print('ERR: No suitable annotations found, skipping...')
            continue

        # Segment the audio with VAD
        times, segments = timeit(VAD_chunk, msg='Getting VAD chunks')(hp.data.aggressiveness, path)
        if segments == []:
            print('ERR: No segments found, skipping...')
            continue

        # Concatenate segments
        segments, times = concat(segments, times)

        # Get STFT frames
        frames, times = get_STFTs(segments, times)
        frames = np.stack(frames, axis=2)
        frames = torch.tensor(np.transpose(frames, axes=(2,1,0))).to(device)

        # Get speaker embeddings
        embeddings = get_speaker_embeddings(embedder_net, frames)

        # Align speaker embeddings into a standard sequence of embeddings
        sequence, times = align_embeddings(embeddings.cpu().detach().numpy(), times)
        
        # Get cluster ids for each frame
        cluster_ids = get_cluster_ids(times, dataset, annotations)

        # Add the sequence and cluster ids to the list of all sessions
        all_seqs.append(sequence) 
        all_cids.append(cluster_ids)
        all_times.append(times)
        sessions.append(session)

    # Save split dataset 
    dataset.save(all_seqs, all_cids, all_times, sessions)

def get_args():
    parser = argparse.ArgumentParser(description='Extract dvectors from some dataset.')
    parser.add_argument('--corpus', '-c', metavar='DATASET', choices=['CAAML', 'ICSI', 'TIMIT'],
                        required=True, help='Dataset to process (CAAML, ICSI, TIMIT)')
    parser.add_argument('--data_path', '-d', metavar='DATA_PATH', 
                        help='Glob path for dataset audio')
    parser.add_argument('--save_path', '-s', metavar='SAVE_PATH', default='./', 
                        help='Location to save dvectors')
    parser.add_argument('--split', required='CAAML' in sys.argv, metavar='SPLIT', 
                        help='Path of .csv defining dataset splits')
    args = parser.parse_args()

    if args.data_path is None:
        args.data_path = {
            'CAAML': './data/CAAML/audio/*/*/*/*.wav',
            'ICSI' : './data/ICSI/audio/*/chanF.wav'
        }[args.corpus]

    return args

@timeit(msg='Getting speaker embeddings')
def get_speaker_embeddings(embedder, frames, batch_size=64):
    with torch.no_grad():
        embeddings = torch.empty(len(frames), hp.model.proj)
        frames = torch.split(frames, batch_size, dim=0)
        cur = 0
        for frame in frames:
            nxt = cur + len(frame)
            embeddings[cur:nxt] = embedder(frame)
            cur = nxt

    return embeddings

@timeit(msg='Concatenating segments')
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

@timeit(msg='Getting STFT frames')
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

@timeit(msg='Aligning speaker embeddings')
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
    avg_embeddings = np.zeros((len(partitions), hp.model.proj))
    window = np.zeros((len(partitions), 2))
    for i, partition in enumerate(partitions):
        avg_embeddings[i] = np.average(embeddings[partition[0]:partition[1]],axis=0) 
        window[i] = times[partition[0]][0], times[partition[0]][1]
    return avg_embeddings, window

@timeit(msg='Getting cluster IDs')
def get_cluster_ids(times, dataset, annotations):
    cluster_ids = []
    for time in times:
        # Get all annotations intersecting with frame
        label_df = annotations[~( (annotations['start'] >= time[1]) | (annotations['stop'] <= time[0]) )].copy()
        if len(label_df) == 0:
            cluster_ids.append('None')
            continue
        
        # Get id of maximum overlap label
        label_df['overlap'] = label_df['stop'].clip(upper=time[1]) - label_df['start'].clip(lower=time[0])
        cluster_ids.append(label_df.loc[label_df['overlap'].idxmax()['label']])
    return np.asarray(cluster_ids)

if __name__ == '__main__':
    main()
