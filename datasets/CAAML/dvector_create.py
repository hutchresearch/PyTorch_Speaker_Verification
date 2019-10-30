"""
Created on Wed Dec 19 14:34:01 2018

@author: Harry

Creates "segment level d vector embeddings" compatible with
https://github.com/google/uis-rnn

"""

import glob
import librosa
import numpy as np
import os
import torch
import tqdm
import pandas

from hparam import hparam as hp
from speech_embedder_net import SpeechEmbedder
from VAD_segments import VAD_chunk
import xml.etree.ElementTree as ET

def main():
    # Setup vars
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load speech embedder net
    embedder_net = SpeechEmbedder()
    embedder_net.load_state_dict(torch.load(hp.model.model_path))
    embedder_net = embedder_net.to(device)
    embedder_net.eval()

    # Init dvector vars
    sequence   = []
    cluster_id = []
    label = 0
    count = 0
    train_saved = False

    transcript = pandas.read_csv()
        transcript = os.path.join('./ICSI/transcripts/', '{}.mrt'.format(audio_path.split('/')[3]))
        transcript = ET.parse(transcript).getroot().findall('Transcript')
        assert len(transcript) == 1, 'ERR: Number of transcripts != 1: {}'.format(transcript)
        transcript = transcript[0]

        # Segment the audio with VAD
        times, segs = VAD_chunk(2, audio_path)
        if segs == []:
            continue

        # Get STFT frames
        concat_seg, concat_time = concat(times, segs)
        STFT_frames, STFT_times = get_STFTs(concat_seg, concat_time)
        print(len(STFT_frames))
        print(len(STFT_times))
        STFT_frames = np.stack(STFT_frames, axis=2)
        STFT_frames = torch.tensor(np.transpose(STFT_frames, axes=(2,1,0))).to(device)

        # Get speaker embeddings
        BATCH_SIZE = 64
        EMBEDDING_SIZE = 256
        cur = 0
        embeddings = torch.empty(len(STFT_frames), EMBEDDING_SIZE)
        STFT_frames = torch.split(STFT_frames, BATCH_SIZE, dim=0)
        for frame in STFT_frames:
            nxt = cur + len(frame)
            embeddings[cur:nxt] = embedder_net(frame)
            cur = nxt
            
        # Align speaker embeddings
        aligned_embeddings = align_embeddings(embeddings.cpu().detach().numpy())

        # Build train sequences and labels
        # TODO: Make modular for multi-speaker meeting
        sequence.append(aligned_embeddings)

        exit()
        for embedding in aligned_embeddings:
            train_cluster_id.append(str(label))
        count = count + 1
        
        for segment in tqdm.tqdm(transcript, desc="Segment", position=1):
            pass
        
        # Save training set
        if not train_saved and i > train_speaker_num:
            sequence = np.concatenate(sequence,axis=0)
            train_cluster_id = np.asarray(cluster_id)
            np.save('train_sequence', sequence)
            np.save('train_cluster_id', cluster_id)
            train_saved = True
            sequence = []
            tcluster_id = []

    print('Processed {0}/{1} files ({2} ns {3} nf)'.format(count, nfiles, ns, nf))
            
    sequence = np.concatenate(sequence,axis=0)
    cluster_id = np.asarray(cluster_id)
    np.save('test_sequence', sequence)
    np.save('test_cluster_id', cluster_id)


def concat(times, segs):
    # Times are 0.4 second intervals
    # Segs contains amplitudes? detections? @ 16k Hz
    # Concatenate continuous voiced segments
    concat_seg = []
    concat_time = []

    time_start = times[0][0]
    seg_concat = segs[0]
    for i in range(0, len(times)-1):
        if times[i][1] == times[i+1][0]:
            seg_concat = np.concatenate((seg_concat, segs[i+1]))
        else:
            concat_seg.append(seg_concat)
            concat_time.append((time_start, times[i][1]))
            seg_concat = segs[i+1]
            time_start = times[i+1][0]
    else:
        concat_seg.append(seg_concat)
        concat_time.append((time_start, times[-1][1]))

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

def align_embeddings(embeddings):
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
    for i, partition in enumerate(partitions):
        avg_embeddings[i] = np.average(embeddings[partition[0]:partition[1]],axis=0) 
    return avg_embeddings

if __name__ == '__main__':
    main()