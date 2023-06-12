import sys
import os
import numpy as np
sys.path.append('/XXX/skyscraper-python/workloads')
import torch
from torchvision import models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transMOT.feature_extraction import FeatureExtractor, extract_feature
from transMOT.object_detection import get_frames_from_gt
from transMOT.transmot import TransMOT
from transMOT.pygcn import GCN
import warnings
import time

warnings.filterwarnings('ignore')


class MOTDataset(Dataset):
    def __init__(self, frame_no, features, edges, tracks, sinks, T):
        self.frame_no = frame_no
        self.feature_mask = torch.tensor([len(t) for t in features]).to(torch.int64)
        self.features = pad_sequence(features, batch_first=True, padding_value=0)
        self.edges = self.square_pad(edges, padding_value=0)
        self.tracks = pad_sequence(tracks, batch_first=True, padding_value=0)
        self.sinks = pad_sequence(sinks, batch_first=True, padding_value=0)
        self.N = self.tracks.shape[1]
        self.T = T

    def __len__(self):
        return len(self.frame_no)

    def square_pad(self, edges, padding_value=0):
        N = self.features.shape[1]
        res = torch.zeros((len(edges), N, N))
        for i, edge in enumerate(edges):
            res[i, 0:len(edge), 0:len(edge)] = edge
        return res

    def __getitem__(self, idx):
        curr_frame_no = self.frame_no[idx]
        dec_features = F.pad(self.features[idx], (0, 0, 1, 0), 'constant', value=1)
        dec_edges = F.pad(self.edges[idx], (0, 1, 0, 1), 'constant', value=0.5)
        dec_mask = torch.zeros(self.N + 1)
        dec_mask[: (self.feature_mask[idx] + 1)] = 1
        dec_mask = dec_mask.to(torch.bool)

        label_tracks = self.tracks[idx]
        label_sinks = self.sinks[idx]

        if curr_frame_no >= self.T:
            enc_features = self.features[(idx - self.T):idx]
            enc_edges = self.edges[(idx - self.T):idx]
            enc_mask = torch.zeros(self.T, self.N)
            for i, j in enumerate(range(idx - self.T, idx)):
                enc_mask[i, :self.feature_mask[j]] = 1
            enc_mask = enc_mask.to(torch.bool)

        else:
            N = self.features.shape[1]
            num_feature = self.features.shape[2]
            if curr_frame_no == 0:
                # first frame of a video, no training
                N = dec_features.shape[0]
                num_feature = dec_features.shape[1]
                return 0, torch.zeros((N + 1, num_feature)), torch.zeros((N + 1, N + 1)), dec_mask, \
                       torch.zeros((self.T, N, num_feature)), torch.zeros((self.T, N, N)), torch.zeros(self.T, N).to(
                    torch.bool), \
                       torch.zeros(N), torch.zeros(N)
            else:
                enc_features = self.features[(idx - curr_frame_no):idx]
                enc_mask = torch.zeros(curr_frame_no, N)
                for i, j in enumerate(range((idx - curr_frame_no), idx)):
                    enc_mask[i, :self.feature_mask[j]] = 1
                enc_mask = enc_mask.to(torch.bool)
                enc_edges = self.edges[(idx - curr_frame_no):idx]

        return curr_frame_no, dec_features, dec_edges, dec_mask, enc_features, enc_edges, enc_mask, \
               label_tracks, label_sinks


def compute_gt_tracks(frames):
    """
        compute the ground truth tracks
    """
    all_frame_no = []
    all_features = []
    all_edges = []
    all_tracks = []
    all_sinks = []
    for i, frame in enumerate(frames):
        all_frame_no.append(i)
        all_features.append(torch.tensor(frame.features).to(torch.float32))
        all_edges.append(torch.tensor(frame.edges).to(torch.float32))
        if i == 0:
            all_tracks.append(torch.zeros(len(frame.raw_data)).to(torch.int64))
            all_sinks.append(torch.zeros(len(frame.raw_data)).to(torch.int64))
        else:
            curr_tracks = torch.zeros(len(frame.raw_data)).to(torch.int64)
            prev_frame = frames[i - 1]
            prev_id = list(prev_frame.raw_data[:, 0])
            curr_sink = torch.zeros(len(prev_id)).to(torch.int64)
            seen = set()
            for j, j_id in enumerate(frame.raw_data[:, 0]):
                if j_id in prev_id:
                    curr_tracks[j] = prev_id.index(j_id) + 1  # +1 for virtual source node
                    seen.add(j_id)
                else:
                    curr_tracks[j] = 0  # virtual source node

            for l, l_id in enumerate(prev_id):
                if l_id in seen:
                    curr_sink[l] = 0
                else:
                    curr_sink[l] = 1
            all_tracks.append(curr_tracks)
            all_sinks.append(curr_sink)
    return all_frame_no, all_features, all_edges, all_tracks, all_sinks


def make_dataset(data_folders, T=5):
    model = models.vgg16(pretrained=True)
    extractor = FeatureExtractor(model)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    extractor = extractor.to(device)

    all_frame_no = []
    all_features = []
    all_edges = []
    all_tracks = []
    all_sinks = []
    for data_folder in data_folders:
        for i, dataset in enumerate(os.listdir(data_folder)):
            print(dataset)
            img_path = data_folder + dataset + "/img1"
            gt_path = data_folder + dataset + "/gt/gt.txt"
            frames = get_frames_from_gt(img_path, gt_path)
            frames = extract_feature(extractor, device, frames)
            dataset_frame_no, dataset_features, dataset_edges, dataset_tracks, dataset_sinks = compute_gt_tracks(frames)
            all_frame_no.extend(dataset_frame_no)
            all_features.extend(dataset_features)
            all_edges.extend(dataset_edges)
            all_tracks.extend(dataset_tracks)
            all_sinks.extend(dataset_sinks)
            break
    return all_frame_no, all_features, all_edges, all_tracks, all_sinks, T
