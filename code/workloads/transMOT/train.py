import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
from torch import optim


def MOT_loss(dec_logits, target, target_sink, panelty, dec_mask, debug=False):
    """
    ::param:: pred: output of the transmot with shape (M+1, N+1)
    ::param:: target: ground-truth mapping of all object with shape (M)
    ::param:: target_sink: ground-truth of all object that disappears, i.e. sink, with shape (N)
    """
    N = len(target_sink)
    M_t = len(target)
    # print(N, M_t)
    # print(target)
    # print(target_sink)
    target = target.to(torch.int64)
    target_sink = target_sink.to(torch.int64)
    # print(target_sink.shape, dec_logits.shape, target[dec_mask[1:]].shape)
    dec_logits = torch.squeeze(dec_logits, axis=0)  # simplified version for batch_size=1
    target_sink = target_sink[0: (dec_logits.shape[1] - 1)]
    if debug:
        print(dec_logits)
        print(target)
    # print(dec_logits[:-1].shape)
    # print(target)
    cross_entropy_loss = F.cross_entropy(dec_logits[1:], target[dec_mask[1:]])
    sink_prob = F.softmax(dec_logits[0, 1:])
    soft_margin_loss = target_sink * F.logsigmoid(sink_prob) + (1 - target_sink) * F.logsigmoid(-sink_prob)
    return cross_entropy_loss + panelty * torch.mean(soft_margin_loss)


def get_accuracy(outputs, label_tracks, label_sinks, dec_mask, is_cuda, gamma=None):
    label_tracks = label_tracks.to(torch.int64)
    cross_entropy_loss = F.cross_entropy(outputs[1:], label_tracks[dec_mask[1:]])
    cross_entropy_loss = float(cross_entropy_loss)
    pred = F.softmax(outputs, dim=1)
    label_sinks = label_sinks.to(torch.int64)
    pred_tracks = torch.argmax(pred[1:], dim=1).to(torch.int64)
    acc_track = float(torch.sum(pred_tracks == label_tracks[dec_mask[1:]]) / len(pred_tracks))
    pred_sinks = torch.zeros(pred.shape[1] - 1).to(torch.int64)
    pred_sinks[pred[0, 1:] > 0.5] = 1
    if is_cuda:
        pred_sinks = pred_sinks.cuda()
    label_sinks = label_sinks[0: len(pred_sinks)]
    acc_sink = float(torch.sum(pred_sinks == label_sinks) / len(pred_sinks))
    if gamma:
        return cross_entropy_loss, acc_track * gamma + acc_sink * (1 - gamma), 0
    else:
        return cross_entropy_loss, acc_track, acc_sink


def test_transformer_no_batch(model, test_dataloader, gamma=None):
    # TODO: current the model does not support batch, so dataloader is torch datasets
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        model = model.cuda()
    model.eval()
    t = time.time()
    accuracies_track = []
    accuracies_sink = []
    losses = []
    test_idx = np.arange(len(test_dataloader))
    for idx in test_idx:
        curr_frame_no, dec_features, dec_edges, dec_mask, enc_features, enc_edges, enc_mask, label_tracks, label_sinks = \
        test_dataloader[idx]
        if curr_frame_no == 0:
            # frist frame of a video has no tracking
            continue

        enc_features = Variable(enc_features, requires_grad=False)
        enc_edges = Variable(enc_edges, requires_grad=False)
        enc_mask = Variable(enc_mask, requires_grad=False)
        dec_features = Variable(dec_features.unsqueeze(0), requires_grad=False)
        dec_edges = Variable(dec_edges.unsqueeze(0), requires_grad=False)
        dec_mask = Variable(dec_mask.unsqueeze(0), requires_grad=False)

        if is_cuda:
            enc_features = enc_features.cuda()
            enc_edges = enc_edges.cuda()
            enc_mask = enc_mask.cuda()
            dec_features = dec_features.cuda()
            dec_edges = dec_edges.cuda()
            dec_mask = dec_mask.cuda()
            label_tracks = label_tracks.cuda()
            label_sinks = label_sinks.cuda()

        outputs = model(enc_features, enc_edges, enc_mask, dec_features, dec_edges, dec_mask)
        dec_mask = dec_mask.squeeze(0)
        loss, acc_track, acc_sink = get_accuracy(outputs, label_tracks, label_sinks, dec_mask, is_cuda)
        accuracies_track.append(acc_track)
        accuracies_sink.append(acc_sink)
        losses.append(loss)
    inference_time = time.time() - t
    return np.mean(losses), np.mean(accuracies_track), np.mean(accuracies_sink), inference_time / (
                len(test_dataloader) - 1)


def train_transformer_no_batch(model, train_dataloader, test_dataloader, lr=0.001, num_epoch=50, batch_size=32,
                               eval_every=5, panelty=0.1, debug=False):
    # TODO: current the model does not support batch, so dataloader is torch datasets
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        model = model.cuda()
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epoch):
        train_idx = np.arange(len(train_dataloader))
        np.random.shuffle(train_idx)
        curr_loss = 0
        for num_iter, idx in enumerate(train_idx):
            curr_frame_no, dec_features, dec_edges, dec_mask, enc_features, enc_edges, enc_mask, label_tracks, label_sinks = \
            train_dataloader[idx]
            if curr_frame_no == 0:
                # frist frame of a video has no tracking
                continue

            enc_features = Variable(enc_features, requires_grad=False)
            enc_edges = Variable(enc_edges, requires_grad=False)
            enc_mask = Variable(enc_mask, requires_grad=False)
            dec_features = Variable(dec_features.unsqueeze(0), requires_grad=False)
            dec_edges = Variable(dec_edges.unsqueeze(0), requires_grad=False)
            dec_mask = Variable(dec_mask.unsqueeze(0), requires_grad=False)

            if is_cuda:
                enc_features = enc_features.cuda()
                enc_edges = enc_edges.cuda()
                enc_mask = enc_mask.cuda()
                dec_features = dec_features.cuda()
                dec_edges = dec_edges.cuda()
                dec_mask = dec_mask.cuda()
                label_tracks = label_tracks.cuda()
                label_sinks = label_sinks.cuda()
            if (num_iter) % batch_size == 0:
                outputs = model(enc_features, enc_edges, enc_mask, dec_features, dec_edges, dec_mask, debug)
            else:
                outputs = model(enc_features, enc_edges, enc_mask, dec_features, dec_edges, dec_mask)
            dec_mask = dec_mask.squeeze(0)
            # print(outputs)
            # print(label_tracks)

            if (num_iter + 1) % batch_size == 0 or num_iter == len(train_idx) - 1:
                curr_loss += MOT_loss(outputs, label_tracks, label_sinks, panelty, dec_mask, debug)
                optimizer.zero_grad()
                curr_loss.backward()
                optimizer.step()
                print(f"epoch {epoch}, batch {(num_iter + 1) // batch_size}: {curr_loss.item() / batch_size}")
                curr_loss = 0
            else:
                loss = MOT_loss(outputs, label_tracks, label_sinks, panelty, dec_mask)
                curr_loss += loss
        losses, acc_track, acc_sink, inference_time = test_transformer_no_batch(model, test_dataloader)
        print(
            f"epoch {epoch}, loss: {losses}, tracking prediction accuracy: {acc_track}, sink prediction accuracy: {acc_sink}, inference time: {inference_time}")


def train_transformer_single_data(model, train_dataloader, test_dataloader, num_epoch=500, batch_size=32, eval_every=5,
                                  panelty=0.5, debug=False):
    # TODO: current the model does not support batch, so dataloader is torch datasets
    is_cuda = torch.cuda.is_available()
    is_cuda = False
    if is_cuda:
        model = model.cuda()
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    idx = 20
    for epoch in range(num_epoch):
        curr_frame_no, dec_features, dec_edges, dec_mask, enc_features, enc_edges, enc_mask, label_tracks, label_sinks = \
        train_dataloader[idx]
        enc_features = Variable(enc_features, requires_grad=False)
        enc_edges = Variable(enc_edges, requires_grad=False)
        enc_mask = Variable(enc_mask, requires_grad=False)
        dec_features = Variable(dec_features.unsqueeze(0), requires_grad=False)
        dec_edges = Variable(dec_edges.unsqueeze(0), requires_grad=False)
        dec_mask = Variable(dec_mask.unsqueeze(0), requires_grad=False)

        if is_cuda:
            enc_features = enc_features.cuda()
            enc_edges = enc_edges.cuda()
            enc_mask = enc_mask.cuda()
            dec_features = dec_features.cuda()
            dec_edges = dec_edges.cuda()
            dec_mask = dec_mask.cuda()
            label_tracks = label_tracks.cuda()
            label_sinks = label_sinks.cuda()
        if (epoch) % batch_size == 0:
            t = time.time()
            outputs = model(enc_features, enc_edges, enc_mask, dec_features, dec_edges, dec_mask, debug)
            print(time.time() - t)
        else:
            outputs = model(enc_features, enc_edges, enc_mask, dec_features, dec_edges, dec_mask)
        dec_mask = dec_mask.squeeze(0)
        # print(outputs)
        # print(label_tracks)
        loss = MOT_loss(outputs, label_tracks, label_sinks, 0.1, dec_mask, debug)
        optimizer.zero_grad()
        loss.backward()
        if debug:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            for name, p in model.named_parameters():
                if p.requires_grad:
                    if p.grad is not None:
                        print(name, p.grad.norm())
                    else:
                        print(name, "NONE")
        optimizer.step()
        if (epoch + 1) % batch_size == 0:
            print(f"epoch {epoch}: {loss.item()}")


