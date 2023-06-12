import sys
import os
import re
import numpy as np
import json
import time
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#sys.path.append('../')
from workload import Workload
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/offline"))
#sys.path.append("../../src/offline")
from execution_utils import TaskGraph

# Pytorch
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, pairwise_distances

from tqdm import tqdm_notebook
from collections import defaultdict

# Model
from lflstm import make_model, multi_collate, multi_collate_sample

CUDA = torch.cuda.is_available()
FRAME_COUNT = 30

class StreamDataLoader():

    def __init__(self):
        self.gt_model = make_model(hidden=1.5, fc=2, dropout=0.2)
        if CUDA:
            self.gt_model = self.gt_model.cuda()
        self.gt_model.load_state_dict(torch.load('./cache/model_gt.std'))
        
        data = self._load_and_process_data()
        loaders, self.seg_videos = self._get_loaders(data)

        self.y_preds = self._compute_ground_truth(loaders)

    def _load_and_process_data(self):
        test = None
        test_split = ['7l3BNtSE0xc', 'dZFV0lyedX4', '286943', '126872', 'qgC8_emxSIU', 'kld9r0iFkWM', 'rC29Qub0U7A', '4YfyP0uIqw0', 'FMenDv3y8jc', '4wLP4elp1uM', 'KYQTwFVBzME', '27v7Blr0vjw', 'DnBHq5I52LM', 'HR18U0yAlTc', 'x266rUJQC_8', 'd1CDP6sMuLA', 'xSCvspXYU9k', '4EDblUpJieU', '4o4ilPK9rl8', '53609', 'SZ7HK5ns6mE', '243981', 'ySblgk7T7eQ', 'MYEyQUpMe3k', 'EujJ0SwiCRE', '3HyAaqre_Fk', 'iQDB_OkAQWs', 'gE7kUqMqQ9g', 'eFV7iFPYZB4', 'IRSxo_XXArg', '3hOlJf_JQDs', 'BRSyH6yfDLk', '1jogeKX0wGw', '3At-BKm9eYk', 'NVLPURuAVLU', 'pZye4zFzk3o', 'l1jW3OMXUzs', 'XKyumlBmix8', 'eKQKEi2-0Ws', 'WgI8IbJtXHw', 'tnWmVXZ87h0', 'YCEllKyaCrc', 'W1CWpktWtTs', '8wQhzezNcUY', '0bxhZ-LIfZY', 'lrjm6F3JJgg', 'Vdf1McvE9ao', 'eQc5uI7FKCU', '2QXHdu2zlQY', 'YCI-ZzclIPQ', '2Ky9DBSl49w', 'SKTyBOhDX6U', 'b86B3hP8ARM', '23656', 'kpS4BXif_Sw', 'dR68gbeOWOc', 'tC2KicUHB9Q', 'absh1hsZeF0', 'c5zxqITn3ZM', 'uogwnZGb-iE', '46495', 'Sq6DIhFxPqQ', 'PexNiFbPTYM', 'z441aDJvAcU', 'OORklkFql3k', 'WbtsuXkaGeg', 'grsV1YN1z5s', 'Gc_zIjqqUys', '424SXFTCFsA', 'P17tYiqMGRU', 'UweZVaFqruU', 'mzAu5gxjE-w', '8TDAP0KNIIw', 'u9ZV8jb_-U0', 'iPPp6MCythU', 'lwL4hjhkid4', '102389', 'frCWtiam4tE', 'pSxte-ms0t8', 'c9hE1ghElrM', 'WfNiQBXmPw8', '_q7DM8WkzAQ', '257534', 'fU5AYkq0m9k', 'q17gSr9kNww', 'AgH84SNRx5s', '206585', 'yzCHa2qchpg', 'GmpDbIstUdc', 'eREud0qYR3s', 'NoOt0oU843M', 'svsbGQn389o', 'ZsLrKF7_Oos', 'Kyz32PTyP4I', '7idU7rR77Ss', '8lfS97s2AKc', 'X2Hs89fZ2-c', '5vwXp27bCLw', 'tZDNinnrGf8', 'KB5hSnV1emg', 'TxRtSItpGMo', 'eJfT7-dDqzA', 'x2n19Cn96aw', 'XDVit9ASVUg', '6brtMLkjjYk', '-rxZxtG0xmY', 'JATMzuV6sUE', 'LueQT0vY1zI', '267466', 'm-7yRWZLwLY', 'OWWHjP3pX9o', 'QnYlpSeVOYo', 'V7OFSHYcQD0', 'GK-Pprzh0t0', 'yLo-Jl8nBXU', '200941', '61531', 'ezuWKsxPRSM', 'ehZrOdw6PhA', '-6rXp3zJ3kc', 'Z4iYSMMMycQ', 'MtIklGnIMGo', '116213', '3XShFTBsp_Q', 'YQZPnHZRp1w', 'fsd1qPLA3kY', '208322', 'uVM4JWjfGgQ', 'QWll4lS1qqI', 'Vlas-mPydcg', 'teQqaAmqqx0', 'AQ4Pktv4-Gc', 'yCpHmPSshKY', 'pDRdCSIyjkA', 'sIusv36VoBY', '8jY_RhOS89o', 'GKsjv42t284', 'HA2AiTz-qxc', 'GXIfrEUJ5d4', '0Fqav67TDEw', '10219', 'wHeZHLv9wGI', 'qDfSYz0PX9g', '180971', 'qBanrqkzobg', 'NgENhhXf0LE', 'SH0OYx3fR7s', 'lc5bSoGlQwY', 'XrNaL-MTjXg', '8i7u3fl-hP8', 'N-NnCI6U52c', 'r46amqjpWgg', 'QVyxySAaehE', 'JKueLneBoik', '110565', 'TqQgIxR7thU', '267694', 'ZKErPftd--w', 'GMa0cIAltnw', 'MSHBEntSDjU', 'AB1PbMaW03s', 'oBS-IW-BO00', '5fKPJPFqPho', '8NPaDkOiXw4', '104741', '2ItiGjefTRA', 'LFOwCSiGOvw', 'YLK58srjhNI', '233939', '5xa0Ac2krGs', 'CbRexsp1HKw', '112425', 'bCBMKwafZKY', '_0efYOjQYRc', 'O2ShYliS3CU', 'Oa2xVjzAMFc', 'bnzVR2ETQQ8', 'yBtMwyQFXwA', 'TtAyUQtmTLk', 'yXE_XZUb8qE', '3wHE78v9zr4', 'cml9rShionM', 'rePYTlT5_AY', '9TAGpMywQyE', 'ryE9VBiR3p8', '238063', 'NlrCjfHELLE', 'oGFDE-6nd7Q', 'bWmrrWQOVCM', '29751', '11UtTaDYgII', 'jZe-2w7pkd8', '275267', 'tymso_pAxhk', 'PcqKFt1p3UQ', 'sfaWfZ2-4c0', 'kLAXmTx2xOA', '3OYY5Tsz_2k', 'wk5lFk5kFjY', 'hE-sA5umuCk', '3IUVpwx23cY', '92291', '102213', '236442', 'nbru7qLot04', 'zhNksSReaQk', '8VhVf0TbjDA', '35694', '20LfN8ENbhM', '257277', 'VsXGwSZazwA', 'EyoMU2yoJPY', 'E1r0FrFyNTw', 'CO2YoTZbUr0', 'wC_1M7KIv9s', '24196', '194299', 'R9xTBw3MCWI', 'cY8DcaHXNNs', 'SwT0gh0V8fI', 'UiurP5k-f1A', 'N0d2JL7JC1s', '208592', 'GAVpYuhMZAw', 'pvIQWWiT4-0', 'namehdJxRIM', 'tNd3--lvSXE', 'NaWmaHwjElo', 'mfpR4CN9LZo', 'U6IqFbpM-VM', 'XLjpZUsFEXo', 'YUNxD04EvfE', 'hI7ObFqn9Bg', 'CO6n-skQJss', 'RsE2gYghZ2s', '2ze94yo2aPo', '254427', 'MHyW857u_X8', 'Xa086gxLJ3Y', 'Uu_XyXyKHAk', 'TsfyeZ8hgwE', 'vI5JH0daNsM', 'mmg_eTDHjkk', 'lD4xtQ6NpDY', 'XWIp0zH3qDM', '259470', '0eTibWQdO5M', 'fcxbB7ybUfs', '5pxFqJ5hKMU', '245582', 'WQFRctNL8AA', '2m58ShI1QSI', 'cn0WZ8-0Z1Y', '25640', 'huzEsVEJPaY', 'UTErYLpdAY0', 'F2hc2FLOdhI', 'vGxqVh_kJdo', 'F_YaG_pvZrA', 'UNLD7eYPzfQ', '0K7dCp80n9c', 'xBE9YWYGjtk', 'nTZSH0EwpnY', 'mZ_8em_-CGc', 'fdc7iyzKvFQ', '221137', 'QBc7X5jj8oA', 'pnpFPX34Agk', '63951', 'veA6ECsGFxI', 'XbkYA-iXUwA', '1LkYxsqRPZM', 'qAip3lZRj-g', 'gR3igiwaeyc', 'pIaEcqnzI-s', 'oHff2W51wZ8', 'XlTYSOaZ_vM', '3WZ6R9B0PcU', 'IOpWjKAHG8Q', '53766', '190743', '107585', 'SYQ_zv8dWng', 'hBzw4r0kfjA', '0uftSGdwo0Q', 'jj8aSNPHMw8', '86c2OkQ3_U8', 'rhQB8e999-Q', 'qyqVc352g3Q', '1zXAYdPdzy8', 'nZFPKP9kBkw', 'A1lFJXUpxZo', '-cEhr0cQcDM', 'Kn5eKHlPD0k', '255408', 'eD5cScqaf6c', 'FHDVQkU-nGI', '24351', 'NOGhjdK-rDI', 'fz-MzQcOBwQ', 'DjcZrtcBZi4', '1HS2HcN2LDo', '209758', '2o2ljA0QD7g', '211875', '5lrDS7LluCA', 'ybK5wRaaUyE', 'M6HKUzJNlsE', 'QIonRUsCqBs', 'k8yDywC4gt8', 'jPtaz1rN6lc', '69824', 'kI6jzM_aLGs', 'x8UZQkN52o4', 'ZKZ8UjaQQT4', 'obGF3RfWQKE', '221153', 'YgyeyooSz0g', 'faUvT7zfsyk', 'ddWHTdJz2O8', 'OKJPFisBoPY', 'HAnQVHOd3hg', 'EO_5o9Gup6g', 'F7zQPzwFToE', '273250', '1pl2FVdQWj0', '91844', 'bvycs3BXtx0', 'hbJfSyJKBEA', 'ZHUFlEgKk-w', 'OyK86reBnJE', 'xwvSYhQrHoA', 'H-74k5vclCU', 'Sb6ftNgzz9M', 'Hq3cHc6X8BM', 'jscKL5jS-SQ', '2vsgDSlJ9pU', 'DtbT85s3i94', 'LcfubBagG6Q', 'f-VdKweez2U', 'a8UMRrUjavI', 'MvEw24PU2Ac', 'MZUr1DfYNNw', 'UcV-bpksJi0', '2W-U94hXuK0', 'OctOcfI4KSs', 'NocexkPXja8', 'eUwbRLhV1vs', 'bdFCD-3BCRg', 'TXiOSZdaLJ8', 'XadAy93f1P8', '136196', 'gJjkCPO7iXg', '210433', 'oH9fMma8jiQ', 'd-Uw_uZyUys', 'oQizLbmte0c', 'X6N7UEFJLNY', '0PlQc98SccA', '3REPHw7oLWo', 'vB_kZocHtYo', '2BuFtglEcaY', 'HMRqR-P68Ws', 'V27mmEkN80g', 'Y2F51I-dzAg', 'dTcz1am1eUw', 'gL8h7lOPv1Q', 'WoL4fCxGd8Q', '135623', '41381', 'IHp8hd1jm6k', 'dHk--ExZbHs', 'o2XbNJDpOlc', 'V2X1NU5RkwY', '9orb0lQnVW4', 'fsBzpr4k3rY', '2fbBrB1nJEQ', 'um8WVjZMLUc', 'eE8Qr9fOvVA', 'fVCDn6SdtVM', '83400', 'an_GzG40hcE', 'xkEK17UUyi4', 'y3r2kk8zvl0', 'KanWhGY33Hk', '210259', 'DR65no1jCbg', 'lkIe41StoGI', 'RB3HA-ZMtFw', 'qEuJj4uW93E', 'ydzNAuqUAnc', 'GO0V4ZGSF28', '9PzZSheh10U', '6RFVsOWK1m4', '-s9qJ7ATP7w', 'ey1lr8wFFDc', 'oZxMx8e0x2U', 'UjqA6KVW2m8', 'OaWYjsS02fk', '79356', '34cU3HO_hEA', 'KZzFCrEyKF0', 'c5VEvmutmVg', 'O4UkHVJlIs8', '22373', 'v_8QeoNc4QY', 'BR2pkk3TK-0', 'EMS14J0odIE', '221274', '92496', 'DMtFQjrY7Hg', 'h1ZZHUU4j0k', 'gpn71-aKWwQ', 'tW5xAWDnbGU', '88791', 'vJDDEuE-FlY', 'kaudsLIvYC8', 'x0rLwBIocuI', 'wnL3ld9bM2o', '8wNr-NQImFg', '37NMbnd7r20', '56989', 'ctAZf4sMBUQ', '7npCA0zoQ8Q', 'u9I5WD3Nglk', 'IIPYcCii7Sg', 'JNhqI4JtPXA', 'Bpy61RdLAvo', 'C5-cY1nPQ20', 'ihPjsliqz5o', '4t5k_yILGJM', 'mgsvwAVQAQo', 'Ie6sDptjAsU', 'lkeVfgI0eEk', 'O-b3DQg0QmA', 'PzI6yY9Y2xQ', 'lYwgLa4R5XQ', 'NIpuKcoJhGM', 'LpTbjgELALo', '0YiAKqU36KE', 'ZznoGQVwTtw', 'QCR7uyowjhM', 'ossKC1VrusE', 'p4WmcxrXkc4', 'ZS1Nb0OWYNE', 'P0UHzR4CmYg', 'qTkazqluJ_I', '252097', '0JaYazphxfM', 'zvZd3V5D5Ik', '-RfYyzHpjk4', 'l0vCKpk6Aes', 'ktblaVOnFVE', 'KRje7su4I5U', 'FqEekswKPWE', '130366', 'HFPGeaEPy9o', '-HeZS2-Prhc', '93iGT5oueTA', 'L-7oRnbi9-w', 'pQpy7RSfWzM', 'YsMM_1R1vtw', 'SBFnCNzlynQ', '4dAYMzRyndc', 'CU6U-gS76K4', 'NiAjz4ciA60', '-9y-fZ3swSY', 'LD3HYOwk1Bc', 'QXXexH-ow_k', '-UUCSKoHeMA', 'RVC8l5hf2Eg', '89ZmOPOilu4', 'xobMRm5Vs44', 'xmLJHru6Z1M', 'zfZUOvZZTuk', 'LJGL2sGvSS0', 'Pbu6oAKFCvo', 'nFTo-Lz4Fr8', 'CwF97vXPYX4', 'WJM8u2I2rQ4', '8XZszYrUSCQ', 'fT6SrlsWV7M', 'fWAKek8jA5M', 'jXQmVFcOiUI', 'KrmVX-HANew', 'kXhJ3hHK9hQ', 'oPlnhc0DkcU', 'OFia3dWgaoI', 'VVtx4IDsHZA', 'VIVkYG31Oas', 'I9iV9q3ePhI', 'ZeH7gZw94k0', 'wznRBN1fWj4', '226601', 'TcTGCIh6e5s', '1S6ji_d4OLI', 'mRqqH_gx7Q0', 'Zb7bdrjWyEY', 'VDkBM0ZG4q8', 'BJS5KCSowgU', '3UOqbf_B_Yc', 'LyOq9mOEPuk', '-MeTTeMJBNc', 'QLI-OnegxFM', '3odZe3AGilc', 'VXy21GwGs8o', '7deoh0NoMs4', 'DzdPl68gV5o', 'VS7xSvno7NA', 'H_x5O9GdknI', '67uKYi7mslE', 'Kn99u05vlpA', 'gKojBrXHhUc', 'AHiA9hohKr8', 'kmgsC68hIL8', 'XVWiAArXYpE', 'X_TusmBIlC0', 'dlE05KC95uk', 'KXsjl0DKgL0', 'b92iw0OAnI4', 'j1m6ctAgjsM', '6EDoVEm16fU', 'jE3gYsxr_5s', '6EJHA6IDLNk', 'xXXcgb9eZ9Y', 'rcfnqiD0y8o', '224370', '237363', '7IxmlIwqigw', '6gtfeIqGasE', '9GzFQBljNIY', 'vGkIaClHKDg', 'iXiMifHNKPI', 'BTjV5dU_rfY', 'g6VJg6ycUk0', 'Y8dI1GTWCk4', 'y5Jpf48SUX8', 'iFxFTtCQ6zA', 'ZKKNd0zR8Io', 'U8VYG_g6yVE', 'GNP0PFas12Y', 'ussfmzwftQ8', 'jLN6B0aSrW0', 'OXsTIPdytiw', '121400', 'CKqDDh50gcU', 'HJTxq72GuMs', '273207', 'lxBKEPIUSgc', '234046', '59673', 'USkMa4Evv7U', '213327', 'kXiBdruxTvE', '201005', '94481', 'ChhZna-aBK4', 'a4PHWxILDp0', 'SqAiJrvHXNA', 'kg-W6-hP2Do', '9Lr4i7bIB6w', 'fhADeWE5VgU', '-yRb-Jum7EQ', 'DVAY-u_cJWU', 'wd8VQ5E7o7o', 'N2nCB34Am-E', 'f8Puta8k8fU', '272838', 'Qfa1fY_07bQ', 'WBA79Q3e_PU', 'ozA7pRW4gFM', 'WuaoxGAKue0', '-AUZQgSxyPQ', 'l4oMbKDuW3Y', '198112', 'H9BNzzxlscA', '224631', '111881', 'U-KihZeIfKI', 'JXYot3NDQn0', '7f3ndBCx_JE', 'F8eQI8E-6q4', '112509', '46615', 'p1zqEGwRMI8', 'Wu-wQTmxRgo', 'V0SvSPkiJUY', '28006', 'cia8OM3Oe7Q', '_aJghSQmxD8', '97ENTofrmNo', '252912', 'bkX5FOX22Tw', '108146', 'Iemw8vqt-54', 'vTAV6FThy30', '_WA0HtVEe8U', '132028', 'nXWNHVZtV9A', 'MLegxOZBGUc', '224263', 'q5M1God4M6Y', '9cYNT_YSo8o', '22689', 'SD6wphlw72U', 'SqofxdeEcjg', '_on_E-ZWy0I', '222247', 'cX8FScpsfLE', 'k1BrzX5bc7U', 'Rb1uzHNcYcA', 'RChkXDS8ulE', '7mb8Y2AhXIY', '226640', 'MM2qPdnRmQ8', 'unOeTDc2rlY', 'ROC2YI3tDsk', 'AlJX3Jw3xKk', 'MHVrwCEWLPI', 'bNQOeiAotbk', 'TLPlduck5II', 'An8x4UfwZ7k', 'JHJOK6cdW-0', 'xU3N7ujUB-g', 'RvmTZVNAAuQ', 'LtlL-03S79Q', 'Xy57UpKRNEo', 'lO6N9dyvPTA', 'cW-aX4dPVfk', 'VwGPIUNayKM', 'p7zuPEZgtY4', 'ZtocGyL3Tfc', '24504', '5eY4S1F62Z4', 'ttfaZ8IIWCo', 'z0y1ZxH1f74', 'VLQgw-88v4Q', '1Gp4l-ZTCVk', 'QJBQIOmG1CA', 'jqutn5ou8_0', 'gcFECfN4BCU', 'Dm8AL84e11w', 'tO68uTk-T_E', '215318', 'DlX-WyVe-D0', 'gLTxaEcx41E', 'RmX9t_HY_dU', 'HbaycY0VuZk', 'dxsPkcG-Q30', 'ZcFzcd4ZoMg', 'yUqNp-poh9M', 'yoDMh8FlHR8', '167521', 'kbRtSmJM5aU', 'skRqBxLLJkE', '100178', '-ri04Z7vwnc', 'mVnqP-vLpuo', 'B9oeT0K92wU', '_OmWVs0-6UM', 'DebbnL-_rnI', 'GlShH_ua_GU', 'Jz1oMq6l-ZM', 'L-a4Sh6iAcw', 'LDKWr94J0wM', 'aa0J1AXSseY', '173CpFb3clw', '202826', 'Wmhif6hmPTQ', '283935', 'naZi9AusrW4', 'wO8fUOC4OSE', '_1nvuNk7EFY', 'PHZIx22aFhU', 'ex-dKshlXoY', '6uoM8-9d0zA', 'ahG9c_uaf8s', 'vR90Pdx9wxs']
        cache_file = "./cache/cmumosei_test.pkl"
        if os.path.isfile(cache_file):
            print("Tensors {} already exist. Loading from cache directory".format(cache_file))
            with open(cache_file, "rb") as input_file:
                test = pickle.load(input_file)
        else:
            print("Tensors {} do not exist in cache directory".format(cache_file))

        # Stitch segments by video key

        data = []
        segments = [v[2] for v in test]

        for key in test_split:
            sentence = None
            acoustic = None
            visual = None
            label = None
            seg_keys = [v for v in segments if key in v]

            if len(seg_keys) > 0:
                for skey in seg_keys:
                    segment = self._get_segment_by_key(skey, test)
                    if segment is not None:
                        if sentence is None:
                            label = segment[1]
                            sentence = segment[0][0]
                            visual = segment[0][1]
                            acoustic = segment[0][2]
                        else:
                            sentence = np.append(sentence, segment[0][0], axis=0)
                            visual = np.append(visual, segment[0][1], axis=0)
                            acoustic = np.append(acoustic, segment[0][2], axis=0)

            if sentence is not None:
                data.append(((sentence, visual, acoustic), label, key))

        return data

    def _get_segment_by_key(self, skey, data):
        for d in data:
            if skey in d[2]:
                return d
        return None

    def _get_loaders(self, test, frames_per_sample=FRAME_COUNT):
        test_loaders = []
        seg_videos = []

        for i in range(len(test)):
            seg_vid = self._segment_vid(test[i], frames_per_sample)
            n_segs = len(seg_vid)
            seg_videos.append(seg_vid)
            test_loader = DataLoader(seg_vid, shuffle=False, batch_size=n_segs, collate_fn=multi_collate)
            test_loaders.append(test_loader)
        return test_loaders, seg_videos

    def _segment_vid(self, vid, frames_per_sample):
        length = vid[0][0].shape[0]
        new_length = length - (length % frames_per_sample)
        ms = []
        seg_vid = []

        for m in range(3):
            modality = vid[0][m]
            modality_dim = vid[0][m].shape[1]
            new_modality = list(modality[:new_length, :].reshape((int(new_length / frames_per_sample) , frames_per_sample, modality_dim)))
            if new_length < length:
                new_modality.append(modality[new_length:length, :])
            ms.append(new_modality)

        ms = list(zip(ms[0], ms[1], ms[2]))
        for m in ms:
            seg = (m, vid[1], vid[2])
            seg_vid.append(seg)
        return seg_vid

    def _compute_ground_truth(self, test_loaders):
        y_preds = []
        y_true = []
        self.gt_model.eval()
        with torch.no_grad():
            for test_loader in test_loaders:
                y_pred = []
                for batch in test_loader:
                    self.gt_model.zero_grad()
                    t, v, a, y, l = batch
                    if CUDA:
                        t = t.cuda()
                        v = v.cuda()
                        a = a.cuda()
                        y = y.cuda()
                        l = l.cuda()
                    y_tilde = self.gt_model(t, v, a, l)
                    y_tilde = y_tilde.detach().cpu().numpy()
                    y_pred.append(y_tilde)
                y_preds.append(y_pred)
        return y_preds

    def update_data(self, n_skip):
        updated_videos = []
        for i in range(len(self.y_preds)):
            prev_frame = None
            c = 0
            for j in range(len(self.seg_videos[i])):
                if c % (n_skip+1) == 0:
                    prev_frame = self.seg_videos[i][j][0]
                updated_seg = (prev_frame, self.y_preds[i][0][j].reshape((1,-1)), self.seg_videos[i][j][2])
                updated_videos.append(updated_seg)
                c += 1
        updated_videos = [v for v in updated_videos if isinstance(v[2], str)]
        return updated_videos

    def skip_data(self, n_skip):
        updated_videos = []
        for i in range(len(self.y_preds)):
            prev_frame = None
            c = 0
            for j in range(len(self.seg_videos[i])):
                if c % (n_skip+1) == 0:
                    updated_seg = (self.seg_videos[i][j][0], self.y_preds[i][0][j].reshape((1,-1)), self.seg_videos[i][j][2])
                    updated_videos.append(updated_seg)
                c += 1
        updated_videos = [v for v in updated_videos if isinstance(v[2], str)]
        return updated_videos

def get_pred_loss(model, loader, gpu=False):
    y_pred = []
    y_true = []
    criterion = nn.L1Loss(reduction='sum')
    pred_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            #model.zero_grad()
            t, v, a, y, l = batch
            if gpu:
                t = t.cuda()
                v = v.cuda()
                a = a.cuda()
                y = y.cuda()
                l = l.cuda()
            y_tilde = model(t, v, a, l)
            loss = criterion(y_tilde, y)
            pred_loss += loss.item()
            y_tilde = y_tilde.detach().cpu().numpy()
            y_pred.append(y_tilde)
            y_true.append(y.detach().cpu().numpy())
    
    return pred_loss

def get_pred_runtime(model, loader, gpu=False, rounds=10):
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            t, v, a, y, l = batch
            if gpu:
                t = t.cuda()
                v = v.cuda()
                a = a.cuda()
                y = y.cuda()
                l = l.cuda()
            for s in range(rounds):
                y_tilde = model(t, v, a, l)

class SocialWorkload(Workload):

    def __init__(self):
        # knobs exposed to KnobTuner
        #self.dataloader = get_StreamDataloader()
        self.dataloader = StreamDataLoader()
        self.CUDA = False
        self.knob_names = ["models", "num_skip_segs", "num_sample_frames"]
        self.knob_domains = [["regular", "small", "mini"], list(np.arange(6)), list(np.arange(5, FRAME_COUNT+1, 5))]
        self.knobs = ("regular", 0, 30)
        self.model = None
        
        # Load models from ./cache
        with open('./cache/configs.json', 'r') as f:
            self.configs = json.load(f)
        
        self.model_mini = make_model(hidden=self.configs["mini"]["hidden"], fc=self.configs["mini"]["fc"], dropout=self.configs["mini"]["dropout"])
        self.model_small = make_model(hidden=self.configs["small"]["hidden"], fc=self.configs["small"]["fc"], dropout=self.configs["small"]["dropout"])
        self.model_regular = make_model(hidden=self.configs["regular"]["hidden"], fc=self.configs["regular"]["fc"], dropout=self.configs["regular"]["dropout"])
        
        if self.CUDA:
            self.model_mini = self.model_mini.cuda()
            self.model_small = self.model_small.cuda()
            self.model_regular = self.model_regular.cuda()
        
        self.model_mini.load_state_dict(torch.load(self.configs["mini"]["path"]))
        self.model_small.load_state_dict(torch.load(self.configs["small"]["path"]))
        self.model_regular.load_state_dict(torch.load(self.configs["regular"]["path"]))
        
        assert len(self.knob_names) == len(self.knobs) == len(self.knob_domains)
        assert len(self.knob_domains[0]) == len(self.configs)

        # TODO


    def get_taskgraph(self, num_frames=150):
        graph = TaskGraph()

        for i in range(num_frames):
            # insert mask detection tasks

            # insert social distance tasks
            break
        return graph
    
    def set_knobs(self, k, verbose=True):
        for i in range(len(k)):
            assert k[i] in self.knob_domains[i]
        if verbose:
            print(f"Setting knobs from {self.knobs} to {k}")
        self.knobs = k

    def process(self):
        #out_file = "debug.csv" #"{}-{}-{}.csv".format(video_path.split("/")[-1].split(".")[0],
        #    self.knobs[0], self.knobs[1])
        #f = open(out_file, "a+")
        
        # Switch 
        m = self.knobs[0]
        if m == "mini":
            self.model = self.model_mini
        elif m == "small":
            self.model = self.model_small
        elif m == "regular":
            self.model = self.model_regular
        else:
            raise("Cannot find the corresponding model!")
        
        # Load and process data
        # Skip segments
        n_skips, n_frames = self.knobs[1], self.knobs[2]
        videos_quality = self.dataloader.update_data(n_skips)
        videos_runtime = self.dataloader.skip_data(n_skips)
        # Sample frames
        quality_loader = DataLoader(videos_quality, shuffle=False, batch_size=168, collate_fn=lambda b: multi_collate_sample(b, n_frames))
        runtime_loader = DataLoader(videos_runtime, shuffle=False, batch_size=168, collate_fn=lambda b: multi_collate_sample(b, n_frames))

        # Start processing
        # Measure quality
        pred_loss = get_pred_loss(self.model, quality_loader, gpu=self.CUDA)
        # Measure runtime
        start_time = time.time()
        get_pred_runtime(self.model, runtime_loader, gpu=self.CUDA)
        runtime = time.time() - start_time

        #f.close()

        print("Runtime in second", runtime)
        
if __name__ == "__main__":
    sw = SocialWorkload()
    sw.process()
    sw.set_knobs(("mini", 2, 10))
    sw.process()