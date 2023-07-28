# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from absl import flags, app
import sys
sys.path.insert(0,'third_party')
import numpy as np
import torch
import os
import glob
import cv2
import trimesh
from scipy.spatial.transform import Rotation as R
import imageio

from utils.io import save_vid, str_to_frame, save_bones
from utils.colors import label_colormap
from nnutils.train_utils import v2s_trainer
import matplotlib
from tqdm import tqdm
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")

opts = flags.FLAGS
                

# TODO : color palettte

def palette():
    return ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

def save_code(trainer):
    pose_code_net = trainer.model.pose_code
    root_code_net = trainer.model.nerf_root_rts.root_code

    print("len(trainer.evalloader) : ", len(trainer.evalloader))

    idx_render = range(len(trainer.evalloader))
    video_idx = 0
    for idx, _ in tqdm(enumerate(idx_render)):
        batch = [trainer.evalloader.dataset[idx]]
        batch = trainer.evalloader.collate_fn(batch)
        pose_code = pose_code_net(batch['frameid'][0].to("cuda:0"))[0].cpu()
        root_code = root_code_net(batch['frameid'][0].to("cuda:0"))[0].cpu()
        if idx >= pose_code_net.vid_offset[video_idx + 1]:
            video_idx += 1
        torch.save(pose_code, f'save_code/vid{video_idx}_pose_frame{idx - pose_code_net.vid_offset[video_idx]}.pt')
        torch.save(root_code, f'save_code/vid{video_idx}_root_frame{idx - pose_code_net.vid_offset[video_idx]}.pt')

    

    pass

def analysis_code(trainer, loss_txt:str):

    # parsing loss
    loss_graph = []
    with open(loss_txt, 'r') as f:
        loss_file = f.readlines()
        for line in loss_file:
            if line[:3] == 'cd:':
                loss_graph.append(float(line[3:7]))

    # pose code
    offset = trainer.model.pose_code.vid_offset
    path = 'save_code'
    many_video_pose_codes = [[] for _ in range(len(offset)-1)]
    many_video_root_codes = [[] for _ in range(len(offset)-1)]
    for i in range(len(many_video_pose_codes)):
        many_video_pose_codes[i] = sorted(glob.glob(path + f'/vid{i}_pose_*'))
        many_video_root_codes[i] = sorted(glob.glob(path + f'/vid{i}_root_*'))

    # cosine similarity
    for idx, (video_pose_codes, video_root_codes) in tqdm(enumerate(zip(many_video_pose_codes, many_video_root_codes))):
        plt.clf()
        fig = plt.figure(figsize=(10, 14))
        _, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 6]})
        matrix = torch.zeros((len(video_pose_codes), 128))
        for i, frame_path in enumerate(video_pose_codes):
            code = torch.load(frame_path)
            matrix[i] = code

        val = torch.mm(matrix, matrix.T)
        denorm = torch.mm(matrix.norm(dim=1)[:, None], matrix.norm(dim=1)[:, None].T)
        result = val / denorm
        result = (1 - torch.eye(len(result))) * result
        # idx_max = torch.argmax(result, dim=1).numpy()
        idx_max = torch.topk(result, k=3, dim=1)[1].numpy()

        top1_idx = idx_max[:, 0]
        top2_idx = idx_max[:, 1]
        top3_idx = idx_max[:, 2]


        ax1.plot(top3_idx, label=f'video{idx}_top3', color='violet')
        ax1.plot(top2_idx, label=f'video{idx}_top2', color='blue')
        ax1.plot(top1_idx, label=f'video{idx}_top1', color='royalblue')
        ax1.plot(np.arange(len(idx_max)) + 2, label='+2 frame', color='red', linestyle='--')
        ax1.plot(np.arange(len(idx_max)) - 2, label='-2 frame', color='red', linestyle='--')
        plt.legend()
        ax0.plot(loss_graph, label='loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'save_code/analysis/pose_code_cosine_sim_{idx}.png')

        plt.clf()   
        fig = plt.figure(figsize=(10, 14))
        _, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 6]})
        matrix = torch.zeros((len(video_root_codes), 128))
        for i, frame_path in enumerate(video_root_codes):
            code = torch.load(frame_path)
            matrix[i] = code

        val = torch.mm(matrix, matrix.T)
        denorm = torch.mm(matrix.norm(dim=1)[:, None], matrix.norm(dim=1)[:, None].T)
        result = val / denorm
        result = (1 - torch.eye(len(result))) * result
        # idx_max = torch.argmax(result, dim=1).numpy()
        idx_max = torch.topk(result, k=3, dim=1)[1].numpy()

        top1_idx = idx_max[:, 0]
        top2_idx = idx_max[:, 1]
        top3_idx = idx_max[:, 2]

        

        ax1.plot(top3_idx, label=f'video{idx}_top3', color='green')
        ax1.plot(top2_idx, label=f'video{idx}_top2', color='olive')
        ax1.plot(top1_idx, label=f'video{idx}_top1', color='orange')
        
        ax1.plot(np.arange(len(idx_max)) + 2, label='+2 frame', color='red', linestyle='--')
        ax1.plot(np.arange(len(idx_max)) - 2, label='-2 frame', color='red', linestyle='--')

        plt.legend()
        ax0.plot(loss_graph, label='loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'save_code/analysis/root_code_cosine_sim_{idx}.png')
    # PCA
    




def main(_):

    ############## CUSTOM ###########

    opts.frame_chunk = 1
    opts.rnd_frame_chunk =1
    opts.render_size = 64
    #################################
    print(opts)
   

    trainer = v2s_trainer(opts, is_eval=True)
    data_info = trainer.init_dataset()    
    trainer.define_model(data_info)
    # print(data_info)
    
    ############## CUSTOM ###########
    trainer.model.img_size = 64
    loss_txt = opts.model_path.replace('params_latest.pth', 'metric.txt')

    # save_code(trainer)
    analysis_code(trainer, loss_txt)   


if __name__ == '__main__':
    app.run(main)
