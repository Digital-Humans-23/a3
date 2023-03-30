import os, sys, glob
sys.path.append(os.path.join(os.getcwd(), 'lisst'))
sys.path.append(os.path.join(os.getcwd(), 'scripts'))
sys.path.append(os.getcwd())
import torch

import numpy as np
from scripts.app_multiview_mocap_ZJUMocap import img_project




def test_img_project_loss(cam_rotmat, cam_transl, cam_K):
    """this function is to test the implementation of `img_project_loss`.
    """
    
    # load cameras in np
    cam_ids = [0,6,12,18]
    data_all = np.load(os.path.join('data/mediapipe_all.pkl'), allow_pickle=True)    
    cam_rotmat = data_all['cam_rotmat'][cam_ids]
    cam_transl = data_all['cam_transl'][cam_ids]
    cam_K = data_all['cam_K'][cam_ids]

    # load input and ground truth
    testdata = np.load('results/test_img_project_loss_data.pkl',allow_pickle=True)
    J_rec = testdata['J_rec']
    J_rec_proj_gt = testdata['J_rec_proj']
    
    # load result
    J_rec_proj = np.load('data/test_img_project_loss_data2.npy')
    assert F.l1_loss(J_rec_proj, J_rec_proj_gt) < 0.001



