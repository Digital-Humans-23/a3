import os, sys, glob
sys.path.append(os.path.join(os.getcwd(), 'lisst'))
sys.path.append(os.getcwd())

import numpy as np



def test_img_project_loss():
    """this function is to test the implementation of `img_project_loss`.
    """
    
    # load input and ground truth
    testdata = np.load('data/test_img_project_loss_data.pkl',allow_pickle=True)
    J_rec = testdata['J_rec']
    J_rec_proj_gt = testdata['J_rec_proj']
    
    # load result
    J_rec_proj = np.load('data/test_img_project_loss_data2.npy')
    assert np.mean(np.abs(J_rec_proj-J_rec_proj_gt)) < 0.005



