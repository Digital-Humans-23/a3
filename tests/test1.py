import os, sys, glob
sys.path.append(os.path.join(os.getcwd(), 'lisst'))
sys.path.append(os.getcwd())

import numpy as np
import pytest


def test():
    """load the pseudo gt data"""
    data_pgt = np.load('data/CoreView_313_test.pkl', allow_pickle=True)

    """load the result"""
    data = np.load('results/mocap_zju_a3/CoreView_313.pkl', allow_pickle=True)


    """compute the metrics"""
    # joint locations
    J_locs_3d_pgt = data_pgt['J_locs_3d']
    J_locs_3d = data['J_locs_3d']
    err_locs_3d = np.mean(np.linalg.norm(J_locs_3d_pgt-J_locs_3d,ord=1,axis=-1))

    # joint rotations comparison
    # http://www.boris-belousov.net/2016/12/01/quat-dist/#:~:text=The%20distance%20between%20rotations%20represented%20by%20rotation%20matrices%20P%20and,matrix%20R%20%3D%20P%20Q%20%E2%88%97%20.
    J_rotmat_pgt = data_pgt['J_rotmat'].reshape([-1,3,3]) #[t*J,3,3]
    J_rotmat = data['J_rotmat'].reshape([-1,3,3]).transpose([0,2,1])
    err = np.arccos((np.trace(np.matmul(J_rotmat, J_rotmat_pgt),axis1=1,axis2=2)-1)/2.0)
    err_angles = np.mean(err) * 180 / np.pi

    assert err_locs_3d <=0.25 and err_angles<=30

