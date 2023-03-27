import os
import sys
import numpy as np
import open3d as o3d
import torch
import cv2
import pickle
import pdb
import re
import glob
import json
import argparse

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'lisst'))

from lisst.utils.vislib import *


#----------- one can use ffmpeg later to compose the video, e.g.
# ffmpeg -framerate 25 -pattern_type glob -i '2d_0/*.png' -c:v libx264 -pix_fmt yuv420p 2d_0.mp4



def visualize2d(joints_locs_2d, outfile_path, idx_view_to_render, frame_canvas, ):
    video_frames = sorted(glob.glob(frame_canvas+'/*.png')) 
    J2d = joints_locs_2d[:,idx_view_to_render]
    nt, nj = J2d.shape[:2]
    
    # cv2.namedWindow('frame3')
    ## visualize 2D points
    for it in range(0,nt):
        vf = cv2.imread(video_frames[it])
        # vf = cv2.cvtColor(vf, cv2.COLOR_BGR2RGB)
        j2d_curr = J2d[it]
        for jj in range(j2d_curr.shape[-2]):
            if np.any(j2d_curr[jj]<=0):
                continue
            vf = cv2.circle(vf, np.uint(j2d_curr[jj]), 8, color=(69,74,196), thickness=3)
            
        renderimgname = os.path.join(outfile_path, 'img_{:05d}.png'.format(it))
        # cv2.imshow("frame3", vf)
        cv2.imwrite(renderimgname, vf)
        # cv2.waitKey(5)




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='CoreView_313', 
                        help='the subject in the original ZJUMocap')
    parser.add_argument('--folder', default='mocap_zju_a3', 
                        help='the folder containing the results')
    args = parser.parse_args()

    proj_path = os.getcwd()
    
    exp = args.exp
    folder = args.folder
    res_file_name = proj_path+'/results/{}/{}.pkl'.format(folder, exp)
    data = np.load(res_file_name, allow_pickle=True)
    
    
    idx_view_to_render = 0 # dont change this.
    renderfolder2d = 'results/{}/{}/2d_{}'.format(folder, exp, idx_view_to_render)

    if not os.path.exists(renderfolder2d):
        os.makedirs(renderfolder2d)
    
    frame_canvas = os.path.join('data/1_mediapipe/renders')
    joints_locs_2d = data['J_locs_2d']
    visualize2d(joints_locs_2d, renderfolder2d,
                # 1,
                idx_view_to_render, 
                frame_canvas)

