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
# ffmpeg -framerate 25 -pattern_type glob -i '3d/*.png' -c:v libx264 -pix_fmt yuv420p 3d_0.mp4




def visualize3d(data, outfile_path=None, datatype='kps'):
    ## prepare data
    jts = data['J_locs_3d'] # array of [t, J, 3]
    jts_rotmat = data['J_rotmat'] #[t,J,3,3]
    n_frames, n_jts = jts.shape[:2]
    
    # open3d.geometry.create_mesh_box(width=1.0, height=1.0, depth=1.0)

    ## prepare visualizer
    np.random.seed(0)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=960, height=540,visible=True)
    # vis.create_window(width=480, height=270,visible=True)
    render_opt=vis.get_render_option()
    render_opt.mesh_show_back_face=True
    render_opt.line_width=10
    render_opt.point_size=5
    render_opt.background_color = color_hex2rgb('#1c2434')
    vis.update_renderer()

    ### top lighting
    box = o3d.geometry.TriangleMesh.create_box(width=200, depth=1,height=200)
    box.translate(np.array([-200,-200,6]))
    vis.add_geometry(box)
    vis.poll_events()
    vis.update_renderer()

    #### world coordinate
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
    vis.add_geometry(coord)
    vis.poll_events()
    vis.update_renderer()

    ## create body mesh from data
    ball_list = []
    for i in range(n_jts):
        # ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        ball = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
        vis.add_geometry(ball)
        vis.poll_events()
        vis.update_renderer()
        ball_list.append(ball)

    
    frame_idx = 0
    cv2.namedWindow('frame2')
    for it in range(0,n_frames):
        for i,b in enumerate(ball_list):
            # b.translate(jts[it,i], relative=False)
            b.rotate(jts_rotmat[it,i]).translate(jts[it,i], relative=False)
            vis.update_geometry(b)

        ## update camera.
        ctr = vis.get_view_control()
        ctr.set_constant_z_far(10)
        cam_param = ctr.convert_to_pinhole_camera_parameters()
        ### get cam T
        # body_t = np.mean(data[it],axis=0) # let cam follow the body
        body_t = np.array([0,0,0])
        cam_t = body_t + 2.5*np.ones(3)
        ### get cam R
        cam_z =  body_t - cam_t
        cam_z = cam_z / np.linalg.norm(cam_z)
        cam_x = np.array([cam_z[1], -cam_z[0], 0.0])
        cam_x = cam_x / np.linalg.norm(cam_x)
        cam_y = np.array([cam_z[0], cam_z[1], -(cam_z[0]**2 + cam_z[1]**2)/cam_z[2] ])
        cam_y = cam_y / np.linalg.norm(cam_y)
        cam_r = np.stack([cam_x, -cam_y, cam_z], axis=1)
        ### update render cam
        transf = np.eye(4)
        transf[:3,:3]=cam_r
        transf[:3,-1] = cam_t
        cam_param = update_render_cam(cam_param, transf)
        ctr.convert_from_pinhole_camera_parameters(cam_param)
        vis.poll_events()
        vis.update_renderer()

        ## capture RGB appearance
        rgb = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        cv2.imshow("frame2", np.uint8(255*rgb[:,:,[2,1,0]]))
        if outfile_path is not None:
            renderimgname = os.path.join(outfile_path, 'img_{:05d}.png'.format(frame_idx))
            frame_idx = frame_idx + 1
            cv2.imwrite(renderimgname, np.uint8(255*rgb[:,:,[2,1,0]]))
        cv2.waitKey(5)

        # rotate them back
        for i,b in enumerate(ball_list):
            b.rotate(jts_rotmat[it,i].T)
            vis.update_geometry(b)




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
    parser.add_argument('--folder', default='mocap_zju', 
                        help='the folder containing the results')
    parser.add_argument('--mode', default='3d', required=True)
    args = parser.parse_args()

    mode = args.mode
    if mode not in ['3d', '2d']:
        raise ValueError('mode should be either `2d` or `3d`')

    proj_path = os.getcwd()
    
    exp = args.exp
    folder = args.folder
    res_file_name = proj_path+'/results/{}/{}.pkl'.format(folder, exp)
    data = np.load(res_file_name, allow_pickle=True)
    
    if mode == '3d':
        renderfolder3d = '/tmp/render_cmu/{}/{}/3d'.format(folder, exp)
        # render 3D motion
        
        if not os.path.exists(renderfolder3d):
            os.makedirs(renderfolder3d)
        visualize3d(data,
                outfile_path=renderfolder3d, datatype='kps')
    elif mode == '2d':
    ## render 2D motion
        idx_view_to_render = 0
        renderfolder2d = '/tmp/render_cmu/{}/{}/2d_{}'.format(folder, exp, idx_view_to_render)

        if not os.path.exists(renderfolder2d):
            os.makedirs(renderfolder2d)
        
        zju_data = '/mnt/hdd/datasets/ZJUMocap/{}/'.format(exp)
        cam_file = os.path.join(zju_data, 'cam_params.json')
    
        with open(cam_file) as f:
            cam_params = json.load(f)
        cam_names = cam_params['all_cam_names']
        name_view_to_render = cam_names[idx_view_to_render]
        frame_canvas = os.path.join(zju_data, name_view_to_render+'_mediapipe/frames')
        joints_locs_2d = data['J_locs_2d']
        visualize2d(joints_locs_2d, renderfolder2d,
                    # 1,
                    idx_view_to_render, 
                    frame_canvas)

