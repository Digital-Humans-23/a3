from __future__ import annotations

import os, sys, glob
sys.path.append(os.path.join(os.getcwd(), 'lisst'))
sys.path.append(os.getcwd())

import time
import numpy as np
import torch
import argparse
import pickle
import json
from typing import Union
from tqdm import tqdm

from torch import optim
from torch.nn import functional as F

from lisst.utils.config_creator import ConfigCreator
from lisst.utils.joint_matching import LISST_TO_MEDIAPOSE
from lisst.models.baseops import (RotConverter, get_scheduler)
from lisst.models.body import LISSTPoser, LISSTCore







class LISSTRecOP():
    """operation to perform motion capture from multi-view cameras in ZJUMocap
        
        - the cameras are well calibrated.
        
        - mediapose has been applied applied to perform 2D keypoint estimation.
        
        - only LISST shape and pose priors are used. No advanced motion prior.
    
    """
    
    
    def __init__(self, shapeconfig, poseconfig, testconfig):
        self.dtype = torch.float32
        gpu_index = testconfig['gpu_index']
        if gpu_index >= 0:
            self.device = torch.device('cuda',
                    index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device('cpu')
        self.shapeconfig = shapeconfig
        self.poseconfig = poseconfig
        self.testconfig = testconfig

    def build_model(self):
        # load shape model
        self.shaper = LISSTCore(self.shapeconfig)
        self.shaper.eval()
        self.shaper.to(self.device)
        self.nj = self.shaper.num_kpts
        self.shaper.load(self.testconfig['shaper_ckpt_path'])
        
        # load pose model
        self.poser = LISSTPoser(self.poseconfig)
        self.poser.eval()
        self.poser.to(self.device)
        self.poser.load(self.testconfig['poser_ckpt_path'])

        self.weight_sprior = self.testconfig['weight_sprior']
        self.weight_pprior = self.testconfig['weight_pprior']
        self.weight_smoothness = self.testconfig['weight_smoothness']


    def _cont2rotmat(self, rotcont):
        '''local process from continuous rotation to rotation matrix
        
        Args:
            - rotcont: [t,b,J,6]

        Returns:
            - rotmat: [t,b,J,3,3]
        '''
        nt, nb, nj = rotcont.shape[:-1]
        rotcont = rotcont.contiguous().view(nt*nb*nj, -1)
        rotmat = RotConverter.cont2rotmat(rotcont).view(nt, nb, nj,3,3)
        return rotmat

    def _rotmat2cont(self, rotmat):
        '''local process from rotatiom matrix to continuous rotation
        
        Args:
            - rotmat: [t,b,J,3,3]

        Returns:
            - rotcont: [t,b,J,6]
        '''
        nt,nb,nj = rotmat.shape[:3]
        rotcont = rotmat[:,:,:,:,:-1].contiguous().view(nt,nb,nj,-1)
        return rotcont


    def _add_additional_joints(self, J_rotcont):
        """local implementation of add nose or heels to the model

        Args:
            J_rotcont (torch.Tensor): the poser output without new joints.
        """
        nt, nb = J_rotcont.shape[:2]
        out = self.poser.add_additional_bones(J_rotcont.contiguous().view(nt*nb, -1, 6), 
                        self.shaper.joint_names,
                        self.shaper.get_new_joints())
        
        return out.contiguous().view(nt, nb, -1, 6)


    def fk(self, 
            r_locs: torch.Tensor, 
            J_rotcont: torch.Tensor, 
            bone_length: torch.Tensor, 
            transf_rotcont: Union[None, torch.Tensor]=None,
            transf_transl: Union[None, torch.Tensor]=None) -> torch.Tensor:
        '''forward kinematics.
        The predicted joint locations are discarded and others are preserved for FK.

        Args:
            - r_locs, [t,b,1,3]. The root locations
            - J_rotcont, [t,b,J,6]. The joint rotations
            - bone_length [b,d]. The provided bone length.
            - transf_rotcont, [t,b,1,6]. transfrom from canonical to world
            - transf_transl, [t,b,1,3]. transfrom from canonical to world
        
        Returns:
            - Y_rec_new: [t,b,J,9] The projected bone transform
            - J_locs_fk: [t,b,J,3]. The joint locations via forward kinematics.
        
        '''

        nt, nb = r_locs.shape[:2]
        rotmat = self._cont2rotmat(J_rotcont)
        if transf_rotcont is not None:
            transf_rotmat = self._cont2rotmat(transf_rotcont)
            rotmat = torch.einsum('tbpij,tbpjk->tbpik', transf_rotmat, rotmat)
        if transf_transl is not None:
            r_locs = r_locs + transf_transl
        bone_length = bone_length.unsqueeze(0).repeat(nt, 1,1)
        J_locs_fk = self.shaper.forward_kinematics(r_locs.reshape(nt*nb, 1, 3), 
                                        bone_length.reshape(nt*nb, -1), 
                                        rotmat.reshape(nt*nb, self.nj, 3,3))
        J_locs_fk = J_locs_fk.reshape(nt, nb, self.nj, 3)
        
        
        return J_locs_fk, rotmat




    def img_project(self,
                    J_rec: torch.Tensor, #[t,b,p,3], the last dimension denotes the 3D location
                    cam_rotmat: torch.Tensor, #6D camera rotation w.r.t. origin of the first mp
                    cam_transl: torch.Tensor, # cam translation w.r.t. origin of the first mp,
                    cam_K: torch.Tensor, 
        ):
        # =======================================
        ### TODO!!! Ex.1: implement here
        # Hints:
        # - J_rec is the 3D joint locations in the world coordinate system. 
        # - Based on the camera extrinsics, convert J_rec to J_rec_cam, which has the joint locations in individual camera coordinate systems.
        # - Based on the camera intrinsics, convert J_rec_cam to J_rec_proj, which has the 2D joint locations in the image planes. 
        # - if encountering tensor shape misalignment, you could print these tensor shapes for debugging.
        # - please use our file `data/test_img_project_loss_data.pkl` to test the keypoint detection. 
        J_rec_proj = None
        # =======================================
       
        
        return J_rec_proj




    def generate_img_project_test_file(self, cam_rotmat, cam_transl, cam_K):
        """this function is to test the implementation of `img_project_loss`.
        """
        
        # load data in np
        testdata = np.load('data/test_img_project_loss_data.pkl',allow_pickle=True)
        J_rec = torch.tensor(testdata['J_rec']).float().to(self.device)
        J_rec_proj_gt = torch.tensor(testdata['J_rec_proj']).float().to(self.device)
        J_rec_proj = self.img_project(J_rec, cam_rotmat, cam_transl, cam_K).detach().cpu().numpy()
        np.save('data/test_img_project_loss_data2.npy',J_rec_proj)
        # print('test file saved!')
            
        


    def img_project_loss(self, 
            obs: torch.Tensor, #[t,b,p,3], the last dimension=[x,y,vis]
            J_rec: torch.Tensor, #[t,b,p,3], the last dimension denotes the 3D location
            cam_rotmat: torch.Tensor, #6D camera rotation w.r.t. origin of the first mp
            cam_transl: torch.Tensor, # cam translation w.r.t. origin of the first mp,
            cam_K: torch.Tensor, 
        )->torch.TensorType:
        ''' reprojection loss to multi-view images
        Args:
            - obs: the 2D keypoint detections. [t,b,J,3]. b denotes the camera views. The last dimension = [x,y,vis]
            - J_rec: the 3D joint locations. [t,b=1,J,3]. b=1 if only one person one sequence.
            - cam_rotmat: camera rotation matrix from world to cam. [b,3,3]. b denotes the cam views.
            - cam_transl: camera translation from world to cam. [b,1,3] b denotes the cam views.
            - cam_K: the cam intrinsics. [b,3,3] 
        '''

        # project keypoints and save the file for testing.
        J_rec_proj = self.img_project(J_rec, cam_rotmat, cam_transl, cam_K)
        self.generate_img_project_test_file(cam_rotmat, cam_transl, cam_K)
        
        #calculate the loss
        loss = obs[:,:,:,-1:]*(obs[:,:,:,:-1]-J_rec_proj).abs()
        
        return torch.mean(loss)



    def recover(self,
                motion_obs: torch.Tensor,
                cam_rotmat: torch.Tensor,
                cam_transl: torch.Tensor, 
                cam_K: torch.Tensor,
                lr: float = 0.0003,
                n_iter: int = 500,
                to_numpy: bool=True
        ):
        """recover motion primitives based on the body observations.
        We assume the observation (motion, bparams) is a time sequence of smplx bodies
        torch.Tensor and np.ndarray are supported

        Args:
            - motion_obs: the sequence of undistorted 2D pose detections, with the shape [t,b,J,3]. Used as observation
            - cam_rotmat: the camera rotation matrices from world to cam, [b,3,3]. b denoting the views
            - cam_transl: the camera translations from world to cam, [b,1,3]
            - cam_K: the camera intrinsics, [b,3,3]
            - lr: the learning rate of the optimizer (Adam)
            - n_iter: number of iterations for the inner loop
            - to_numpy: produce numpy if true
            
        Returns:
            Y_rec, r_locs, J_rotmat, bone_length, J_locs_3d, J_locs_2d
            
            - Y_rec: the estimated bone transforms [t,J,9], 3D transl + 6D rotcont
            - r_locs: the 3D joint locations [t,1,3], in world coordinate
            - J_rotmat_rec: the rotation matrics [t,J,3,3] in world coordinate
            - bone_length: the bone length [31]
            - J_locs_3d: [t,J,3] in world coordinate
            - J_locs_2d: [t,b,J,2] in the camera view. b denotes the camera view.
        Raises:
            None
        """
        
        #obtain the 2D joint locations corresponding to the CMU joints.
        traj_idx = []
        traj = []
        for key, val in LISST_TO_MEDIAPOSE.items():
            if len(val) == 0:
                continue
            else:
                traj_idx.append(self.shaper.joint_names.index(key))
                traj.append(motion_obs[:,:,val].mean(dim=-2, keepdim=True))
        traj = torch.cat(traj, dim=-2)
        nt, nb = traj.shape[:2]
    
        #-------setup latent variables to optimize
        #- r_locs: the 3D root translations at all frames about the world coordinate. Note the first joint is the root/pelvis.
        #- J_rotlatent: the joint rotations in the LISSTPoser latent space, about the canonical coordinate.
        #- transf_rotcont: at each frame, we transform the rotation from the canonical frame to the world frame.
        #- betas: the latent variable in the lisst shape space
        nj_cmu = 31
        r_locs = torch.zeros(nt,1,1,3).float().to(self.device)
        J_rotlatent = torch.zeros(nt, nj_cmu, self.poser.z_dim).to(self.device)
        transf_rotcont = torch.tensor([1,0,0,1,0,0]).float().repeat(nt,1,1,1).to(self.device)
        betas = torch.zeros(1, 12).to(self.device)
        
        r_locs.requires_grad=True 
        J_rotlatent.requires_grad=False
        transf_rotcont.requires_grad=True
        betas.requires_grad=False
        
        optimizer = optim.Adam([r_locs, J_rotlatent, transf_rotcont, betas], lr=lr)
        scheduler = get_scheduler(optimizer, policy='lambda',
                                    num_epochs_fix=0.25*n_iter,
                                    num_epochs=n_iter)

        #--------optimization main loop. 
        ## We set to body pose learnable after several iterations. So our method is in principal stage-wise.
        for jj in range(n_iter):
            # =======================================
            ### TODO!!! Ex.2: implement multistage optimization here
            # Q: Why multistage?
            # A: Inverse kinematics is a highly ill-posed problem. A good initialization is essential.
            # Hints:
            # - In early stages, we only optimize the body global parameters.
            # - In late stages, we optimize both the global and the local body parameters.
            # Multistages can be implemented by enabling/disabling updating certain variables.
            # =======================================
        
            
            ss = time.time()
            #yield global motion
            bone_length = self.shaper.decode(betas) #[b,]
            J_rotcont = self.poser.decode(J_rotlatent).contiguous().view(nt, 1, nj_cmu, -1)
            J_rotcont = self._add_additional_joints(J_rotcont)
            J_rec_fk, rotmat_rec_fk = self.fk(r_locs, J_rotcont, bone_length, 
                                        transf_rotcont=transf_rotcont, transf_transl=None)

            #smoothness regularization
            # =======================================
            ### TODO!!! Ex.3: implement a temporal smoothness loss
            # Q: Why temporal smoothness?
            # A: Human motion is smooth. Without this loss, obvious discontinuities are in the result. 
            # Hints:
            # - minimize the l1/l2 norm of the joint location velocity
            # - minimize the l1/l2 norm of the joint rotation velocity
            loss_smoothness = torch.zeros(1).float().to(self.device)
            # =======================================
            
            #shape regularization, encouraging to produce mean shape.
            loss_sprior = self.weight_sprior * torch.mean(betas**2)
            
            #pose regularization, encouraging to produce mean pose.
            loss_pprior = self.weight_pprior * torch.mean(J_rotlatent**2)
            

            '''image reprojection loss'''
            loss_rec = self.img_project_loss(traj, J_rec_fk[:,:,traj_idx], 
                                    cam_rotmat, cam_transl, cam_K)
                        
            # print(loss_rec.item())
            loss = loss_sprior + loss_pprior + loss_rec + loss_smoothness
            '''optimizer'''
            ss = time.time()
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()
            if jj % 200==0 or jj==n_iter-1:
                print('[iter_inner={:2d}] PROJ={:.3f}, SPRIOR={:.3f}, PPRIOR={:.3f}, SMOOTH={:.3f}, TIME={:.2f}'.format(
                        jj, loss_rec.item(), loss_sprior.item(), loss_pprior.item(), loss_smoothness.item(),
                        time.time()-ss))
            scheduler.step()
        
        '''output the final results'''
        bone_length = self.shaper.decode(betas) #[b,]
        J_rotcont = self.poser.decode(J_rotlatent)
        J_rotcont = J_rotcont.contiguous().view(nt, 1, nj_cmu, -1) # in canonical frame
        J_rotcont = self._add_additional_joints(J_rotcont)
        J_rec_fk, rotmat_rec_fk = self.fk(r_locs, J_rotcont, bone_length, 
                                    transf_rotcont=transf_rotcont, transf_transl=None)

        #reproject to 2D
        J_locs_2d = self.img_project(J_rec_fk, cam_rotmat, cam_transl, cam_K)
        # J_rec_cam = torch.einsum('bij,tbpj->tbpi', cam_rotmat, J_rec_fk) + cam_transl.unsqueeze(0)
        
        # #project the 3D joints to the image plane
        # J_rec_proj_unorm = torch.einsum('bij, tbpj->tbpi', cam_K, J_rec_cam)
        # J_rec_proj_x = J_rec_proj_unorm[:,:,:,0]/J_rec_proj_unorm[:,:,:,2]
        # J_rec_proj_y = J_rec_proj_unorm[:,:,:,1]/J_rec_proj_unorm[:,:,:,2]
        # J_locs_2d = torch.stack([J_rec_proj_x, J_rec_proj_y],dim=-1)

        #change body features to body parameters
        r_locs = J_rec_fk[:,:,:1] # the generated root translation
        J_rotmat = rotmat_rec_fk # the genrated joint rotation matrices

        if to_numpy:
            r_locs = r_locs[:,0].detach().cpu().numpy() #[t, 1, 3]
            J_rotmat = J_rotmat[:,0].detach().cpu().numpy() #[ t,J, 3,3]
            bone_length = bone_length[0].detach().cpu().numpy() #[J]
            J_locs_3d = J_rec_fk[:,0].detach().cpu().numpy() #[t,J,3]
            J_locs_2d = J_locs_2d.detach().cpu().numpy()
        
        return r_locs, J_rotmat, bone_length, J_locs_3d, J_locs_2d





    def mocap_for_zju(self, cam_ids: list = None, 
                    data_path: str='/mnt/hdd/datasets/ZJUMocap/CoreView_313/'):
        '''
        - data_path: the batch generator
        '''
        # output placeholder
        rec_results = {
            'r_locs': None, 
            'J_rotmat': None, 
            'J_shape': None, 
            'J_locs_3d': None, 
            'J_locs_2d': None
        }
        data_all = np.load(os.path.join(data_path, 'mediapipe_all.pkl'), allow_pickle=True)
        if cam_ids == None:
            motion_obs = data_all['motion_obs']
            cam_rotmat = data_all['cam_rotmat']
            cam_transl = data_all['cam_transl']
            cam_K = data_all['cam_K']
        else:
            motion_obs = data_all['motion_obs'][:,cam_ids] # only 2D keypoints are given
            cam_rotmat = data_all['cam_rotmat'][cam_ids]
            cam_transl = data_all['cam_transl'][cam_ids]
            cam_K = data_all['cam_K'][cam_ids]

        motion_obs = torch.tensor(motion_obs).float().to(self.device)
        cam_rotmat = torch.tensor(cam_rotmat).float().to(self.device)
        cam_transl = torch.tensor(cam_transl).float().to(self.device)
        cam_K = torch.tensor(cam_K).float().to(self.device)
        print('-- input data prepared..')
        
        # optimization
        ss = time.time()
        results= self.recover(
                            motion_obs,
                            cam_rotmat,
                            cam_transl, 
                            cam_K,
                            lr = self.testconfig['lr'],
                            n_iter = self.testconfig['n_iter'],
                            to_numpy=True
                    )
        eps = time.time()-ss
        print('-- takes {:03f} seconds'.format(eps))
        for idx, key in enumerate(rec_results.keys()):
            rec_results[key] = results[idx]

        ### save to file
        outfilename = os.path.join(
                            'results',
                            'mocap_zju_a3'
                        )
        if not os.path.exists(outfilename):
            os.makedirs(outfilename)
        subjname = os.path.basename(data_path)
        if subjname == '':
            subjname = os.path.basename(data_path[:-1])
        outfilename = os.path.join(outfilename,
                        '{}.pkl'.format(subjname)
                        )
        with open(outfilename, 'wb') as f:
            pickle.dump(rec_results, f)









if __name__ == '__main__':
    """ example command
    python scripts/app_multiview_mocap_ZJUMocap.py --cfg_shaper=LISST_SHAPER_v2 --cfg_poser=LISST_POSER_v0 --data_path=data/CoreView_313
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_shaper', default=None, required=True)
    parser.add_argument('--cfg_poser', default=None, required=True)
    parser.add_argument('--data_path', default=None, required=True,
                        help="specify the datapath")
    parser.add_argument('--gpu_index', type=int, default=0)
    args = parser.parse_args()

    """setup"""
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_printoptions(sci_mode=False)
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    
    cfgall_shaper = ConfigCreator(args.cfg_shaper)
    modelcfg_shaper = cfgall_shaper.modelconfig
    losscfg_shaper = cfgall_shaper.lossconfig
    traincfg_shaper = cfgall_shaper.trainconfig
    
    cfgall_poser = ConfigCreator(args.cfg_poser)
    modelcfg_poser = cfgall_poser.modelconfig
    losscfg_poser = cfgall_poser.lossconfig
    traincfg_poser = cfgall_poser.trainconfig
    
    testcfg = {}
    testcfg['gpu_index'] = args.gpu_index
    testcfg['shaper_ckpt_path'] = os.path.join(traincfg_shaper['save_dir'], 'epoch-000.ckp')
    testcfg['poser_ckpt_path'] = os.path.join(traincfg_poser['save_dir'], 'epoch-500.ckp')
    testcfg['result_dir'] = cfgall_shaper.cfg_result_dir
    testcfg['seed'] = 0
    testcfg['lr'] = 0.1
    testcfg['n_iter'] = 2000
    testcfg['weight_sprior'] = 0.0
    testcfg['weight_pprior'] = 0.05
    testcfg['weight_smoothness'] = 100
    
    """model and testop"""
    testop = LISSTRecOP(shapeconfig=modelcfg_shaper, poseconfig=modelcfg_poser, testconfig=testcfg)
    # testop.gather_mediapipe_data_for_zju(data_path=args.data_path)
    testop.build_model()
    testop.mocap_for_zju(cam_ids=[0,6,12,18],data_path=args.data_path) # from test views
    
    


