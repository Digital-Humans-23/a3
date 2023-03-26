# Assignment 3 - Data-driven inverse kinematics for human motion capture


Hand-in: 11 April 2023, 14:00 CET


----

Leave your name, student ID, ETH email address and the URL link to your motion capture result video here.

- Name:
- Student ID:
- ETH Email:
- URL:

----


In this assignment, we implement markerless human motion capture from multiview videos. 
Our human body is complex, so we leverage a pre-trained human body model, i.e. LISST, to provide body priors and effective regularizations.
Specifically, we provide a codebase, pre-trained models, and pre-processed data, and you implement the most essential parts.

The grading scheme is as follows:
- (50%) Ex 1: Implement the image projection loss, i.e. the data term for motion capture.
- (30%) Ex 2: Implement the multistage optimization.
- (20%) Ex 3: Implement the temporal smoothness term.


**IMPORTANT**
- Without the correct image projection loss, you cannot get valid body motion results, and hence cannot proceed Ex.2 and Ex.3.
Therefore, you will get **zero** point for this assignment if you cannot succeed in Ex 1. 
- Visualization is very important to verify the human motion realism. Therefore, you need to visualize your result based on the code `scripts/vis_ZJUMocap.py` w.r.t. a video, 
and share the video link above. Without this video, you will get **zero** point from this assignment. 
- If A suspected plagiarism case is detected, you will get **zero** point from this assignment.
- After running the code, your results will be stored as `results/mocap_zju_a3/CoreView_313.pkl`. This is automatically generated by `scripts/app_multiview_mocap_ZJUMocap.py`, and should have the following format
```motion_data = {
    'r_locs': ...,      # LISST body root (pelvis) locations in the world coordinate, with shape (t, 1, 3)
    'J_rotmat': ...,    # LISST joint rotation matrices in world coordinate wrt the canonical rest pose, with shape (t,J,3,3)
    'J_shape': ...,     # LISST bone lengths, with shape (J,)
    'J_locs_3d': ...,   # LISST joint locations in the world coordinate, with shape (t,J,3)
    'J_locs_2d': ...,   # LISST joint locations in indiviual camera views, only available for motion capture results, with shape (t, n_views, J, 2)
}
```
Please push this file into your own repo for auto-grading. Please don't change its path and format, otherwise the autograding system will not parse your file correctly.




## Setup
- Download the *Mediapipe Pose* results of the `CoreView_313` sequence [here](https://drive.google.com/drive/folders/1Vfu3vm4_GiZlpGPR2Dwwd56bRxcr1CrD?usp=sharing). 
```
1_mediapipe # this folder contains the rendered results of mediapipe
cam_params.json # the original camera parameters and names
mediapipe_all.pkl # the processed file containing the camera parameters and mediapipe keypoints
```

- You need to install and setup the LISST model (see below). To verify your installation, please check the `demos` folder. Note that you need to adapt the file paths in this codebase to your own paths.
- After LISST is successfully installed and setup, you can only focus on `scripts/app_multiview_mocap_ZJUMocap.py`, because all the implementation tasks are there.
But you may check other code for comprehensive understanding.


## Ex.1 The loss of body joint projection to images

**Code:**

- Files:
  - `scripts/app_multiview_mocap_ZJUMocap.py`
  
- Functions:
  - `img_project_loss(...)`

**Task:**

- Implement most part of this img_project_loss.


**Details:**

- The img_project_loss can be formulated as $L(Y,X) = \sum_{j, c} v^c_j |y^c_j - \phi(P^cx_j)|$, in which $x$, $y$, $v$, $P$ and $\phi(\cdot)$ are the 3D location, the 2D detected location, the detected visibility (ranging between 0 and 1), and the projection matrix, respectively. $_j$ and $^c$ denote the joint j and the camera c, respectively.
- Since this loss is important, we provide an unit test for it. Specifically, we provide a file `results/test_img_project_loss_data.pkl`, which contains the input and the correct output. The output from your implementation should let them match. See the function `test_img_project_loss()` for details.







## Ex.2 Multistage optimization

**Code:**

- Files:
  - `scripts/app_multiview_mocap_ZJUMocap.py`
  
- Functions:
  - some missing code blocks in `recover()`.

**Task:**

- Implement the multistage optimization.


**Details:**
- Inverse kinematics is a highly ill-posed problem. A good initialization is essential.
- In early stages, we only optimize the body global parameters.
- In late stages, we optimize both the global and the local body parameters.
- Multistages can be implemented by enabling/disabling updating certain variables.





## Ex.3 Temporal smoothness regularization

**Code:**

- Files:
  - `scripts/app_multiview_mocap_ZJUMocap.py`
  
- Functions:
  - check `loss_smoothness` in the function `recover()`.

**Task:**

- Implement the temporal smoothness loss.


**Details:**
- Human motion is smooth. Without this loss, obvious discontinuities are in the result.
- The smoothness loss can be formulated as $L = \sum_{j, t} |x^t_j - x^{t-1}_j|$ to penalize the velocity of the 3D joint locations. In addition, temporal smoothness can also be applied to rotations. Think of how they are implemented.



## Final Note

We will release our offcical code after the deadline of assignment 3. Further announcements will be sent.







----

----

----

# APPENDIX: Linear Shaped Skeletons for Human Body Modelling

![License](https://img.shields.io/hexpm/l/apa)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1%2Bcu113-brightgreen)


<img src="misc/lisst1.png" width="1000" class="center">

**LISST** is a *light-weight*, *differentiable*, and *parametric* human body model focusing on skeletons. Its down-streaming applications cover **markerless motion capture**, **motion synthesis and control**, **sports**, **game development**, **healthcare**, and beyond.
We further create an extension package [LISST-blender](https://github.com/yz-cnsdqz/LISST-blender) to use our model for character animation and motion re-targeting.


Provided the `body shape`, `body pose`, and `root configurations`, a posed body skeleton is produced via forward kinematics, and is placed in the world space.
We employ the [CMU Mocap](http://mocap.cs.cmu.edu/) dataset to build the body skeleton, learn the shape spaces, pose priors, and others.
Beyond the 31 body joints in CMU Mocap, additional bones are incorporated into the kinematic tree and the shape space, based on human anatomical statistics.


# Notice
- This version is to provide a framework for Assignment 3 of the course [Digital Humans 2023](https://vlg.inf.ethz.ch/teaching/Digital-Humans.html) at ETH Zurich. 
- Code, data, and model will be updated in future versions.
- [The official tutorial slides are here.](https://docs.google.com/presentation/d/1n9_FWMsHK-Iej1kg661XKpPUYipbXGGwX8pEMvK7uZo/edit?usp=sharing)



# Installation

**First**, create a virtual environment by running
```
python3 -m venv {path_to_venv}
source {path_to_venv}/bin/activate
```

**Second**, install all dependencies by running
```
pip install -r requirements.txt
```
Note that other versions might also work but not are not tested. 
In principle, this codebase is not sensitive to the Pytorch version, so please use the version that fits your own CUDA version and the GPU driver.


**Third**, install the `lisst` module into the environmentby running
```
python setup.py install
```


Before running the demos in `demos/*.ipynb`, datasets and checkpoints should be downloaded beforehand.


## For developers
Note that the files in `scripts` are for developers and have special path settings.
To run them properly, the `lisst` module should NOT be installed. Uninstall lisst can be done by
```
pip uninstall lisst
```



# Data and Models

## Data
### Processed CMU Mocap
Our work is developed based on the [CMU Mocap](http://mocap.cs.cmu.edu/) dataset.
We parse the raw `.asf` and `.amc` files, and re-organize them into the **canonicalized** format, which is defined in 
[MOJO](https://yz-cnsdqz.github.io/eigenmotion/MOJO/index.html) and [GAMMA](https://yz-cnsdqz.github.io/eigenmotion/GAMMA/index.html).
One can read `third_party/ACMParser` for detailed implementations. 
Our processed data can be downloaded [here](https://drive.google.com/drive/folders/14I_ufLfGlyldGeC3WrfBEMOiyP40l9wI?usp=sharing), which was used for training and testing our models.
```
CMU-canon-MPx8.pkl # all processed CMU sequences
CMU-canon-MPx8-train.pkl # all processed CMU sequences for training models
CMU-canon-MPx8-test.pkl # all processed CMU sequences for testing models or providing motion seeds
```
***Note that these pre-processed data may change with release versions.***



### ZJUMocap
We use [ZJUMocap-LightStage](https://chingswy.github.io/Dataset-Demo/) for the demo of markerless motion capture. Please download the this dataset first.
We also provide the *Mediapipe Pose* results of the `CoreView_313` sequence [here](https://drive.google.com/drive/folders/1Vfu3vm4_GiZlpGPR2Dwwd56bRxcr1CrD?usp=sharing). 
```
1_mediapipe # this folder contains the rendered results of mediapipe
cam_params.json # the original camera parameters and names
mediapipe_all.pkl # the processed file containing the camera parameters and mediapipe keypoints
```


## Models
Our pre-trained checkpoints can be downloaded [here](https://drive.google.com/drive/folders/1jcMbJgZtZEHqy-R8e1hjiTkR6V41aX08?usp=sharing). These checkpoints correspond to the model config files.
Please put all the downloaded files to the folder `results/lisst/`.

# Tutorials
We provide 2 tutorials in `demos/*.ipynb` to guide users how to use our LISST model. 
Please check these files for details.

# Citations
Please consider to cite the following works
```
@inproceedings{zhang2021we,
  title={We are more than our joints: Predicting how 3d bodies move},
  author={Zhang, Yan and Black, Michael J and Tang, Siyu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3372--3382},
  year={2021}
}

@inproceedings{zhang2022wanderings,
  title={The wanderings of odysseus in 3D scenes},
  author={Zhang, Yan and Tang, Siyu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20481--20491},
  year={2022}
}
```








