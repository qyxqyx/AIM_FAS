#This code is the implementation of the paper "Learning Meta Model for Zero- and Few-shot Face Anti-spoofing"

The image lists of the protocol OULU-ZF is included in the protocols folder, and the protocol Cross-ZF will come soon.

Env required: tensorflow >= 1.4.0, opencv >= 3.0.0, etc.

To run this code, you should first use PRNet or other tools to generate facical depth map as the labels of all living faces.
The labels of all spoofing faces are a zero array with shape of [32, 32, 1].


Data structure:
Each set of the train, val, and test sets contains fine-grained living and spoofing face types.
For each fine-grained type, it contains several facial images and the corresponding facial box files and facial depth map images.
The image, the facial box file, and the facial depth image should be end-with '_scene.jpg',
'_scene.dat', and '_depth1D.jpg', respectively.


@article{qin2020learning,
title={Learning Meta Model for Zero- and Few-shot Face Anti-spoofing},
author={Qin, Yunxiao and Zhao, Chenxu and Zhu, Xiangyu and Wang, Zezheng and Yu, Zitong and Fu, Tianyu and Zhou, Feng and Shi, Jingping and Lei, Zhen},
year={2020}}
