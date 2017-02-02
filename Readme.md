Machine Learning for grasp selection.
This code is developed in collaboration between Pisa University and Duke University NC.

Author:
    daniela.resasco@gmail.com
    rocchi.alessio@gmail.com
    manuel.bonilla@centropiaggio.unipi.it

The goal of this project is make a deep neural network to allow an underactuate end-effector to learn how to grasp an unknown object. This work is in collaboration between the University of Pisa and Duke University in Durham North Carolina. To choose the desired poses the idea is to decompose the object in Minimum Volume Bounding Box minimizing the volume of the boxes which fit partial point clouds (see https://github.com/manuelbonilla/pacman_bbox).
For the neural network we utilize the supervised learning method, with deep convolutional neural network.


Dependency
    python v > 2.0, 
    pcl, 
    Klampt v0.7
    Theano

Dataset:
    apc2015: http://rll.berkeley.edu/amazon_picking_challenge/scripts/downloader.py
    ycb: http://rll.eecs.berkeley.edu/ycb/
    princeton: http://segeval.cs.princeton.edu/



To create a dataset:
    python Main_dany.py

To create a rotate mesh:
    python mesh_rotate.py 
    python check_poses.py 

Is possible made a poses for specify object as: 
    python Main_dany.py name_file
    python mesh_rotate.py
    python check_poses.py

To run the learning algorithm:
    cd 3DCNN 
    python Main.py

To make a voxel
    ./binvox path_to_file

