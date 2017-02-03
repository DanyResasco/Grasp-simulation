# Machine Learning for grasp selection.
This code is developed in collaboration between Pisa University and Duke University NC.

Authors:
   > - Daniela Resasco   daniela.resasco@gmail.com
   > - Alessio Rocchi    rocchi.alessio@gmail.com
   > - Manuel Bonilla    josemanuelbonilla@gmail.com

The goal of this project is make a deep neural network to allow an underactuate end-effector to learn how to grasp an unknown object. This work is in collaboration between the University of Pisa and Duke University in Durham North Carolina. To choose the desired poses the idea is to decompose the object in Minimum Volume Bounding Box minimizing the volume of the boxes which fit partial point clouds  [GitHub Pages] (https://github.com/manuelbonilla/pacman_bbox).
For the neural network we utilize the supervised learning method, with deep convolutional neural network.


Dependencies
   > - python v > 2.0, 
   > - pcl, 
   > - eigen 3,
   > - Klampt v0.7,
   > - Theano

Datasets
    > - apc2015   http://rll.berkeley.edu/amazon_picking_challenge/scripts/downloader.py
    > - ycb       http://rll.eecs.berkeley.edu/ycb/
    > - princeton http://segeval.cs.princeton.edu/



To create a dataset
   > python Main_dany.py

To create a rotated mesh
   > python mesh_rotate.py mesh_name

To test a pose on the rotated mesh
   > python check_poses.py mesh_name
   > - es: python check_poses.py xwing

To generate poses for a specified object
   > python Main_dany.py name_file

To test poses on the specified object
   > python check_pose_original.py mesh_name
   > - es: python check_pose_original.py xwing


To run the learning algorithm:
    > cd 3DCNN 
    > python Main.py

To create a voxel from a mesh
   > ./binvox path_to_file

