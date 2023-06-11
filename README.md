# computer-vision-project
Project for the Computer Vision course @ University of Trento a.y. 2021/2022

Alessandro Zinni, Davide Guidolin, Joy Battocchio

https://github.com/Davide-Guidolin/computer-vision-project/assets/17050407/76dbc502-4414-458c-96c8-74fae6ae5a7b

## Goal
This project aimed to reconstruct the 3D model of an object using Neural Radiance Fields (NeRF).
For our experiments we used [ngp_pl](https://github.com/kwea123/ngp_pl), a PyTorch implementation of [instant-ngp](https://github.com/NVlabs/instant-ngp).
We tested also [NeuS](https://github.com/Totoro97/NeuS) which is another neural surface reconstruction method, but without success.
After reconstructing the model, we animated it using motion capture data obtained from the same object.

## Usage
We provide a python notebook [NGP_pl.ipynb](./NGP_pl.ipynb) in which we use ngp-pl code to train a model and export the mesh. We used [these tips](https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md) to build our datasets.

Note that if you use colab, the installation of dependencies may give some problems, especially when installing apex. We added some instructions to install it but it may give different problems in the future.

We provide also 2 folders: [project_Folder](./project_Folder) and [project_Folder_NeuS](./project_Folder_NeuS) which contain the code used to run the experiment using NGP_pl and Neus respectively. To run the experiment we used a Nvidia RTX 2070 provided by our University, and we used [determined](https://www.determined.ai/) to connect to the university cluster and run our jobs, so the folders are configured to be used with determined.

[Here](https://drive.google.com/file/d/1uJIecP1xy_qxyxs7qxf_hu0BxEJF5t2L/view?usp=sharing) we provide also the Unity project containing the mesh animation. We used `Unity 2020.3.8f1` to create it.
