# MvSMPLfitting
A multi-view SMPL fitting based on smplify-x

![figure](/images/teaser.jpg)

## Dependencies
```conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch```<br>
```pip install -r requirements.txt```


## Demo
Download the netural model from [SMPLify website](http://smplify.is.tuebingen.mpg.de/) and the male/female model from [SMPL website](https://smpl.is.tue.mpg.de/). Then, rename the .pkl files and put them in ```models/smpl``` folder. (see [models/smpl/readme.txt](./models/smpl/readme.txt))

Run ```python code/main.py```


## Reference
```
@article{outdoor,
  title = {Outdoor Markerless Motion Capture with Sparse Handheld Video Cameras},
  author = {Yangang Wang, Yebin Liu, Xin Tong, Qionghai Dai and Ping Tan},
  booktitle = {IEEE Transactions on Visualization and Computer Graphics},
  year = {2018}
}
@inproceedings{SMPL-X:2019,
  title = {Expressive Body Capture: 3D Hands, Face, and Body from a Single Image},
  author = {Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
  booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019}
}
```
