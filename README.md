# MvSMPLfitting
A multi-view SMPL fitting based on smplify-x

![figure](/images/teaser.jpg)

## Dependencies
```conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch```<br>
```pip install -r requirements.txt```


## Demo
Download the official SMPL model from [SMPLify website](http://smplify.is.tuebingen.mpg.de/) \(netural) and [SMPL website](https://smpl.is.tue.mpg.de/) \(male/female). Then, rename the .pkl files and put them in the ```models/smpl``` folder. (see [models/smpl/readme.txt](./models/smpl/readme.txt))

Run ```python code/main.py --config cfg_files/fit_smpl.yaml```


## Reference
```
@inproceedings{zhang2020object,
  title={Object-Occluded Human Shape and Pose Estimation From a Single Color Image},
  author={Zhang, Tianshu and Huang, Buzhen and Wang, Yangang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7376--7385},
  year={2020}
}
@inproceedings{SMPL-X:2019,
  title = {Expressive Body Capture: 3D Hands, Face, and Body from a Single Image},
  author = {Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
  booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019}
}
```
