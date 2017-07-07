# Face_Recognition
Real-time face recognition by using Keras(Tensorflow as backend), can be applied on Raspberry Pi3.  
## Installation
We need to download python3.5 for using tensorflow on Windows, and we also need to download some python packages such as: numpy, opencv, scikit-learn, scipy, h5py, tensorflow, keras... For Tensorflow-GPU, we need to download CUDA Toolkit8.0.  
### For Windows
All packages used can be download on http://www.lfd.uci.edu/~gohlke/pythonlibs/.    
Use `python -m pip install <packages.xml>` to install, run as root.    
### For Linux
```
sudo pip3 install numpy opencv-python scipy scikit-learn h5py
```
#### Install Tensorflow and Keras
For Tensorflow, see https://www.tensorflow.org/install/install_linux.      
For Keras, use `sudo pip3 install keras`.    
### For Raspbian
```
sudo pip3 install numpy opencv-python scipy scikit-learn h5py
```
#### Install Tensorflow and Keras
For Tensorflow, see https://github.com/samjabrahams/tensorflow-on-raspberry-pi.    
For Keras, use `sudo pip3 install keras`.    
## Creating your own data set
```
python training.py --input_dir=<> --model_dir=<> --epoch=<> --batch_size=<> --data_augmentation=<>
```

