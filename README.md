# Deep Learning-based Image Classification of Faulty Ethernet Cables    
## Objectives   
1. To train the [Squeezenet model](https://github.com/forresti/SqueezeNet/tree/master/SqueezeNet_v1.0) on the Ethernet cables images to classify the status of Ethernet cables on the device under test   
2. To create the hardware-optimized inference engine with the trained model using the Intel OpenVINO toolkit for the Intel CPU and Intel-Movidius VPU      
3. To develop the Python/OpenCV-based application to classify the real-time status of Ethernet cables in the webcam images   

## Hardware Platform   
### Training Node   
#### Hardware   
CPU: Intel Core i7-8700K CPU @ 3.70GHz × 12   
GPU: GeForce GTX 1080 x 2   
RAM: 32 GB   
OS: Ubuntu 16.04.5 LTS   
Refer to this [link](https://github.com/ramesh-dev-code/misaligned-heat-sink#training-platform-setup) for the training platform setup   
#### Software   
[Intel-Optimized Caffe](https://github.com/ramesh-dev-code/led-status-inference#installation-of-intel-optimized-caffe)   
[Intel OpenVINO Toolkit](https://github.com/ramesh-dev-code/led-status-inference#installation-of-intel-openvino-toolkit)   

### Edge Node   
#### Hardware   
CPU: Intel Xeon CPU   
RAM: 16 GB   
#### Software      
[Intel OpenVINO Toolkit](https://github.com/ramesh-dev-code/led-status-inference#installation-of-intel-openvino-toolkit)    

## Dataset   
525 images per class   
### Output Classes
ON: All the Ethernet cables are ON   
OFF: At least one Ethernet cable is OFF      

### Sample Images   
**ON**   
![](https://i.imgur.com/m79zxtZ.png)   
**OFF**   
![](https://i.imgur.com/9fTobEk.png)   

## Training   
Train the fine-tuned Squeezenet model on the dataset using two GPUs   
```
$CAFFE_ROOT/build/tools/caffe train --solver solver.prototxt --weights squeezenet_v1.0.caffemodel --gpu all
```
Execution Time: 1h 3m 40s; Iterations: 50000   

### Training and Validation Performance   
![](https://i.imgur.com/vMEmD5t.png)   
![](https://i.imgur.com/TS2y24t.png)   
![](https://i.imgur.com/rK55G40.png)   

## Building Optimized Inference Engine on Edge Node   
Building the optimized inference engine for the Intel CPU   
```
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --framework caffe --data_type FP32 --input_shape [1,3,227,227] --input data --mean_values data[104.0,117.0,123.0] --output prob --input_model train_cp_iter_50000.caffemodel --input_proto deploy.prototxt --output_dir ./
```
Building the optimized inference engine for the Intel VPU   
```
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --framework caffe --data_type FP16 --input_shape [1,3,227,227] --input data --mean_values data[104.0,117.0,123.0] --output prob --input_model train_cp_iter_50000.caffemodel --input_proto ./FP16/deploy.prototxt --output_dir ./FP16/
```
## Testing on Edge Node      
### Image Classification on CPU  
```
/home/puzzle/Documents/CheckPoint/Classification/ImageClassification/image_classification_sync -d CPU -m /home/puzzle/Documents/CheckPoint/Classification/FP32/train_cp_iter_50000.xml -i /home/puzzle/Documents/CheckPoint/Classification/test_input.jpg
```
![](https://i.imgur.com/qEUtZLf.png)   
Inference Time: ~3 ms   
### Image Classification on VPU  
```
/home/puzzle/Documents/CheckPoint/Classification/ImageClassification/image_classification_sync -d MYRIAD -m /home/puzzle/Documents/CheckPoint/Classification/FP16/train_cp_iter_50000.xml -i /home/puzzle/Documents/CheckPoint/Classification/test_input.jpg
```
![](https://i.imgur.com/qtaihZl.png)   
Inference Time: ~15 ms   

### Real-time Image Classification on the webcam video on CPU   
```
/home/puzzle/Documents/CheckPoint/Classification/VideoClassification/video_classification_async -d CPU -m /home/puzzle/Documents/CheckPoint/Classification/FP32/train_cp_iter_50000.xml -i /dev/video0
```
![](https://i.imgur.com/1Wk7edZ.png)   

### Real-time Image Classification on the webcam video on VPU   
```
/home/puzzle/Documents/CheckPoint/Classification/VideoClassification/video_classification_async -d MYRIAD -m /home/puzzle/Documents/CheckPoint/Classification/FP16/train_cp_iter_50000.xml -i /dev/video0
```
![](https://i.imgur.com/Yc4aAur.png)   
