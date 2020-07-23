# tensorflow object detection api sample for RPi


## 모델 파일 다운로드 

```bash
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -d Sample_TFLite_model

```

## 텐서플로 라이트 설치

필수 모듈과 opencv(option) 을 설치하고 [파이썬용 텐서라이트](https://www.tensorflow.org/lite/guide/python?hl=ko) whl 파일을 다운받아 설치한다.
링크로 연결된곳에 들어가면 플랫폼별로 정리된 whl파일 리스트를 얻을수 있다. 라즈베리같은 arm리눅스용 이외에도 win10, mac용 텐서라이트가 있다.


```bash

sudo apt-get -y install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get -y install libxvidcore-dev libx264-dev
sudo apt-get -y install qt4-dev-tools libatlas-base-dev

pip3 install opencv-python==3.4.6.27

pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl
```

## 참고자료  
[객체 감지 공식 자료(텐서플로 라이트)](https://www.tensorflow.org/lite/models/object_detection/overview?hl=ko)  
[라즈베리셋업 방법 정리된 자료](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md)