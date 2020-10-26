## tf lite  예제

### 개요
텐서플로라이트는 텐서와는 별로로 실행가능한 인터프리터만 따로 제공하고있다. 뿐만아니라 텐서 프로로우 풀버전에도 인터프리터가 포함되어있으므로 둘중하나 만설치하면된다.
라즈베리같은 임배디드 머신에서는 학습시키는목적 보단 학습된 모델을 불러들어 추론하는 용도로만 사용하므로 텐서라이트만 설치하는것이 합리적이다. 

### 설치하기

https://www.tensorflow.org/lite/guide/python 에서 맞는 버전을 선택하여 설치한다.

mac용 python3.7용 설치 
```
pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-macosx_10_14_x86_64.whl

```


### 샘플모델 다운받기
```
wget https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1?lite-format=tflite
```

tf1용 mobile ssd 모델에서 tflite 파일을 제공한다.  
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#mobile-models   


