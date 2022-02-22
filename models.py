import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.layers import *
from tensorflow.keras.models import *


class makemodel:
    def __init__(self):
        self.result = 0

    def Vgg19():
        vgg19 = VGG19()
        model = Sequential()
        model.add(vgg19)

        return model
    
    def Resnet():
        resnet = ResNet50()
        model = Sequential()
        model.add(resnet)

        return model

    def Efficientnet():
        effnet = EfficientNetB0()
        model = Sequential()
        model.add(effnet)

        return model