import functools
import os
import warnings
import yaml
import tensorflow as tf
from efficientnet.tfkeras import EfficientNetB4
from efficientnet import model as efficientnetmodel

warnings.filterwarnings("ignore")

# 設定読み込み
config_file = os.path.join(os.path.dirname(__file__), '../../configs/config.yaml')
with open(config_file, encoding='utf-8') as file:
    yml = yaml.load(file)
common_setting = yml['COMMON_SETTING']
NUMBER_CLASSES =  common_setting['NUMBER_CLASSES'] # クラス数

class Models:

    def __init__(self, num_classes = 8, height=256, width=256):
        self.height = height
        self.width = width
        self.num_classes = num_classes
        self.input_tensor = tf.keras.layers.Input(shape=(self.height, self.width, 3))

    def custom_loss(self, y_true, y_pred):
        """ Categorical Crossentropy + Label Smoothing """
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=0.1)

    def inject_tfkeras_modules(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            kwargs['backend'] = tf.keras.backend
            kwargs['layers'] = tf.keras.layers
            kwargs['models'] = tf.keras.models
            kwargs['utils'] = tf.keras.utils
            return func(*args, **kwargs)
        return wrapper

    def load_model(self, model_file):
        """ 保存したモデルのロード """
        return tf.keras.models.load_model(model_file, 
                custom_objects={'swish':self.inject_tfkeras_modules(efficientnetmodel.get_swish)(), 
                                'FixedDropout':self.inject_tfkeras_modules(efficientnetmodel.get_dropout)(), 
                                'custom_loss':self.custom_loss})

    def get_model(self, model ,lr=0.001):
        """ 学習済みモデルの取得 """
        models = {"ResNet152":self.resnet152, 
                  "Xception":self.xception,
                  "EfficientNet":self.efficientnet
                 }
        model = models[model]()
        optimizer = tf.keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=True)
        model.compile(optimizer=optimizer, loss=self.custom_loss, metrics=['acc'])
        return model

    def efficientnet(self):
        """ EfficientNetB4 """
        base_model = EfficientNetB4(weights='imagenet',include_top=False,input_tensor=self.input_tensor)

        # 約8割の層はパラメータ固定
        for layer in base_model.layers[:370]:
            layer.trainable = False
            # Batch Normalization freeze解除
            if layer.name.startswith('batch_normalization'):
                layer.trainable = True
            if layer.name.endswith('bn'):
                layer.trainable = True
        # 約2割だけ学習
        for layer in base_model.layers[370:]:
            layer.trainable = True

        x_in = self.input_tensor
        x = base_model(x_in)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(2028, activation='relu', kernel_initializer='he_normal', 
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(self.num_classes, activation='softmax', use_bias=False, 
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        return tf.keras.models.Model(x_in, x) 

    def xception(self):
        """ Xception """
        base_model = tf.keras.applications.Xception(weights="imagenet",
                                                    include_top=False,
                                                    input_tensor=self.input_tensor)

        # 約8割の層はパラメータ固定
        for layer in base_model.layers[:100]:
            layer.trainable = False
            # Batch Normalization freeze解除
            if layer.name.startswith('batch_normalization'):
                layer.trainable = True
            if layer.name.endswith('bn'):
                layer.trainable = True
        # 約2割だけ学習                
        for layer in base_model.layers[100:]:
            layer.trainable = True

        x_in = self.input_tensor
        x = base_model(x_in)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(2048, activation='relu', 
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(self.num_classes, 
                                  activation='softmax', 
                                  use_bias=False, 
                                  kernel_initializer='he_normal', 
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        return tf.keras.models.Model(x_in, x)

    def resnet152(self):
        """ ResNet152V2 """
        base_model = tf.keras.applications.ResNet152V2(weights="imagenet",
                                                       include_top=False,
                                                       input_tensor=self.input_tensor)

        # 約8割の層はパラメータ固定
        for layer in base_model.layers[:450]:
            layer.trainable = False
            # Batch Normalization freeze解除
            if layer.name.startswith('batch_normalization'):
                layer.trainable = True
            if layer.name.endswith('bn'):
                layer.trainable = True
        # 約2割だけ学習
        for layer in base_model.layers[450:]:
            layer.trainable = True
        x_in = self.input_tensor
        x = base_model(x_in)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(2048, activation='relu', 
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(self.num_classes, activation='softmax', 
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        return tf.keras.models.Model(x_in, x)
