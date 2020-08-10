import glob
import warnings
import os
import pickle
import yaml

import numpy as np
import pandas as pd
import tensorflow as tf


warnings.filterwarnings("ignore")


# 設定読み込み
config_file = os.path.join(os.path.dirname(__file__), '../../configs/config.yaml')
with open(config_file, encoding='utf-8') as file:
    yml = yaml.load(file)
common_setting = yml['COMMON_SETTING']
TRAIN_LABEL_PATH = common_setting['TRAIN_LABEL_PATH']
TRAIN_IMG_PATH = common_setting['TRAIN_IMG_PATH']
TEST_IMG_PATH = common_setting['TEST_IMG_PATH']

train_label_path = os.path.join(os.path.dirname(__file__), '../../' + TRAIN_LABEL_PATH)
train_img_path = os.path.join(os.path.dirname(__file__), '../../' + TRAIN_IMG_PATH)
test_img_path = os.path.join(os.path.dirname(__file__), '../../' + TEST_IMG_PATH)


class ImageLoader(object):
    """ 画像ファイル読み込み
    Attributes
        validation_size : float
          検証用データの割合(デフォルト:0)
        height : int
          画像高さ(デフォルト:256)
        width : int
        　画像幅(デフォルト:256)

    """

    def __init__(self, validation_size: float=0.0, height = 256, width = 256):
        """
        Parameters
            validation_size : float
                検証用データの割合(デフォルト:0.0)
            height : int
                画像高さ(デフォルト:256)
            width : int
            　　画像幅(デフォルト:256)

        """
        self.validation_size = validation_size
        self.height = height
        self.width = width

    def load_train(self, random_seed: int=123) -> np.ndarray:
        """  訓練用画像ファイルを読み込む

        Parameters
            random_seed : int
          シャッフル時のシード(デフォルト:123)

        Returns
            X_train : List
                学習用画像
            y_train : List
                学習用画像
            X_valid : List
                検証用画像
            y_valid : List
                検証用ラベル

        """
        # 画像の読み込み
        img_files = glob.glob(train_img_path)
        all_imgs = []
        for img_file in img_files:
            img = tf.keras.preprocessing.image.load_img(img_file,(self.height, self.width) )
            image_array = tf.keras.preprocessing.image.img_to_array(img)
            all_imgs.append(image_array)
        all_imgs = np.array(all_imgs)

        # ラベルの読み込み
        all_lbls = pd.read_csv(train_label_path)
        all_lbls = all_lbls.sort_values(by="image").gender_status.values
        all_lbls = all_lbls.reshape(len(all_lbls),1)

        # データをシャッフル
        np.random.seed(random_seed)
        perm_idx = np.random.permutation(len(all_lbls))
        all_imgs = all_imgs[perm_idx]
        all_lbls = all_lbls[perm_idx]

        # バリデーション用分割
        valid_num = int(len(all_lbls)*self.validation_size)
        X_train = all_imgs[valid_num:]
        y_train = all_lbls[valid_num:]
        X_train = np.array(X_train).astype('uint8')
        y_train = np.array(y_train).reshape(len(y_train), -1).astype('uint8')
        X_valid = []
        y_valid = []
        if self.validation_size > 0:
            X_valid = all_imgs[:valid_num]
            y_valid = all_lbls[:valid_num]
            X_valid = np.array(X_valid).astype('uint8')
            y_valid = np.array(y_valid).reshape(len(y_valid), -1).astype('uint8')
        return X_train, y_train, X_valid, y_valid

    def load_test(self) -> np.ndarray:
        """  テスト用画像ファイルを読み込む

        Returns
            X_test : List
                テスト画像
            image_name : List
                画像ファイル名

        """
        img_files = glob.glob(test_img_path)
        all_imgs = []
        image_name = []
        for img_file in img_files:
            image_name.append(os.path.split(img_file)[1])
            img = tf.keras.preprocessing.image.load_img(img_file,(self.height, self.width) )
            image_array = tf.keras.preprocessing.image.img_to_array(img)
            image_array = image_array.astype('uint8')
            all_imgs.append(image_array)
        X_test = np.array(all_imgs)
        return X_test, image_name

    def load_pseudo(self, pseudo_label_path, threshold:float=0.5 ) -> np.ndarray:
        """  疑似ラベルを付与したテストデータを取得する

        Parameters
            pseudo_label_path : string
                疑似ラベルのファイルのパス
            threshold : float
                閾値

        Returns
            X_test : List
                テスト画像
            y_pseudo : List
                疑似ラベル

        """
        # 画像の読み込み
        X_test, image_name = self.load_test()
        # 疑似ラベル読み込み
        pseudo_label_path = os.path.join(os.path.dirname(__file__), '../../' + pseudo_label_path)
        with open(pseudo_label_path, mode='rb') as f:
            y_pseudo = np.array(pickle.load(f))
        # 閾値以上を取り出し
        index = np.where(np.max(y_pseudo,axis=1) > threshold)
        X_test = X_test[index]
        y_pseudo = y_pseudo[index]

        return X_test, y_pseudo