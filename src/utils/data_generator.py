import abc
import glob
import secrets
import warnings
import os
import yaml

import albumentations as A 
import numpy as np
import tensorflow as tf

warnings.filterwarnings("ignore")

# 設定読み込み
config_file = os.path.join(os.path.dirname(__file__), '../../configs/config.yaml')
with open(config_file, encoding='utf-8') as file:
    yml = yaml.load(file)
common_setting = yml['COMMON_SETTING']
NUMBER_CLASSES =  common_setting['NUMBER_CLASSES'] # クラス数
MEAN =  common_setting['MEAN'] # 平均
STD =  common_setting['STD'] # 偏差値

class Dataset(metaclass=abc.ABCMeta):
  """データセットを取得する"""
  @abc.abstractmethod
  def __len__(self):
    raise NotImplementedError

  @abc.abstractmethod
  def __getitem__(self, index):
    raise NotImplementedError

  def __iter__(self):
    for i in range(len(self)):
      yield self[i]

class DataLoader(tf.keras.utils.Sequence):
  """ バッチサイズごとにDatasetからデータを取り出す """

  def __init__(self, dataset, batch_size, shuffle=False, cutmix=False):
    assert len(dataset) > 0
    self.dataset = dataset
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.cutmix = cutmix
    self.indices = np.arange(len(self.dataset))
    if self.shuffle:
      np.random.shuffle(self.indices)

  def on_epoch_end(self):
    if self.shuffle:
      np.random.shuffle(self.indices)

  def __len__(self):
    return int(np.ceil(len(self.indices) / float(self.batch_size)))
  
  def __getitem__(self, index):
    seed = secrets.randbelow(2 ** 31)
    np.random.seed(seed)
    batch_indices = self.indices[self.batch_size * index : self.batch_size * (index + 1)]
    results = [self.get_sample(i) for i in batch_indices]
    X_batch, y_batch = zip(*results)
    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)
    return X_batch, y_batch

  def get_sample(self, index):
    X_i, y_i = self.dataset[index]
    if self.cutmix:
      randInt = np.random.rand()
      t = np.random.choice(len(self.indices))
      X_t, y_t = self.dataset[t]
      r =np.random.beta(0.3, 0.3)
      if randInt <= 0.5:
        # mixup
        X_i = X_i * r + X_t * (1 - r)
        y_i = y_i * r + y_t * (1 - r)
      else:
        # cutmix
        bx1, by1, bx2, by2 = self.get_rand_bbox(X_t, r)
        X_i[bx1:bx2, by1:by2, :] = X_t[bx1:bx2, by1:by2, :]
        r = 1 - ((bx2 - bx1) * (by2 - by1) / (X_i.shape[0] * X_i.shape[1]))
        y_i = r * y_i + (1 - r) * y_t
    return X_i, y_i

  def get_rand_bbox(self,image, l):
    """
    Cutmix用のバウンディングボックス
    """
    width = image.shape[0]
    height = image.shape[1]
    r_x = np.random.randint(width)
    r_y = np.random.randint(height)
    r_l = np.sqrt(1 - l)
    r_w = np.int(width * r_l)
    r_h = np.int(height * r_l)
    bb_x_1 = np.int(np.clip(r_x - r_w // 2, 0, width))
    bb_y_1 = np.int(np.clip(r_y - r_h // 2, 0, height))
    bb_x_2 = np.int(np.clip(r_x + r_w // 2, 0, width))
    bb_y_2 = np.int(np.clip(r_y + r_h // 2, 0, height))
    return bb_x_1, bb_y_1, bb_x_2, bb_y_2

class DatasetGenerator(Dataset):
  """ データソースから1個ずつデータを取り出す """
  def __init__(self, X, y=None, mode="", height=256, width=256, X_test=[], y_pseudo=None):
    self.X = X
    self.y = y
    self.mode = mode
    self.height = height
    self.width = width
    self.X_test = X_test
    self.y_pseudo = y_pseudo

  def __len__(self):
    return len(self.X) + len(self.X_test)
 
  def __getitem__(self, index):
    if len(self.X) > index:
      X =  self.X[index]
      y = tf.keras.utils.to_categorical(self.y[index][0], NUMBER_CLASSES)
    else:
      X = self.X_test[index - len(self.X)]
      y = self.y_pseudo[index - len(self.X)]
    if self.mode == "train": 
      aug = self.transforms_train()
      X = self.normalize(aug(image=X)['image']) / 255
    elif self.mode == "test":
      aug = self.transforms_test()
      X = self.normalize(aug(image=X)['image']) / 255
    else:
      X = self.normalize(X.astype('float32')) / 255
    return X, y

  def normalize(self, image):
    """ 標準化 """
    image = image.transpose(2, 0, 1)  # Switch to channel-first
    mean, std = np.array(MEAN), np.array(STD)
    image = (image - mean[:, None, None]) / std[:, None, None]
    return image.transpose(1, 2, 0)

  def transforms_train(self, p=1.0):
      """ 学習時用DataAugmentation """
      return A.Compose([
                 A.HorizontalFlip(p=0.5),
                 A.CenterCrop(height=int(self.height), width=int(self.width*0.8), p=0.3),
                 A.RandomCrop(height=int(self.height*0.9), width=int(self.width*0.8), p=0.3),
                 A.OneOf([
                          A.CLAHE(clip_limit=4.0, tile_grid_size=(16,16),p=0.7),
                          A.Blur(blur_limit=7, p=0.1),
                          ],p=0.3),
                 A.HueSaturationValue(hue_shift_limit=5,sat_shift_limit=30,val_shift_limit=50,p=0.3),
                 A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, brightness_by_max=True,p=0.2),
                 A.OneOf([
                          A.IAAAffine(shear = 20.0,p=0.5), 
                          A.Rotate(limit = 30.0,p=1), 
                          A.IAAPerspective(keep_size=True,p=0.5),
                 ],p=0.5),
                 A.Resize(height = self.height, width = self.width),
              ], p=p)

  def transforms_test(self, p=1.0):
      """ テスト時用DataAugmentation """
      return A.Compose([
                 A.HorizontalFlip(p=0.5),
                 A.OneOf([
                          A.CLAHE(clip_limit=4.0, tile_grid_size=(16,16),p=0.7),
                          A.Blur(blur_limit=7, p=0.1),
                          ],p=0.3),
                 A.HueSaturationValue(hue_shift_limit=5,sat_shift_limit=30,val_shift_limit=50,p=0.3),
                 A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, brightness_by_max=True,p=0.2),
              ], p=p)