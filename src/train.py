import warnings
import argparse
import os
import yaml

import numpy as np
from sklearn.metrics import confusion_matrix
from utils.data_loader import ImageLoader
from utils.data_spliter import oversampled_Kfold
from utils.data_generator import DataLoader, DatasetGenerator
from model.callbacks import CosineAnnealing
from model.models import Models

warnings.filterwarnings("ignore")

# 設定読み込み
config_file = os.path.join(os.path.dirname(__file__), '../configs/config.yaml')
with open(config_file, encoding='utf-8') as file:
    yml = yaml.load(file)
common_setting = yml['COMMON_SETTING']
HEIGHT = common_setting['HEIGHT'] # 画像高さ
WIDTH =  common_setting['WIDTH'] # 画像幅
NUMBER_CLASSES =  common_setting['NUMBER_CLASSES'] # クラス数

def main(hyperparam, step):
    # ハイパーパラメータ設定
    batch_size =  hyperparam['BATCH_SIZE'] # バッチサイズ
    epochs =  hyperparam['EPOCHS'] # エポック数
    lr =  hyperparam['LEARNING_RATE'] # 学習率
    model_list = hyperparam['MODEL'] # モデルリスト
    pseudo = hyperparam['PSEUDO'] # 疑似ラベル

    # 画像読み込み
    loader = ImageLoader(validation_size=0.0, height=HEIGHT, width=WIDTH)
    X_train, y_train, X_valid, y_valid = loader.load_train()
    # 疑似ラベル取得
    X_test = []
    y_pseudo = None
    if pseudo != None:
        X_test, y_pseudo = loader.load_pseudo(pseudo["PATH"], pseudo["THRESHOLD"])
    # OverSampling & K-fold
    rkf_search = oversampled_Kfold(n_splits=5, n_repeats=1)
    # モデル
    models = Models(num_classes = NUMBER_CLASSES, height=HEIGHT, width=WIDTH)

    for model_name in model_list:
        for n,(train_index, valid_index) in enumerate(rkf_search.split(X_train, y_train)):
            # 学習用データ
            X_train_kfold = X_train[train_index]
            y_train_kfold = y_train[train_index]
            train_dataset = DatasetGenerator(X_train_kfold,  y_train_kfold, mode="train",
                                            height=HEIGHT, width=WIDTH, X_test=X_test, y_pseudo=y_pseudo)
            train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True ,cutmix=True)
            # 検証用データ
            X_valid = X_train[valid_index]
            y_valid = y_train[valid_index]
            valid_dataset = DatasetGenerator(X_valid, y_valid, height=HEIGHT, width=WIDTH)
            valid_data = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
            # モデル取得
            model = models.get_model(model_name, lr = lr*batch_size)
            # 学習
            model.fit(train_data, 
                validation_data=valid_data,
                epochs=epochs,
                callbacks=[CosineAnnealing()],
                verbose=1,
            )
            # 学習モデルの保存
            model_output = os.path.join(os.path.dirname(__file__),
                                        '../models/{}_{}_{}.h5'.format(step, model_name, n))
            model.save(model_output)
        
            # 混同行列を表示
            predict = model.predict(valid_data)
            predict_lbls = np.argmax(predict, axis=1)
            print(confusion_matrix(y_valid, predict_lbls))

            del model
            del train_dataset
            del train_data
            del valid_dataset
            del valid_data

if __name__ == "__main__":
    step = {"1st":"TRAINING_1ST_STEP", "2nd":"TRAINING_2ND_STEP" }
    # ハイパーパラメータ読み込み
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', default='1st')
    args = parser.parse_args()
    hyperparam = yml[step[args.step]]

    main(hyperparam, args.step)
