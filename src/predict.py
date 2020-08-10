import glob
import numpy as np
import os
import pickle
import pandas as pd
import argparse
import warnings
import yaml

from model.models import Models
from utils.data_generator import DataLoader, DatasetGenerator
from utils.data_loader import ImageLoader

warnings.filterwarnings("ignore")

# 予測結果保存先
predict_label_path = os.path.join(os.path.dirname(__file__), '../data/output/Pseudo_Label.pickle')
submission_path =  os.path.join(os.path.dirname(__file__), '../results/submission.csv')

# 設定読み込み
config_file = os.path.join(os.path.dirname(__file__), '../configs/config.yaml')
with open(config_file, encoding='utf-8') as file:
    yml = yaml.load(file)
common_setting = yml['COMMON_SETTING']
NUMBER_CLASSES =  common_setting['NUMBER_CLASSES'] # クラス数
HEIGHT = common_setting['HEIGHT'] # 画像高さ
WIDTH =  common_setting['WIDTH'] # 画像幅

def main(hyperparam, step):
    tta_step =  hyperparam['TTA_STEP'] #Test Time Augmentationの回数
    model_list = hyperparam['MODEL']  # モデルリスト
    
    predictions = []

    # テスト画像のロード
    loader = ImageLoader(validation_size=0.0, height=HEIGHT, width=WIDTH)
    X_test, image_name = loader.load_test()
    test_dataset = DatasetGenerator(X_test,np.zeros((len(X_test), 1)),8)
    test_dataset = DataLoader(test_dataset, batch_size=64)

    # モデルパスの取得
    model_path_list = []
    for model in model_list:
      path = config_file = os.path.join(os.path.dirname(__file__), '../' + model)
      model_path_list.extend(glob.glob(path))
    models = Models(num_classes = NUMBER_CLASSES, height=HEIGHT, width=WIDTH)

    for model_path in model_path_list:
        print(model_path)

        # モデルのロード
        model = models.load_model(model_path)

        for i in range(tta_step):
            print("{}/{}".format(i+1,tta_step))
            # 予測
            preds = model.predict(test_dataset)
            predictions.append(preds)
        del model

    # Ensemble（予測結果の平均）
    predict = np.mean(predictions, axis=0)

    if step == "1st":
        # 疑似ラベル用に予測結果を保存
        with open(predict_label_path, 'wb') as f:
            pickle.dump(predict,f)
    else:
        # 提出ファイルを作成
        predict_lbls = np.argmax(predict, axis=1)
        df_sample = pd.DataFrame(
            {
                'image': image_name,
                'gender_status':predict_lbls
            },
            columns = ['image','gender_status'] 
        )
        df_sample.to_csv(submission_path, index=False)
        df_sample.head()

if __name__ == "__main__":
    step = {"1st":"PREDICT_1ST_STEP", "2nd":"PREDICT_2ND_STEP" }
    # ハイパーパラメータ読み込み
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', default='1st')
    args = parser.parse_args()
    hyperparam = yml[step[args.step]]

    main(hyperparam, args.step)