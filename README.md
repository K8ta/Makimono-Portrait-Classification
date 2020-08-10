# Makimono Portrait Classification

## 環境
OS:Windows10

Python3.7

### ライブラリ
- tensorflow:2.1.0
- albumentations:0.4.5
- efficientnet:1.1.0
- imbalanced-learn:0.6.2
- numpy:1.18.3
- pandas:1.0.3
- scikit-learn:0.22.2.post1
- scipy:1.4.1


## 使用方法

### 準備
学習用ラベルの配置
```sh
data/input/train.csv
```

学習用画像の配置
```sh
data/input/train/
```
テスト用画像の配置
```sh
data/input/test/
```

### 実行
```sh
./run.sh
```

### 出力
```sh
results/submission.csv
```
