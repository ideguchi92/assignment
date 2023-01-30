# 課題2 - 解決手法の実装

## 実装進捗

能力不足により、全ての機能の実装が間に合いませんでした。現在の進捗は以下の通りで、実行すると推定したボックスと姿勢を追記した映像を出力します。

### 完了

  Multi Object Detection: YOLOX(ByteTrack)  
  Multi Object Tracking: ByteTrack  
  Multi Pose Estimation: ViTPose

### 未完了

  静止判定  
  視認判定  
  点灯/消灯判定

## Install

Dockerがインストール済みと仮定する。
1. 下記コマンドを実行する。
  ```shell
  git clone --depth=1 https://github.com/ideguchi92/assignment
  cd assignment/
  docker build --rm -t assignment_env .
  mkdir {data,logs,models}
  ```

2. 推論する動画をdataディレクトリに配置する。  
  例:  
  https://pixabay.com/ja/videos/%E3%83%AA%E3%83%90%E3%83%97%E3%83%BC%E3%83%AB-%E4%B8%80-%E8%B2%B7%E3%81%84%E7%89%A9-%E4%B8%AD%E5%BF%83-29248/  
  https://pixabay.com/ja/videos/%E3%83%96%E3%83%A9%E3%83%83%E3%82%AF%E3%83%95%E3%83%A9%E3%82%A4%E3%83%87%E3%83%BC-%E9%BB%92-%E9%87%91%E6%9B%9C%E6%97%A5-29459/

3. 下記モデルをダウンロードして、modelsディレクトリに配置する。  
  ByteTrack:  
  https://drive.google.com/file/d/1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj/view?usp=sharing  
  ViTPose:  
  https://1drv.ms/u/s!AimBgYV7JjTlgccifT1XlGRatxg3vw?e=9wz7BY

## Run

assignmentディレクトリで下記コマンドを実行する。入出力ファイル名は適宜変更する。
```shell
docker run --gpus all -it --rm \
-v ${PWD}/main.py:/assignment/main.py \
-v ${PWD}/data/:/assignment/data/ \
-v ${PWD}/logs/:/assignment/logs/ \
-v ${PWD}/models/:/assignment/models/ \
-v ${PWD}/src/:/assignment/src/ \
assignment_env python3 main.py -i <require: ex. data/input.mp4> -o <default: data/output.mp4> --half
```

## Demo

現在実装完了部分で実行した結果はこちら  
https://drive.google.com/file/d/14VSTVNWGoEN1-yKBdrzPY3umvjg3aKWL/view?usp=share_link  
https://drive.google.com/file/d/1JaM7vK9gzyn_3cDUAE4Ge2EgZBE_MQtU/view?usp=share_link
