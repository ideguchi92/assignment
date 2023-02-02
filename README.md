# 課題2 - 解決手法の実装

2023/02/03ver.  
[2023/01/30ver.](https://github.com/ideguchi92/assignment/tree/v0.0.1)

## 実装進捗

### 完了

  Multi Object Detection: YOLOX(ByteTrack)  
  Multi Object Tracking: ByteTrack  
  Multi Pose Estimation: ViTPose  
  静止判定  
  視認判定  
  点灯/消灯判定

## Install

Dockerがインストール済みと仮定する。
1. 下記コマンドを実行する。
  ```shell
  git clone --depth=1 -b v0.0.2 https://github.com/ideguchi92/assignment.git
  cd assignment/
  docker build --rm -t assignment_env .
  mkdir {data,logs,models}
  ```

2. 推論する動画をdataディレクトリに配置する。  
  例:  
  https://pixabay.com/ja/videos/%E3%83%AA%E3%83%90%E3%83%97%E3%83%BC%E3%83%AB-%E4%B8%80-%E8%B2%B7%E3%81%84%E7%89%A9-%E4%B8%AD%E5%BF%83-29248/  
  (適切な動画が見つからなかったため、50fpsですがこの動画を使用しています)

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
assignment_env python3 main.py -i <require: ex. data/input.mp4> -o <default: data/output.mp4> --half -s <int: skip rate, estimate every number frames>
```

### Option
i: str インプットファイル名  
o: str, default=data/output.mp4 アウトプットファイル名  
half: bool, default=False 半精度に変換する  
s: int, default=1 sフレーム毎に推論を行う

## Demo

実行結果はこちら  
5fps(リアルタイム実行可能)  
https://drive.google.com/file/d/1_2WKGrr3yRLuXFInS1okclUMEGi7ROPz/view?usp=share_link  
50fps(リアルタイム実行不可)  
https://drive.google.com/file/d/1DjR8s8053TfOFcfcNm5wjqFZI6J8IhXa/view?usp=share_link
