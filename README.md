# Automatic-3D-model-extraction-system-using-LiDAR-and-object-surface-attributes

MediaPipeを用いて姿勢推定を行い，

## 概要

このシステムはiPadに搭載されているLiDARとカメラを用いて現実の空間やオブジェクト(物体)を3Dスキャンし，その3DスキャンデータをBlenderとYOLOを用いて物体ごとに3Dデータを自動的に抽出するシステムです．基本的に3Dスキャンした3Dデータはすべてのオブジェクトが一体化していますが，抽出するオブジェクトを選択するとそのオブジェクト以外のポリゴンを削除していき，選択したオブジェクトだけが残ります．
![エラー](imge/split.png)

システムの流れとしてLiDARでスキャンした3DデータをBlenderに表示させ，その画面をキャプチャし仮想カメラに書き込み配信を行います．配信されている画像をPythonのYOLOを用いて物体検出を行いバウンディングボックス座標を取得します．Blender上でバウンディングボックス外にあるオブジェクトの頂点を選択し削除します．Blender上でオブジェクトを少し回転させ，再びその画面を使って物体検出を行う．オブジェクトが一回転したら終了です．．

![エラー](imge/abstract.png)

このシステムは物体検出を行う「Automatic-3D-model-extraction-system-using-LiDAR-and-object-surface-attributes/pyautogui_yolo.py」とBlender上のオブジェクト操作を行う「Automatic-3D-model-extraction-system-using-LiDAR-and-object-surface-attributes/object_split.py」の二つコードで動作しています．

## 期間
2ヶ月

## 言語
言語：Python

## 開発環境
個人開発でLinux環境で開発を行いました．

## 制作背景
大学の卒業研究として作成しました．3Dスキャンに興味を持ち，3Dスキャンが手軽に行えるiPhone13Proを購入して様々なものをスキャンしていました．そんな時に3Dスキャンデータをもっと扱いやすくする方法は無いかと考え，このシステムを開発しようとしました．
Pythonは最も慣れてい言語でしたが，Linux環境での開発やLinuxの仮想カメラなど初めて扱うものもあり，使いこなすまで時間がかかりました．しかし今ではLinuxが一番好きになりました．

## 動作風景URL
動作している様子を撮影しました．システムを起動するとBlender上の3DスキャンデータからYOLOを用いて物体検出を行います(今回は人のみを検出します)．物体検出の結果からPythonで自動制御マウスを用いたBlender上の抽出対象ではない3Dオブジェクト(人以外のオブジェクト)を削除します．その後，3Dデータを少しだけ回転させて，再び物体検出及び余分な3Dオブジェクトの削除を行います．これを3Dデータが一回転したら終了です．

編集有URL:[https://youtu.be/u3_v-uW22-Q](https://youtu.be/u3_v-uW22-Q)

編集無URL:[https://youtu.be/9OahinX9l5k](https://youtu.be/9OahinX9l5k)
