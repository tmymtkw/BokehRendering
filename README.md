# Bokeh Effect Rendering Source Code
## 概要
入力画像、正解画像のあるImage-to-Image modelを学習するライブラリ  
EBB! (Everithing is Better with Bokeh!) Dataset用にカスタマイズされているので、  
他のデータセットを利用する場合は[汎用ライブラリ](https://github.com/tmymtkw/GeneralLibrary)を参照する。  


## ファイル構成
bokeh  
├─ config **学習条件のjsonファイル用ディレクトリ**  
│  └─ default.json  
├─ loss **損失関数用ディレクトリ**  
   └─ mse.py  
├─ metrics  **評価指標用ディレクトリ**  
├─ model  **学習モデル用ディレクトリ**  
│  └─ net.py  
├─ scripts  
│  └─ runner.py  
├─ util  
   ├─ data  
   ├─ logtool  
   ├─ parser  
   └─ check.py  


## 環境
システム  
> Linux OS  
> CUDA == 12.0.4  
> cuDNN == 8.7.0  
> gpu GeForce RTX 1080 ~ 3090  


ライブラリ  
> python == 3.10.4  
