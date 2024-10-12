# XMem Export

> [!WARNING]
> This repogitory is work in progress.

## ▼ What's this?
This repository is Inference code of XMem tracker model included in [Track-Anything](https://github.com/gaomingqi/Track-Anything/tree/master/tracker) from PyTorch to ONNX.

### Demo Result
![result](https://github.com/5PB-3-4/XMem_ONNX/blob/main/demo/result.png)

<br>

### Tested Environment
|name|version|
|----|-------|
|os|windows 10|
|python|3.10.14|
|uv|0.3.4|
|onnxruntime|1.18.0|
|numpy|1.26.4|
|opencv|4.10.0.84|

demo file link: https://github.com/gaomingqi/Track-Anything/tree/master/test_sample

<br><br>


## ▼ Get Started
### Get code
```shell
git clone https://github.com/5PB-3-4/XMem_ONNX.git
```
<br>

### Check Dependency Library
Check out [requirement.txt]() file.

<br>

### Prepare Converted XMem checkpoint file
Please convert models before Inference.

convert repository is [here](https://github.com/5PB-3-4/XMem_Export/tree/main)

Original pretrained model is [here](https://github.com/hkchengrex/XMem/releases/tag/v1.0).


<br>

### Run
```shell
cd XMem_ONNX
python eval_1object.py --encode_key export/XMem-encode_key.onnx --encode_value export/XMem-encode_value-m1.onnx --decode export/XMem-decode-m1.onnx
```

Parser option ->
``` python eval_1object.py -h ```

> [!TIP]
> Only __one__ object can be cut from this repository. If you want to cut out multiple objects, rewrite this.

```python
# eval_1object.py
_, best_mask = cv2.threshold(masks, 10, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
```


