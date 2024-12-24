# XMem ONNX

> [!WARNING]
> This repogitory is work in progress.

## ▼ What's this?
This repository is the ONNX inference code for the XMem tracker model included in Track-Anything.

### Demo Result
![result](https://github.com/5PB-3-4/XMem_ONNX/blob/main/demo/result.png)

source: https://github.com/gaomingqi/Track-Anything/tree/master/test_sample

<br>

### Tested Environment
|name|version|
|----|-------|
|os|windows 10|
|cuda|11.8|
|python|3.10.15|
|uv|0.4.20|


<br><br>

## ▼ Get Started
### Get Code
```shell
git clone https://github.com/5PB-3-4/XMem_ONNX.git
```
<br>

### Check Dependency Library
Check out [requirement.txt](https://github.com/5PB-3-4/XMem_ONNX/blob/main/requirements.txt).

<br>

### Prepare Converted XMem ONNX File
Please convert models before Inference.
- [convert repository](https://github.com/5PB-3-4/XMem_Export/tree/main)
- [original pretrained checkpoint](https://github.com/hkchengrex/XMem/releases/tag/v1.0)

<br>

### Run Inference
```shell
# Run
cd XMem_ONNX
python eval_1object.py --encode_key export/XMem-encode_key.onnx --encode_value export/XMem-encode_value-m1.onnx --decode export/XMem-decode-m1.onnx

# Parser option
python eval_1object.py -h
```

> [!TIP]
> Only __one__ object can be cut from this repository. If you want to cut out multiple objects, rewrite this.
> (Or use __nightly__ repogitry for variable input shape of arrays)

```python
# eval_1object.py
_, best_mask = cv2.threshold(masks, 10, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
```

