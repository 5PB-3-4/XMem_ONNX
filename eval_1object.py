import argparse
import numpy as np
import cv2
import time
import sys
from base_tracker import BaseTracker


parser = argparse.ArgumentParser(description='Welcome to XMem ONNX Infer v0.1')
parser.add_argument('--encode_key', required=True, help='file path of XMem key encoder model')
parser.add_argument('--encode_value', required=True, help='file path of XMem value encoder model')
parser.add_argument('--decode', required=True, help='file path of  XMem key decoder model')
parser.add_argument('--width', required=False, default=640, help='model input width. default is [ 640 ] ')
parser.add_argument('--height', required=False, default=480, help='model input height. default is [ 480 ] ')
args = parser.parse_args()
print()


# initialize BaseTracker
xmem_checkpoint = {
    "EncodeKey"   : args.encode_key,
    "EncoderValue": args.encode_value,
    "Decoder"     : args.decode
}
device = ['CUDAExecutionProvider', 'CPUExecutionProvider']
Btrack = BaseTracker(xmem_checkpoint, device)
Btrack.tracker.clear_memory()


# read Video File
video_path = 'demo/test-sample1.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fn = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
new_w = abs(int(args.width))
new_h = abs(int(args.height))
print(f'original frame size: (w={w}, h={h})')
print(f'resize frame size: (w={w}, h={h})')
print(f'frame num: {fn}')


# read mask
masks = cv2.imread('demo/test-sample1-1frame-mask.png', cv2.IMREAD_GRAYSCALE)
masks = cv2.resize(masks, dsize=(new_w, new_h))
_, best_mask = cv2.threshold(masks, 10, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


# debug
if False:
    cv2.namedWindow('first mask', cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow("first mask", int(new_w*0.8), int(new_h*0.8))
    cv2.imshow("first mask", best_mask)

    ret, frame = cap.read()
    if ret:
        cv2.namedWindow('first frame', cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow("first frame", int(new_w*0.8), int(new_h*0.8))
        cv2.imshow("first frame", cv2.resize(frame, dsize=(new_w, new_h)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("failed to read frame ...")
    cap.release()
    sys.exit(0)


# tracking
cv2.namedWindow('result', cv2.WINDOW_GUI_NORMAL)
# w:h = nw:640 or 640:nh
if (w/h) < 1:
    cv2.resizeWindow("result", int(w/h*640), 640)
else:
    cv2.resizeWindow("result", 640, int(h/w*640))
first_frame = True
print("start.")
# while True:
for i in range(fn):
    print("===================================")
    print(f'frame #{i}')
    
    try:
        ret, frame = cap.read()
        if ret is False:
            raise IOError
        frame = cv2.resize(frame, dsize=(new_w, new_h))

        # eval
        start = time.perf_counter()
        if first_frame:
            mask, prob, painted_frame = Btrack.track(frame, best_mask)
            first_frame = False
        else:
            mask, prob, painted_frame = Btrack.track(frame)
        
        print('elapsed time: {:.3f} sec'.format(time.perf_counter() - start))

        best_mask = cv2.normalize(src=mask, dst=None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        best_mask = cv2.convertScaleAbs(best_mask, dst=None, alpha=255.0, beta=0.0)

        # show result
        # cvtImg = np.hstack((frame, cv2.cvtColor(best_mask, cv2.COLOR_GRAY2BGR)))
        # cv2.imshow("result", cv2.resize( cvtImg, dsize=(w, int(h/2)) ))
        cv2.imshow("result", painted_frame)
        if cv2.waitKey(1)==ord('q'):
            break
    except KeyboardInterrupt:
        break
    except Exception:
        raise

print("===================================")
print("\n end program \n")
