# Real Time Text Detection

This repository contains the implementation of the real time text detection project for CS7643. The repository is adapted from the [YOLOv7](https://github.com/WongKinYiu/yolov7) with additional tools and tuning for training on the COCO-Text V2.0 dataset.

# Usage
## Training
To train the real time text detection project, you will need the following:
1. Download the COCO-Text V2.0 dataset from [here](https://bgshih.github.io/cocotext/). You will need both the annotations and the MS COCO 2014 images.
2. Run the convert-coco-to-yolo.ipynb to convert the COCO-Text V2.0 dataset to YOLO ready format for training.
3. Install the required packages from the `requirements.txt` file
4. In the python environment, we can now train the model. An example training command is:
`python train.py --workers 8 --device 0 --batch-size 16 --data data/coco-textbox.yaml --img 1280 --cfg cfg/training/yolov7-textbox.yaml --weights "yolov7-tiny.pt" --name yolov7-textbox-tiny-1280-rect --hyp data/hyp.scratch.textbox-tiny-1280-rect.yaml --epochs 50 --adam`

## Inference
To perform inference, you will need the following:
1. Install the required packages from the `requirements.txt` file.
2. Using the trained model weights or downloading the pretrained model weights, we can now run inference in the python environment. An example inference command is:
`python detect.py --weights runs/train/yolov7-textbox-tiny-1280/weights/best.pt --conf 0.25 --img-size 1280 --source coco-text/images/test/img18000.jpg --device 0`

# Summary of Results
| Model | Image Size | Special Arguments | mAP@0.5 | mAP@0.5:0.95 | FPS (GPU) | FPS (CPU) |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOv7-Tiny | 640 | | 0.5485 | 0.2791 | 81.2 ± 0.7 | 38.3 ± 0.2 |
| YOLOv7-Tiny | 640 | --rect | 0.1558 | 0.049 | 86.0 ± 0.3 | 38.8 ± 0.7 |
| YOLOv7-Tiny | 640 | –-multi-scale | 0.5168 | 0.2563 | 79.6 ± 0.9 | 38.1 ± 0.7 |
| YOLOv7-Tiny | 1280 | | 0.6821 | 0.405 | 44.6 ± 1.6 | 11.8 ± 0.1 |
| YOLOv7x | 640 | | 0.5951 | 0.3231 | 47.4 ± 0.5 | 5.0 ± 0.1 |
| YOLOv7-e6e | 640 | | 0.5883 | 0.3208 | n/a | n/a |

The YOLOv7-e6e model inference failed to run due to compute limitations.