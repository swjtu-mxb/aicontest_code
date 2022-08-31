#train
python tools/train.py -c configs/yolov3/yolov3_mobilenet_v3_sub.yml
# python tools/train.py -c configs/yolov3/yolov3_darknet53_landslide.yml

#eval

#python tools/eval.py -c configs/yolov3/yolov3_darknet53_landslide.yml -o weights=output/yolov3_darknet53_landslide/best_model.pdparams

#python tools/eval.py -c configs/yolov3/yolov3_mobilenet_v3_landslide.yml -o  weights=output/yolov3_mobilenet_v3_landslide/best_model.pdparams

python tools/eval.py -c configs/yolov3/yolov3_mobilenet_v3_sub.yml  -o weights=output/yolov3_mobilenet_v3_sub/best_model.pdparams

#infer
#python tools/infer.py -c configs/yolov3/yolov3_darknet53_landslide.yml --infer_img=dataset/landslide/VOCdevkit/VOC2007/JPEGImages/18_1.jpg -o weights=output/yolov3_darknet53_landslide/best_model.pdparams
python tools/infer.py -c configs/yolov3/yolov3_mobilenet_v3_sub.yml --infer_img=../LandSlide/VOC2007/JPEGImages/8_4.jpg -o weights=output/yolov3_mobilenet_v3_sub/best_model.pdparams