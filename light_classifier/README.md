### Instructions

#### I - Convert your dataset to TFRecord
Sample script to convert the FlickrLogos dataset

``` bash
python create_bosch_tf_record.py  \
  --label_map_path='../data/bosch/bosch_label_map.pbtxt' \
  --data_dir='../data/bosch/' \
  --output_path='../data/bosch/tfrecords/' \
  --set='test'
```

#### II - Train the model
``` bash
python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path='./models/in_training/rcnn_resnet_augmented.config' \
    --train_dir='./models/in_training/rcnn_resnet_augmented'
```

#### II - Evaluate the model

``` bash
python object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path='./models/in_training/rcnn_resnet_augmented.config' \
    --eval_dir='./models/in_training/rcnn_resnet_augmented/eval''
```

#### IV - "Publish" the model as a frozen inference graph

``` bash
python object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path='./models/in_training/rcnn_resnet_augmented.config' \
    --checkpoint_path='./models/in_training/rcnn_resnet_augmented/model.ckpt-511' \
    --inference_graph_path='./models/deployed/faster_rcnn_resnet101_50regions_bosch/frozen_inference_graph.pb'
```

#### V - Run tensorboard

``` bash
tensorboard --logdir='./models/in_training/rcnn_resnet_augmented/eval'
```

#### Obs:
Time to evaluate Faster RCNN on 518 images:
CPU:      33 minutes (too long to matter)
GTX-1070:  1 minute  (77s to be precise - 148ms per image or 6.7 fps)
