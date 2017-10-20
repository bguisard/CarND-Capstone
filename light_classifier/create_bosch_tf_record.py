# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert the Bosch dataset to TFRecord for object_detection.

Example usage:
    ./python create_bosch_tf_record \
        --label_map_path=PATH_TO_DATASET_LABELS \
        --data_dir=PATH_TO_DATA_FOLDER \
        --output_path=PATH_TO_OUTPUT_FILE

This creates the test set:

      python create_bosch_tf_record.py  \
        --label_map_path='../data/bosch/bosch_label_map.pbtxt' \
        --data_dir='../data/bosch/' \
        --output_path='../data/bosch/tfrecords/' \
        --set='test'
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import yaml

import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw Bosch dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set or merged set.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', '../data/bosch/bosch_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS

SETS = ['train', 'validation', 'test']


def get_label_map_text(label_map_path):
    """Reads a label map and returns a dictionary of label names to display name.

    Args:
      label_map_path: path to label_map.

    Returns:
      A dictionary mapping label names to id.
    """
    label_map = label_map_util.load_labelmap(label_map_path)
    label_map_text = {}
    for item in label_map.item:
        label_map_text[item.name] = item.name
    return label_map_text


def data_to_tf_example(data,
                       label_map_dict,
                       label_map_text,
                       ignore_difficult_instances=False):
    """Converts YAML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      data: dict holding path to image and annotations

      label_map_dict: A map from string label names to integers ids.

      ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    img_path = data['img_path']

    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_png = fid.read()

    encoded_png_io = io.BytesIO(encoded_png)

    image = PIL.Image.open(encoded_png_io)

    if image.format != 'PNG':
        raise ValueError('Image format not PNG')

    key = hashlib.sha256(encoded_png).hexdigest()

    width = image.width
    height = image.height

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []

    for obj in data['annos']:
        difficult = False  #bool(int(obj['difficult']))
        if ignore_difficult_instances and difficult:
            continue

        difficult_obj.append(int(difficult))
        xmin.append(float(obj['x_min']) / width)
        ymin.append(float(obj['y_min']) / height)
        xmax.append(float(obj['x_max']) / width)
        ymax.append(float(obj['y_max']) / height)
        classes_text.append(label_map_text[obj['label']].encode('utf8'))
        classes.append(label_map_dict[obj['label']])
        truncated.append(int(obj['occluded']))
        poses.append(str('Frontal').encode('utf8'))  # Hardcoded to Frontal pose

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['img_path'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['img_path'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_png),
        'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example


def main(_):
    if FLAGS.set not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))

    data_dir = FLAGS.data_dir

    output_file = os.path.join(FLAGS.output_path, FLAGS.set + '.record')
    writer = tf.python_io.TFRecordWriter(output_file)

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    label_map_text = get_label_map_text(FLAGS.label_map_path)

    logging.info('Reading from Bosch dataset.')

    yamlmap_path = os.path.join(data_dir, FLAGS.set + '.yaml')

    #set_dir = os.path.join(data_dir, FLAGS.set)
    #examples_list = dataset_util.read_examples_list(examples_path)

    all_data = yaml.load(open(yamlmap_path, 'rb').read())

    for idx, data_point in enumerate(all_data):
        if idx % 100 == 0:
            logging.info('Converting  image %d of %d', idx, len(all_data))
        img_path = os.path.abspath(os.path.join(data_dir, data_point['path']))

        data = {}
        data['img_path'] = img_path
        data['annos'] = data_point['boxes']

        tf_example = data_to_tf_example(data, label_map_dict,
                                        label_map_text,
                                        FLAGS.ignore_difficult_instances)

        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()

