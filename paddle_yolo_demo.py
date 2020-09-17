""" Copyright 2016-2022 by Bitmain Technologies Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import sys
import os
import argparse
import json
import numpy as np
import sophon.sail as sail
import ctypes
import struct

class PreProcessor:
  """ Preprocessing class.
  """
  def __init__(self, bmcv, scale):
    """ Constructor.
    """
    self.bmcv = bmcv
    self.ab = [x * scale for x in [1, 0, 1, 0, 1, 0]]

  def process(self, input, output, height, width):
    """ Execution function of preprocessing.
    Args:
      input: sail.BMImage, input image
      output: sail.BMImage, output data

    Returns:
      None
    """
    tmp = self.bmcv.vpp_resize(input, width, height)
    self.bmcv.convert_to(tmp, output, ((self.ab[0], self.ab[1]), \
                                       (self.ab[2], self.ab[3]), \
                                       (self.ab[4], self.ab[5])))

class Net:
  input_shapes_ = {}
  output_shapes_ = {}
  input_tensors_ = {}
  output_tensors_ = {}
  post_process_inputs_ = []
  output_names_ = []
  preprocessor_ = 0
  tpu_id_ = 0
  handle_ = 0
  img_dtype_ = 0
  engine_ = 0
  graph_name_ = 0
  bmcv_ = 0
  input_names_ = []
  lib_post_process_ = 0
  input_dtype = 0
  def __init__(self, bmodel_path, tpu_id):
    # init Engine
    Net.engine_ = sail.Engine(tpu_id)
    # load bmodel without builtin input and output tensors
    Net.engine_.load(bmodel_path)
    # get model info
    # only one model loaded for this engine
    # only one input tensor and only one output tensor in this graph
    Net.handle_ = Net.engine_.get_handle()
    Net.graph_name_ = Net.engine_.get_graph_names()[0]
    Net.input_names_ = Net.engine_.get_input_names(Net.graph_name_)
    input_dtype = 0
    Net.tpu_id_ = tpu_id
    for i in range(len(Net.input_names_)): 
      Net.input_shapes_[Net.input_names_[i]] = Net.engine_.get_input_shape(Net.graph_name_, Net.input_names_[i])
      input_dtype = Net.engine_.get_input_dtype(Net.graph_name_, Net.input_names_[i])
      alloc_flag = False
      if i == 1:
        alloc_flag = True
      input = sail.Tensor(Net.handle_, Net.input_shapes_[Net.input_names_[i]], input_dtype, alloc_flag, alloc_flag)
      Net.input_tensors_[Net.input_names_[i]] = input
      Net.input_dtype_ = input_dtype
    Net.output_names_ = Net.engine_.get_output_names(Net.graph_name_)
    for i in range(len(Net.output_names_)): 
      Net.output_shapes_[Net.output_names_[i]] = Net.engine_.get_output_shape(Net.graph_name_, Net.output_names_[i])
      output_dtype = Net.engine_.get_output_dtype(Net.graph_name_, Net.output_names_[i])
      output = sail.Tensor(Net.handle_, Net.output_shapes_[Net.output_names_[i]], output_dtype, True, True)
      Net.output_tensors_[Net.output_names_[i]] = output
    print (Net.input_shapes_)
    print (Net.output_shapes_)

    # set io_mode
    Net.engine_.set_io_mode(Net.graph_name_, sail.IOMode.SYSO)
    Net.bmcv_ = sail.Bmcv(Net.handle_)
    Net.img_dtype_ = Net.bmcv_.get_bm_image_data_format(input_dtype)
    scale = Net.engine_.get_input_scale(Net.graph_name_, Net.input_names_[0])
    scale *= 0.003922
    Net.preprocessor_ = PreProcessor(Net.bmcv_, scale)

    if os.path.exists('result_imgs') is False:
      os.system('mkdir -p result_imgs')

  def cut(obj, sec):
    return [obj[i : i + sec] for i in range(0, len(obj), sec)]

  def detect(self, video_path):
    decoder = sail.Decoder(video_path, True, Net.tpu_id_)
    frame_id = 0
    while 1:
      img = sail.BMImage()
      ret = decoder.read(Net.handle_, img)
      if ret != 0:
        print("Finished to read the video!");
        return
      # preprocess
      img_proceesed = sail.BMImage(Net.handle_, Net.input_shapes_[Net.input_names_[0]][2],
                        Net.input_shapes_[Net.input_names_[0]][3],
                        sail.Format.FORMAT_RGB_PLANAR, Net.img_dtype_)
      Net.preprocessor_.process(img,
          img_proceesed, Net.input_shapes_[Net.input_names_[0]][2], Net.input_shapes_[Net.input_names_[0]][3])
      Net.bmcv_.bm_image_to_tensor(img_proceesed, Net.input_tensors_[Net.input_names_[0]])

      # input tensor 2  image height and width
      img_shape = np.array((Net.input_shapes_[Net.input_names_[1]][0], 2))
      img_shape = img_shape.astype('float32')
      img_shape[0] = img.height()
      img_shape[1] = img.width()
      Net.input_tensors_[Net.input_names_[1]].update_data(img_shape)
      Net.input_tensors_[Net.input_names_[1]].sync_s2d()

      # inference bmodel
      Net.engine_.process(Net.graph_name_,
              Net.input_tensors_, Net.input_shapes_, Net.output_tensors_)

      # save results
      thresh_conf = 0.5
      result = Net.output_tensors_[Net.output_names_[0]].asnumpy()
      result = result.flatten()
      for i in range(0, result.shape[0], 6): 
        class_id = int(result[i])
        score = float(result[i + 1])
        if score < thresh_conf:
          continue
        left = int(result[i + 2])
        top = int(result[i + 3])
        right = int(result[i + 4])
        bottom = int(result[i + 5])
        print (left, top, right, bottom)
        Net.bmcv_.rectangle(img, left, top, right - left + 1, bottom - top + 1, (255, 0, 0), 3)
      Net.bmcv_.imwrite(os.path.join('result_imgs', str(frame_id) + '_video.jpg'), img)
      frame_id += 1

if __name__ == '__main__':
  """ A Yolo example using bm-ffmpeg to decode and bmcv to preprocess.
  """
  desc='decode (ffmpeg) + preprocess (bmcv) + inference (sophon inference)'
  PARSER = argparse.ArgumentParser(description=desc)
  PARSER.add_argument('--bmodel', default='', required=True)
  PARSER.add_argument('--input', default='', required=True)
  PARSER.add_argument('--tpu_id', default=0, type=int, required=False)
  ARGS = PARSER.parse_args()
  if not os.path.isfile(ARGS.input):
    print("Error: {} not exists!".format(ARGS.input))
    sys.exit(-2)
  yolo_net = Net(ARGS.bmodel, ARGS.tpu_id)
  yolo_net.detect(ARGS.input)
