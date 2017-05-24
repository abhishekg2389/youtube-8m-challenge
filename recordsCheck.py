import tensorflow as tf
import glob as glob
import getopt
import sys
import cPickle as pkl
import numpy as np
import time
import os

opts, _ = getopt.getopt(sys.argv[1:],"",["input_dir=", "input_file=", "output_file=", "start_from="])
input_dir = "/data/video_level_feat_v3/"
input_file = ""
output_file = ""
start_from = ""
print(opts)
for opt, arg in opts:
  if opt in ("--input_dir"):
    input_dir = arg
  if opt in ("--input_file"):
    input_file = arg
  if opt in ("--output_file"):
    output_file = arg
  if opt in ("--start_from"):
    start_from = int(float(arg))

f = open(input_file, 'rb')
filepaths = pkl.load(f)
f.close()

if(os.path.isfile(output_file)):
  f = open(output_file, 'rb')
  errors = pkl.load(f)
  counter = pkl.load(f) - 1
  f.close()
else:
  errors = []
  counter = start_from - 1

filepaths = [input_dir+x for x in filepaths]

features_format = {}
feature_names = []
for x in ['q0', 'q1', 'q2', 'q3', 'q4', 'mean', 'stddv', 'skew', 'kurt', 'iqr', 'rng', 'coeffvar', 'efficiency']:
    features_format[x + '_rgb_frame'] = tf.FixedLenFeature([1024], tf.float32)
    features_format[x + '_audio_frame'] = tf.FixedLenFeature([128], tf.float32)
    feature_names.append(str(x + '_rgb_frame'))
    feature_names.append(str(x + '_audio_frame'))

features_format['video_id'] = tf.FixedLenFeature([], tf.string)
features_format['labels'] = tf.VarLenFeature(tf.int64)
features_format['video_length'] = tf.FixedLenFeature([], tf.float32)

start_time = time.time()
counter = start_from-1

for filepath in filepaths[start_from-1:]:
  print(counter)
  counter += 1
  filepaths_queue = tf.train.string_input_producer([filepath], num_epochs=1)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filepaths_queue)
  
  features = tf.parse_single_example(serialized_example,features=features_format)
  with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
      while True:
          proc_features, = sess.run([features])
    except tf.errors.OutOfRangeError, e:
      coord.request_stop(e)
    except:
      print("ERROR : "+filepath)
      errors.append(filepath)
    finally:
      f = open(output_file, 'wb')
      pkl.dump(errors, f, protocol=pkl.HIGHEST_PROTOCOL)
      pkl.dump(counter, f, protocol=pkl.HIGHEST_PROTOCOL)
      f.close()
      
      print(time.time() - start_time)
      coord.request_stop()
      coord.join(threads)

f = open(output_file, 'wb')
pkl.dump(errors, f, protocol=pkl.HIGHEST_PROTOCOL)
pkl.dump(counter, f, protocol=pkl.HIGHEST_PROTOCOL)
f.close()
