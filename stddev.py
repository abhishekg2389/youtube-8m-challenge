import tensorflow as tf
import glob as glob
import getopt
import sys
import cPickle as pkl
import numpy as np
import time

opts, _ = getopt.getopt(sys.argv[1:],"",["input_dir=", "input_file=", "output_file=", "means_file_path="])
input_dir = ""
input_file = ""
output_file = ""
means_file_path = ""
for opt, arg in opts:
  if opt in ("--input_dir"):
    input_dir = arg
  if opt in ("--input_file"):
    input_file = arg
  if opt in ("--output_file"):
    output_file = arg
  if opt in ("--means_file_path"):
    means_file_path = arg

f = open(means_file_path, 'rb')
means_agg = pkl.load(f)
f.close()

f = open(input_file, 'rb')
filepaths = pkl.load(f)
f.close()
filepaths = [input_dir+x for x in filepaths]
filepaths_queue = tf.train.string_input_producer(filepaths, num_epochs=1)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filepaths_queue)

features_format = {}
stddev = {}
feature_names = []
for x in ['q0', 'q1', 'q2', 'q3', 'q4', 'mean', 'stddv', 'skew', 'kurt', 'iqr', 'rng', 'coeffvar', 'efficiency']:
    features_format[x + '_rgb_frame'] = tf.FixedLenFeature([1024], tf.float32)
    features_format[x + '_audio_frame'] = tf.FixedLenFeature([128], tf.float32)
    feature_names.append(str(x + '_rgb_frame'))
    feature_names.append(str(x + '_audio_frame'))
    stddev[x + '_rgb_frame'] = np.zeros((1024,))
    stddev[x + '_audio_frame'] = np.zeros((128,))

features_format['video_id'] = tf.FixedLenFeature([], tf.string)
features_format['labels'] = tf.VarLenFeature(tf.int64)
features_format['video_length'] = tf.FixedLenFeature([], tf.float32)
features = tf.parse_single_example(serialized_example,features=features_format)

start_time = time.time()
with tf.Session() as sess:
  init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
  sess.run(init_op)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  counter = 0
  try:
    while True:
      proc_features, = sess.run([features])
      for feature_name in feature_names:
        stddev[feature_name] += (proc_features[feature_name] - means_agg[feature_name])**2
      counter += 1
      if(counter%10000 == 1):
        print(counter)
        print(time.time() - start_time)
  except tf.errors.OutOfRangeError, e:
    coord.request_stop(e)
  finally:
    print(counter)
    coord.request_stop()
    coord.join(threads)

f = open(output_file, 'wb')
pkl.dump(stddev, f, protocol=pkl.HIGHEST_PROTOCOL)
pkl.dump(counter, f, protocol=pkl.HIGHEST_PROTOCOL)
f.close()
