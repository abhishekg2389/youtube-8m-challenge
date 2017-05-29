import tensorflow as tf
import glob as glob
import getopt
import sys
import cPickle as pkl
import numpy as np
import time
import os

opts, _ = getopt.getopt(sys.argv[1:],"",["stddevs_file_path=", "means_file_path=", "output_dir=", "input_dir=", "input_file=", "output_file=", "videos_done_filepath="])
for opt, arg in opts:
  if opt in ("--input_file"):
    input_file = arg
  if opt in ("--input_dir"):
    input_dir = arg
  if opt in ("--output_dir"):
    output_dir = arg
  if opt in ("--output_file"):
    output_file = arg
  if opt in ("--videos_done_filepath"):
    videos_done_filepath = arg
  if opt in ("--means_file_path"):
    means_file_path = arg
  if opt in ("--stddevs_file_path"):
    stddevs_file_path = arg
  if opt in ("--videos_done_filepath"):
    videos_done_filepath = arg

# filepaths to do
f = file(input_file, 'rb')
records_todo = pkl.load(f)
f.close()

# means
f = open(means_file_path, 'rb')
means = pkl.load(f)
f.close()

# stddevs
f = open(stddevs_file_path, 'rb')
stddevs = pkl.load(f)
f.close()

# videos done
f = file(videos_done_filepath, 'rb')
videos_done = pkl.load(f)
f.close()

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

for record in records_todo:
  if os.path.isfile(output_file):
    f = file(output_file, 'rb')
    records_done = pkl.load(f)
    f.close()
  else:
    records_done = {}
  
  if record in records_done:
    print(record + ' : Skipped')
    print(len(records_done)/float(len(records_todo)))
    continue
  
  filepaths_queue = tf.train.string_input_producer([input_dir+record], num_epochs=1)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filepaths_queue)
  features = tf.parse_single_example(serialized_example,features=features_format)

  with tf.Session() as sess:
    new_filepath = output_dir+record
    writer = tf.python_io.TFRecordWriter(new_filepath)
  
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    counter = 0
    try:
      while True:
        proc_features, = sess.run([features])
        counter += 1
        for feature_name in feature_names:
          proc_features[feature_name] = (proc_features[feature_name] - means[feature_name])/stddevs[feature_name]
        
        # writing tfrecord v1
        proc_features['video_id'] = [proc_features['video_id']]
        proc_features['video_length'] = [proc_features['video_length']]
        proc_features['labels'] = proc_features['labels'].values
        tf_features_format = {}
        for key, value in proc_features.items():
          if key == 'video_id':
            tf_features_format[key] = tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
          elif key == 'labels':
            tf_features_format[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=value))
          else:
            tf_features_format[key] = tf.train.Feature(float_list=tf.train.FloatList(value=value))
        example = tf.train.Example(features=tf.train.Features(feature=tf_features_format))
        writer.write(example.SerializeToString())
        videos_done[proc_features['video_id']+"_"+record] = 1
    except tf.errors.OutOfRangeError, e:
      coord.request_stop(e)
    finally:
      coord.request_stop()
      coord.join(threads)
      
      print(record + ' : Done')
      records_done[record] = 1
      print(len(records_done)/float(len(records_todo)))
      
      f = file(output_file, 'wb')
      pkl.dump(records_done, f, protocol=pkl.HIGHEST_PROTOCOL)
      f.close()
      
      f = file(videos_done_filepath, 'wb')
      pkl.dump(videos_done, f, protocol=pkl.HIGHEST_PROTOCOL)
      f.close()
      
      # writing tfrecord v1
      writer.close()
      print(time.time() - start_time)
