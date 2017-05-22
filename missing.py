import tensorflow as tf
import glob as glob
import getopt
import sys
import cPickle as pkl
import numpy as np
import time
import os

opts, _ = getopt.getopt(sys.argv[1:],"",["chunk_file_path=", "comp_file_path=", "means_file_path=", "output_dir=", "input_dir="])
chunk_file_path = "../video_level_feat_v1/train*.tfrecord"
comp_file_path = "../video_level_feat_v1/train*.tfrecord"
means_file_path = "../video_level_feat_v1/means.pkl"
output_dir = "../video_level_feat_v1/"
input_dir = "../video_level_feat_v1/"
print(opts)
for opt, arg in opts:
  if opt in ("--chunk_file_path"):
    chunk_file_path = arg
  if opt in ("--comp_file_path"):
    comp_file_path = arg
  if opt in ("--means_file_path"):
    means_file_path = arg
  if opt in ("--output_dir"):
    output_dir = arg
  if opt in ("--input_dir"):
    input_dir = arg

# filepaths to do
f = file(chunk_file_path, 'rb')
records_chunk = pkl.load(f)
f.close()

# means
f = open(means_file_path, 'rb')
means = pkl.load(f)
f.close()

filepaths = [input_dir+x for x in records_chunk]
filepaths_queue = tf.train.string_input_producer(filepaths, num_epochs=1)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filepaths_queue)

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
features = tf.parse_single_example(serialized_example,features=features_format)

start_time = time.time()

for record in records_chunk:
  # filepaths done
  if os.path.isfile(comp_file_path):
    f = file(comp_file_path, 'rb')
    records_comp = pkl.load(f)
    f.close()
  else:
    records_comp = {}
  
  if record in records_comp:
    print(record + ' : Skipped')
    print(len(records_comp)/float(len(records_chunk)))
    continue
  
  new_filepath = output_dir+record
  writer = tf.python_io.TFRecordWriter(new_filepath)

  with tf.Session() as sess:
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
          if np.isnan(proc_features[feature_name]).sum() > 0:
            proc_features[feature_name][np.isnan(proc_features[feature_name])] = means[feature_name][np.isnan(proc_features[feature_name])]
          elif np.isinf(proc_features[feature_name]).sum() > 0:
            proc_features[feature_name][np.isinf(proc_features[feature_name])] = means[feature_name][np.isinf(proc_features[feature_name])]
        
        # writing tfrecord v1
        proc_features['video_id'] = [proc_features['video_id']]
        proc_features['video_length'] = [proc_features['video_length']]
        proc_features['labels'] = proc_features['labels'].values
        tf_features_format = {}
        for key, value in proc_features.items():
          print(key)
          if key == 'video_id':
            tf_features_format[key] = tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
          elif key == 'labels':
            tf_features_format[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=value))
          else:
            tf_features_format[key] = tf.train.Feature(float_list=tf.train.FloatList(value=value))
        example = tf.train.Example(features=tf.train.Features(feature=tf_features_format))
        writer.write(example.SerializeToString())
        
        if(counter%100000 == 1):
          print(counter)
    except tf.errors.OutOfRangeError, e:
      coord.request_stop(e)
    finally:
      coord.request_stop()
      coord.join(threads)
      
      print(record + ' : Done')
      records_comp[record] = 1
      print(len(records_comp)/float(len(records_chunk)))
      f = file(comp_file_path, 'wb')
      pkl.dump(records_comp, f, protocol=pkl.HIGHEST_PROTOCOL)
      f.close()

  # writing tfrecord v1
  writer.close()
  print(time.time() - start_time)
