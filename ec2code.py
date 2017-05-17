import numpy as np # linear algebra
import tensorflow as tf
import glob
import collections as col
import itertools
import cPickle as pkl
import time
import os

### reading frame level tfrecords
def get_serialized_example(filepath):
  filename_queue = tf.train.string_input_producer([filepath], num_epochs=1)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  
  return(serialized_example)

def get_raw_frame_data(serialized_example):
  contexts, features = tf.parse_single_sequence_example(
    serialized_example,
    context_features={
      "video_id": tf.FixedLenFeature([], tf.string),
      "labels": tf.VarLenFeature(tf.int64)
    },
    sequence_features={
      "rgb" : tf.FixedLenSequenceFeature([], dtype=tf.string),
      "audio" : tf.FixedLenSequenceFeature([], dtype=tf.string)
    }
  )
  
  video_id = contexts['video_id']
  labels = contexts['labels']
  
  rgb = tf.reshape(tf.cast(tf.decode_raw(features['rgb'], tf.uint8), tf.float32),[-1, 1024])
  audio = tf.reshape(tf.cast(tf.decode_raw(features['audio'], tf.uint8), tf.float32),[-1, 128])
  
  return(video_id, labels, rgb, audio)

def get_processed_frame_data(rgb_frame, audio_frame, feature_list, concat_features=False):
  rgb_frame_trans = tf.transpose(rgb_frame, perm=[1, 0])
  audio_frame_trans = tf.transpose(audio_frame, perm=[1, 0])
  
  video_length = tf.shape(rgb_frame)[0]
  q0_rgb_frame = tf.reduce_min(rgb_frame, reduction_indices=0)
  q1_rgb_frame = tf.reduce_min(tf.nn.top_k(rgb_frame_trans, k = tf.to_int32(tf.scalar_mul(0.75, tf.to_float(video_length))), sorted=False).values, reduction_indices=1)
  q2_rgb_frame = tf.reduce_min(tf.nn.top_k(rgb_frame_trans, k = tf.to_int32(tf.scalar_mul(0.50, tf.to_float(video_length))), sorted=False).values, reduction_indices=1)
  q3_rgb_frame = tf.reduce_min(tf.nn.top_k(rgb_frame_trans, k = tf.to_int32(tf.scalar_mul(0.25, tf.to_float(video_length))), sorted=False).values, reduction_indices=1)
  q4_rgb_frame = tf.reduce_max(rgb_frame, reduction_indices=0)
  mean_rgb_frame = tf.reduce_mean(rgb_frame, reduction_indices=0)
  stddv_rgb_frame = tf.sqrt(tf.reduce_mean(tf.square(rgb_frame - mean_rgb_frame), reduction_indices=0))
  skew_rgb_frame = tf.div(tf.reduce_mean(tf.pow(rgb_frame - mean_rgb_frame, 3), reduction_indices=0), tf.pow(stddv_rgb_frame, 3))
  kurt_rgb_frame = tf.div(tf.reduce_mean(tf.pow(rgb_frame - mean_rgb_frame, 4), reduction_indices=0), tf.pow(stddv_rgb_frame, 4))
  
  iqr_rgb_frame = tf.subtract(q3_rgb_frame, q1_rgb_frame)
  rng_rgb_frame = tf.subtract(q4_rgb_frame, q0_rgb_frame)
  
  coeffvar_rgb_frame = tf.div(stddv_rgb_frame, mean_rgb_frame)
  efficiency_rgb_frame = tf.div(tf.square(stddv_rgb_frame), tf.square(mean_rgb_frame))
  midhinge_rgb_frame = tf.add(q3_rgb_frame, q1_rgb_frame)
  qntcoeffdisp_rgb_frame = tf.div(iqr_rgb_frame, midhinge_rgb_frame)
  
  # Mean Absolute Difference
  md_rgb_frame = tf.div(tf.reduce_sum(tf.abs(tf.matrix_band_part(tf.subtract(tf.expand_dims(rgb_frame_trans, 2), tf.expand_dims(rgb_frame_trans, 1)), 0, -1)), reduction_indices=[1,2]), tf.cast(tf.multiply(video_length, video_length-1), tf.float32))
  # Median Absolute Deviation around Median
  abs_dev_median = tf.transpose(tf.abs(tf.subtract(rgb_frame, q2_rgb_frame)), perm=[1,0])
  mean_abs_med_rgb_frame = tf.reduce_min(tf.nn.top_k(abs_dev_median, k = tf.to_int32(tf.scalar_mul(0.50, tf.to_float(video_length))), sorted=False).values, reduction_indices=1)
  # Mean Absolute Deviation around Mean
  mean_abs_mean_rgb_frame = tf.reduce_mean(tf.abs(tf.subtract(rgb_frame, mean_rgb_frame)), reduction_indices=0)
  # Mean Absolute Deviation around Median
  mean_abs_mean_rgb_frame = tf.reduce_mean(tf.abs(tf.subtract(rgb_frame, mean_rgb_frame)), reduction_indices=0)
  # Mean Absolute Deviation around Mode
  mean_abs_mean_rgb_frame = tf.reduce_mean(tf.abs(tf.subtract(rgb_frame, mean_rgb_frame)), reduction_indices=0)
  
  pairwise_man, _ = tf.unique(tf.reshape(tf.matrix_band_part(tf.reduce_sum(tf.abs(tf.subtract(tf.expand_dims(rgb_frame, 0), tf.expand_dims(rgb_frame, 1))), reduction_indices=[2]), 0, -1), [-1]))
  
  local_features = locals()
  if(concat_features):
    features = []
    for x in feature_list:
      if x != 'video_length':
        features.append(local_features[x])
      else:
        features.append(tf.cast(tf.convert_to_tensor([video_length]), tf.float32))
    features = tf.concat(features, 0)
  else:
    features = {feature : local_features[feature] for feature in feature_list}
  
  return(features)

def extract_video_features_from_frame_features(cluster_features=False):
  start_time = time.time()
  filepaths = glob.glob('/data1/frame_level_feat/train*.tfrecord')
  
  filepaths_completed = {}
  for filepath in filepaths:
    record = filepath.split('/')[-1]
    if os.path.isfile('/data/video_level_feat_v1/completed.pkl'):
      f = file('/data/video_level_feat_v1/completed.pkl', 'rb')
      records_done = pkl.load(f)
      f.close()
      if record in records_done:
        print(record + ' : Skipped')
        filepaths_completed[record] = 1
        print(len(filepaths_completed)/4096.0)
        continue
    
    serialized_example = get_serialized_example(filepath)
    raw_frame_data = get_raw_frame_data(serialized_example)
    
    feature_list = ['video_length', 'q0_rgb_frame', 'q1_rgb_frame', 'q2_rgb_frame', 'q3_rgb_frame', 'q4_rgb_frame', 'mean_rgb_frame', 'stddv_rgb_frame', 'skew_rgb_frame', 'kurt_rgb_frame']
    processed_frame_data = get_processed_frame_data(raw_frame_data[2], raw_frame_data[3], feature_list, concat_features=False)
    
    # df = []
    
    # writing tfrecord v1
    new_filepath = '/data/video_level_feat_v1/'+record
    writer = tf.python_io.TFRecordWriter(new_filepath)
    
    with tf.Session() as sess:
      init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
      sess.run(init_op)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord, sess=sess)
      try:
        match_counter = 100
        video_ids_processed = {}
        while True:
          video_id, labels, features = sess.run([raw_frame_data[0], raw_frame_data[1], processed_frame_data])
          
          # writing tfrecord v1
          features_to_write = {key : value if key != 'video_length' else [value] for key, value in features.items()}
          features_to_write['video_id'] = [video_id]
          
          tf_features_format = {}
          for key, value in features_to_write.items():
            if key != 'video_id':
              tf_features_format[key] = tf.train.Feature(float_list=tf.train.FloatList(value=value))
            else:
              tf_features_format[key] = tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
          example = tf.train.Example(features=tf.train.Features(feature=tf_features_format))
          writer.write(example.SerializeToString())
      except tf.errors.OutOfRangeError, e:
        coord.request_stop(e)
      finally:
        coord.request_stop()
        coord.join(threads)
        
        print(record + ' : Done')
        filepaths_completed[record] = 1
        print(len(filepaths_completed)/4096.0)
        f = file('/data/video_level_feat_v1/completed.pkl', 'wb')
        pkl.dump(filepaths_completed, f, protocol=pkl.HIGHEST_PROTOCOL)
        f.close()
    
    # writing tfrecord v1
    writer.close()
    print(time.time() - start_time)

num_files = len(glob.glob('/data/video_level_feat_v1/*'))
num_attempts = 0
while num_attempts < 3:
  if num_files == len(glob.glob('/data1/frame_level_feat/*')):
    num_attempts += 1
    print('Sleeping for '+str(100*num_attempts)+'s')
    time.sleep(100*num_attempts)
  else:
    num_files = len(glob.glob('/data/video_level_feat_v1/*'))
    extract_video_features_from_frame_features()
    num_attempts = 0
