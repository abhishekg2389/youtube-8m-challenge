import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import seaborn as sns
from IPython.display import YouTubeVideo
import matplotlib.pyplot as plt
import glob
import collections as col
import scipy.sparse as sps
import itertools
import cPickle as pkl
import time
import sklearn.cluster as skcluster

from subprocess import check_output

# check_output(["ls", "../video_level_feat"]).decode("utf8")

def analyze_labels():
  labels_df = pd.read_csv('../input/vocabulary.csv')
  train_labels_df = pd.read_csv('../input/train_labels.csv', header=None)
  
  ### labels anal
  verticals = labels_df.Vertical1.values.tolist() + labels_df.Vertical2.values.tolist() + labels_df.Vertical3.values.tolist()
  verticals_count = dict(col.Counter(verticals))
  print(verticals_count)
  print("# Unique Verticals : "+str(len(verticals_count)))
  print("# Nan Verticals    : "+str(verticals_count[np.nan]))
  print("# Uknown Verticals : "+str(verticals_count['(Unknown)']))
  
  # verticals correlation among labels
  verticals_idx_map = {x : i for i, x in enumerate(sorted(verticals_count.keys()))}
  verticals_corr_mat = np.zeros(shape=(len(verticals_count), len(verticals_count)), dtype=np.int)
  for i, row in labels_df.iterrows():
    verticals_corr_mat[verticals_idx_map[row['Vertical1']], verticals_idx_map[row['Vertical2']]] += 1
    verticals_corr_mat[verticals_idx_map[row['Vertical2']], verticals_idx_map[row['Vertical3']]] += 1
    verticals_corr_mat[verticals_idx_map[row['Vertical3']], verticals_idx_map[row['Vertical1']]] += 1
    verticals_corr_mat[verticals_idx_map[row['Vertical2']], verticals_idx_map[row['Vertical1']]] += 1
    verticals_corr_mat[verticals_idx_map[row['Vertical3']], verticals_idx_map[row['Vertical2']]] += 1
    verticals_corr_mat[verticals_idx_map[row['Vertical1']], verticals_idx_map[row['Vertical3']]] += 1
  
  verticals_corr_mat = np.delete(verticals_corr_mat, 0, 0)
  verticals_corr_mat = np.delete(verticals_corr_mat, 0, 1)
  verticals_corr_mat = pd.DataFrame(verticals_corr_mat)
  verticals_corr_mat.columns = sorted(verticals_idx_map.keys()[1:])
  verticals_corr_mat.index = sorted(verticals_idx_map.keys()[1:])
  sns.heatmap(verticals_corr_mat, linewidths=.5, annot=True)
  plt.yticks(rotation=0)
  plt.xticks(rotation=90)
  plt.show()
  
  # labels distribution over verticals
  verticals_count.pop(np.nan)
  verticals_count.pop('(Unknown)')
  pd.DataFrame.from_dict(verticals_count, orient='index').plot(kind='bar', title='Labels distribution over verticals')
  
  # labels distribution over samples
  train_labels_dump = col.Counter(map(int, ' '.join(train_labels_df.iloc[:,1].values).split(' ')))
  train_labels_dump = [train_labels_dump[i] for i in range(4716)]
  
  # labels correlation over samples
  train_labels_corr = sps.lil_matrix((4716, 4716), dtype=np.int)
  for i, labels in train_labels_df.iterrows():
    labels = map(int, labels[1].split(" "))
    for x, y in itertools.combinations(labels, 2):
      if (train_labels_corr[x, y] > 0):
        train_labels_corr[x,y] += 1
      else:
        train_labels_corr[x,y] = 1
    if(i%10000 == 1):
      print i

  np.save('train_labels_corr', train_labels_corr)

### reading video level tfrecords
def get_static_reader():
  filenames = glob.glob("/Users/abhi/Desktop/kaggle/youtube-8m/video_level_feat/train*.tfrecord")
  for filename in filenames:
    for example in tf.python_io.tf_record_iterator(filename):
        tf_example = tf.train.Example.FromString(example)
        print(tf_example)
        break
    break

### reading frame level tfrecords
def get_serialized_example(filenames):
  filename_queue = tf.train.string_input_producer([filenames[0]])
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

def get_processed_frame_data(rgb_frame, audio_frame):
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
  
  # return(video_length, )
  return({
    'video_length'     : video_length,
    'rgb_frame'        : rgb_frame,
    'audio_frame'      : audio_frame,
    'q0_rgb_frame'     : q0_rgb_frame,
    'q1_rgb_frame'     : q1_rgb_frame,
    'q2_rgb_frame'     : q2_rgb_frame,
    'q3_rgb_frame'     : q3_rgb_frame,
    'q4_rgb_frame'     : q4_rgb_frame,
    'mean_rgb_frame'   : mean_rgb_frame,
    'stddv_rgb_frame'  : stddv_rgb_frame,
    'skew_rgb_frame'   : skew_rgb_frame,
    'kurt_rgb_frame'   : kurt_rgb_frame,
    'pairwise_man'     : pairwise_man
  })

def extract_video_features_from_frame_features():
  start_time = time.time()
  filenames = glob.glob("../frame_level_feat/train*.tfrecord")
  serialized_example = get_serialized_example(filenames)
  raw_frame_data = get_raw_frame_data(serialized_example)
  processed_frame_data = get_processed_frame_data(raw_frame_data[2], raw_frame_data[3])
  
  with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    try:
      match_counter = 100
      video_ids_processed = {}
      while True:
      #for i in range(1):
        video_id, labels, features = sess.run([raw_frame_data[0], raw_frame_data[1], processed_frame_data])
        #print(features)
        for key, val in features.items():
          print(key + ' : ' + str(val.shape))
        print(video_id+" : "+str(features['video_length'])+" : "+str(labels.values))
        features['video_id'] = video_id
        features['labels'] = labels
        
        f = file('../frame_level_feat/frame_video_level_feat/'+video_id+'.pkl', 'wb')
        pkl.dump(features, f, protocol=pkl.HIGHEST_PROTOCOL)
        f.close()
        
        # mdl = skcluster.DBSCAN(eps=max(10000, np.mean(x[-1]) - np.std(x[-1])), min_samples=10, metric='manhattan', algorithm='auto', leaf_size=10, p=None, n_jobs=1)
        # mdl.fit(x[1])
        # print("#Clusters : "+str(np.unique(mdl.labels_)))
        # x_[i] = float((mdl.labels_ == -1).sum())/x[1].shape[0]
        # y_[i] = len(np.unique(mdl.labels_))
        # print(col.Counter(mdl.labels_))
        # break
        if(video_id in video_ids_processed):
          match_counter -= 1
          if(match_counter == 0):
            break
        else:
          video_ids_processed[video_id] = 1
      print(video_ids_processed)
    except tf.errors.OutOfRangeError, e:
      coord.request_stop(e)
    finally:
      coord.request_stop()
      coord.join(threads)
  print(time.time() - start_time)

# processing pickled features
def analyze_video_features_from_frame_features():
  filenames = glob.glob("../frame_level_feat/frame_video_level_feat/*.pkl")
  y = [0]*1000
  for i, filename in enumerate(filenames[:1000]):
    f = open(filename, 'rb')
    x = pkl.load(f)
    rgb_frame = x[0]
    audio_frame = x[0]
    y[i] = (rgb_frame.max(axis=0) == 255).sum()/1024.0

# time period distribution over samples


#

extract_video_features_from_frame_features()
