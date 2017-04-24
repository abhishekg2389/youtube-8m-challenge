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

from subprocess import check_output

check_output(["ls", "../video_level_feat"]).decode("utf8")

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
filenames = glob.glob("/Users/abhi/Desktop/kaggle/youtube-8m/video_level_feat/train*.tfrecord")
for filename in filenames:
    for example in tf.python_io.tf_record_iterator(filename):
        tf_example = tf.train.Example.FromString(example)
        print(tf_example)
        break
    break

### reading frame level tfrecords
filenames = glob.glob("../frame_level_feat/train*.tfrecord")
filename_queue = tf.train.string_input_producer([filenames[4]])
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

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

rgb_frame = tf.reshape(tf.cast(tf.decode_raw(features['rgb'], tf.uint8), tf.uint8),[-1, 1024])
audio_frame = tf.reshape(tf.cast(tf.decode_raw(features['audio'], tf.uint8), tf.uint8),[-1, 128])

with tf.Session() as sess:
  # Start populating the filename queue.
  init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
  sess.run(init_op)

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord, sess=sess)

  try:
    match_counter = 20
    video_ids_processed = {}
    while True:
      _contexts, _rgb_frame, _audio_frame = sess.run([contexts, rgb_frame, audio_frame])
      f = file('../frame_level_feat/frame_video_level_feat/'+_contexts['video_id']+'.pkl', 'wb')
      pkl.dump((_rgb_frame, _audio_frame), f, protocol=pkl.HIGHEST_PROTOCOL)
      f.close()
      print(_contexts['video_id']+" : "+str(len(_rgb_frame)))
      if(_contexts['video_id'] in video_ids_processed):
        match_counter -= 1
        if(match_counter == 0):
          break
      else:
        video_ids_processed[_contexts['video_id']] = 1
    print(video_ids_processed)

  except tf.errors.OutOfRangeError, e:
    coord.request_stop(e)
  finally:
    coord.request_stop()
    coord.join(threads)
