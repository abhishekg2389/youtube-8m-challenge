import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import seaborn as sns
from IPython.display import YouTubeVideo
import matplotlib.pyplot as plt
import glob
import collections as col

from subprocess import check_output

check_output(["ls", "../video_level_feat"]).decode("utf8")

labels_df = pd.read_csv('../input/vocabulary.csv')
filenames = glob.glob("/Users/abhi/Desktop/kaggle/youtube-8m/video_level_feat/train*.tfrecord")

# labels anal
verticals = labels_df.Vertical1.values.tolist() + labels_df.Vertical2.values.tolist() + labels_df.Vertical3.values.tolist()
verticals_count = dict(col.Counter(verticals))
print("# Unique Verticals : "+str(len(verticals_count)))
print("# Nan Verticals    : "+str(verticals_count[np.nan]))
print("# Uknown Verticals : "+str(verticals_count['(Unknown)']))

verticals_count.pop(np.nan)
verticals_count.pop('(Unknown)')
pd.DataFrame.from_dict(verticals_count, orient='index').plot(kind='bar', title='Labels distribution over verticals')
