import pandas as pd
import numpy as np
import time

pred_csv = [
('video_level_logistic_model_skew_rgb_frame_n',0.544)
,('video_level_logistic_model_q3_rgb_audio_frame_n',0.752)
,('video_level_logistic_model_mean_rgb_audio_frame_n',0.755)
,('video_level_logistic_model_rgb_audio',0.743)
,('video_level_moe_model_7_rgb_audio',0.791)
]

pred_csv = [('/data2/pred/'+x[0]+'.csv', x[1]) for x in pred_csv]

rdrs = [None]*len(pred_csv)
for i, pc in enumerate(pred_csv):
    rdrs[i] = pd.read_csv(pc[0], chunksize=10000, skiprows=1, header=None)

tm = time.time()
for j in range(71):
    if j == 70:
        mat = np.zeros((640, 4716))
    else:
        mat = np.zeros((10000, 4716))
    for i, rdr in enumerate(rdrs):
        chnk = rdr.next()
        vids = chnk.iloc[:,0].values
        mat += chnk.iloc[:,1:].values*pred_csv[i][1]
        print(i,j)
    top_indices = np.argpartition(mat, -20)[:,-20:]
    lines = [[None]*20 for _ in range(len(mat))]
    for a in range(mat.shape[0]):
        for b in range(20):
            lines[a][b] = (top_indices[a,b], mat[a,top_indices[a,b]])
    lines = [sorted(line, key=lambda p: -p[1]) for line in lines]
    lines = [[vids[k], " ".join("%i %f" % pair for pair in line)] for k, line in enumerate(lines)]
    df = pd.DataFrame(lines)
    if(j==0):
        df.to_csv('/data2/pred/c.csv', index=False, header=['VideoId','LabelConfidencePairs'], mode='a')
    else:
        df.to_csv('/data2/pred/c.csv', index=False, header=False, mode='a')
    print(time.time() - tm)
