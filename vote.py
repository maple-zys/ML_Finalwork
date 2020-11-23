import pandas as pd
import os
import numpy as np


root = './vote/temp'
items = os.listdir(root)
temp = pd.read_csv('./vote/output_k0_tta5_best.csv', encoding='utf8')
answers = np.zeros([temp.shape[0], len(items)])
print(answers.shape)
for i, item in enumerate(items):
    answers[:, i] = pd.read_csv(os.path.join(root, item), encoding='utf8')['label'].values
answer = np.sum(answers, axis=1)
for i in range(answer.shape[0]):
    if answer[i] > 5:
        answer[i] = 1
    else:
        answer[i] = 0
print(answer[:20])
temp['label'] = answer.astype('int')
temp.to_csv('./vote/result/final___.csv', index=False)

