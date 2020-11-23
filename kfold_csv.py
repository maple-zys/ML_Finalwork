import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

np.random.seed(2020)
df = pd.read_csv('./data/train.csv', encoding='utf8')
sample_index = np.arange(len(df))
np.random.shuffle(sample_index)
images, labels = df.values[:, 0][sample_index], df.values[:, 1][sample_index]
data_size = len(df)

def get_kfold_ds(k=5):
    kf = KFold(n_splits=5)
    fold_num = 0
    for train_index, test_index in kf.split(images):
        train_images, valid_images = images[train_index], images[test_index]
        train_labels, valid_labels = labels[train_index], labels[test_index]
        pd.DataFrame({'id': train_images, 'label': train_labels}).to_csv('./data/train_{}.csv'.format(fold_num),
                                                                         index=False)
        pd.DataFrame({'id': valid_images, 'label': valid_labels}).to_csv('./data/valid_{}.csv'.format(fold_num),
                                                                         index=False)
        fold_num += 1

if __name__ == '__main__':
    get_kfold_ds(5)
