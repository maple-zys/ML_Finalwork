import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
#
# data = pd.read_csv('./data/new_train.csv').values[:, 0]
# # print(np.array(Image.open((os.path.join('./data/train/', str(data[0]) + '.jpg'))).convert('RGB')).reshape(-1, 3).shape)
# # print(data)
# channel_1 = np.zeros([40000, data.shape[0]])
# channel_2 = np.zeros([40000, data.shape[0]])
# channel_3 = np.zeros([40000, data.shape[0]])
# # print(channel_2.shape)
# for i in tqdm(range(data.shape[0])):
#     image = np.array(Image.open((os.path.join('./data/train/', str(data[i]) + '.jpg'))).convert('RGB')).reshape(-1, 3)
#     channel_1[:, i] = image[:, 0]
#     channel_2[:, i] = image[:, 1]
#     channel_3[:, i] = image[:, 2]
# mean_1 = np.mean(channel_1)
# print('mean_1' + str(mean_1))
# mean_2 = np.mean(channel_2)
# print('mean_2' + str(mean_2))
# mean_3 = np.mean(channel_3)
# print('mean_3' + str(mean_3))
# std_1 = np.std(channel_1)
# print('std_1' + str(std_1))
# std_2 = np.std(channel_2)
# print('std_2' + str(std_2))
# std_3 = np.std(channel_3)
# print('std_3' + str(std_3))

print(152.08679828993056 / 255)
print(116.45686931076389 / 255)
print(99.67835131770833 / 255)
print(66.06171200386423 / 255)
print(59.01316991632885 / 255)
print(57.87316438526368 / 255)