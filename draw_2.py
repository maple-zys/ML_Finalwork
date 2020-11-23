import matplotlib.pyplot as plt
import pandas as pd


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2. - 0.1, 1.015*height, '%s' % int(height))

data = pd.read_csv('./data/train.csv').values[:, 1]
print(data.shape)
male, female = 0, 0
for i in range(data.shape[0]):
    if data[i] == 0:
        male += 1
    else:
        female += 1
name_list = ['male', 'female']
num_list = [male, female]
autolabel(plt.bar(range(len(num_list)), num_list, color='br', tick_label=name_list))
plt.xlabel('label')
plt.ylabel('number')
plt.title('label statistics')
plt.savefig('./figure/label statistics.png')
plt.show()
