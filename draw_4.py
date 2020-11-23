import albumentations as A
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.cvtColor(cv2.imread('./data/test/18318.jpg'), cv2.COLOR_BGR2RGB)

res1 = A.CoarseDropout(p=1, max_holes=5, max_height=25, max_width=25)(image=img)
res2 = A.HorizontalFlip(p=1)(image=img)
res3 = A.GaussNoise(var_limit=(5.0, 30.0), p=1)(image=img)
res4 = A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=1)(image=img)
res5 = A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1)(image=img)
img1 = res1['image'].astype(np.float32)
img2 = res2['image'].astype(np.float32)
img3 = res3['image'].astype(np.float32)
img4 = res4['image'].astype(np.float32)
img5 = res5['image'].astype(np.float32)

# plt.subplot(231)
# plt.imshow(img)
# plt.title('original picture')
# plt.subplot(232)
# plt.imshow(img1 / 255)
# plt.title('CoarseDrop')
# plt.subplot(233)
# plt.imshow(img2 / 255)
# plt.title('HorizontalFlip')
# plt.subplot(234)
# plt.imshow(img3 / 255)
# plt.title('GaussNoise')
# plt.subplot(235)
# plt.imshow(img4 / 255)
# plt.title('ShiftScaleRotate')
# plt.subplot(236)
# plt.imshow(img5 / 255)
# plt.title('HueSaturationValue')
# plt.savefig('./figure/albumentation.png')
# plt.show()

plt.imshow(img4 / 255)
plt.xticks([])
plt.yticks([])
plt.savefig('./figure/example3.png')
plt.show()
