import albumentations as A
import cv2

img1 = cv2.cvtColor(cv2.imread('./data/train/1216.jpg'), cv2.COLOR_BGR2RGB)

res1 = A.Normalize(mean=(0.5964188, 0.4566936, 0.3908954), std=(0.2590655, 0.2314241, 0.2269535))
