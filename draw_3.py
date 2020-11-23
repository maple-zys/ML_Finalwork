import matplotlib.pyplot as plt
import cv2

plt.figure(figsize=(20, 10))
plt.subplot(251)
plt.imshow(cv2.cvtColor(cv2.imread('./data/train/1216.jpg'), cv2.COLOR_BGR2RGB))
plt.title('1216.jpg', size=20)
plt.subplot(252)
plt.imshow(cv2.cvtColor(cv2.imread('./data/train/11111.jpg'), cv2.COLOR_BGR2RGB))
plt.title('11111.jpg', size=20)
plt.subplot(253)
plt.imshow(cv2.cvtColor(cv2.imread('./data/train/1.jpg'), cv2.COLOR_BGR2RGB))
plt.title('1.jpg', size=20)
plt.subplot(254)
plt.imshow(cv2.cvtColor(cv2.imread('./data/train/18000.jpg'), cv2.COLOR_BGR2RGB))
plt.title('18000.jpg', size=20)
plt.subplot(255)
plt.imshow(cv2.cvtColor(cv2.imread('./data/train/15.jpg'), cv2.COLOR_BGR2RGB))
plt.title('15.jpg', size=20)
plt.subplot(256)
plt.imshow(cv2.cvtColor(cv2.imread('./data/train/107.jpg'), cv2.COLOR_BGR2RGB))
plt.title('107.jpg', size=20)
plt.subplot(257)
plt.imshow(cv2.cvtColor(cv2.imread('./data/train/116.jpg'), cv2.COLOR_BGR2RGB))
plt.title('116.jpg', size=20)
plt.subplot(258)
plt.imshow(cv2.cvtColor(cv2.imread('./data/train/3162.jpg'), cv2.COLOR_BGR2RGB))
plt.title('3162.jpg', size=20)
plt.subplot(259)
plt.imshow(cv2.cvtColor(cv2.imread('./data/train/9367.jpg'), cv2.COLOR_BGR2RGB))
plt.title('9367.jpg', size=20)
plt.subplot(2, 5, 10)
plt.imshow(cv2.cvtColor(cv2.imread('./data/train/16810.jpg'), cv2.COLOR_BGR2RGB))
plt.title('16810.jpg', size=20)
plt.savefig('./figure/samples.png')
plt.show()

