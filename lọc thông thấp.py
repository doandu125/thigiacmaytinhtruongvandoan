import cv2
import numpy as np
img1=cv2.imread('img.png', 0)
img= cv2.GaussianBlur(img1,(9,9),0)
kernel = np.array([[1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1]])
kernel = kernel/sum(kernel)
img_thap = cv2.filter2D(img,-1,kernel)
cv2.imshow("Anh Goc1", img)
cv2.imshow('Loc Thong Thap', img_thap)
cv2.waitKey()
cv2.destroyAllWindows()