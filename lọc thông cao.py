import cv2
import numpy as np
#Đọc ảnh đầu vào
img1=cv2.imread('img.png', 0)
img= cv2.GaussianBlur(img1,(9,9),0)

#Bộ lọc 3x3
kernel2 = np.array([[0.0, -1.0, 0.0],
                   [-1.0, 4.0, -1.0],
                   [0.0, -1.0, 0.0]])
kernel2 = kernel2/(np.sum(kernel2) if np.sum(kernel2)!=0 else 1)

#Lọc nguồn ảnh cho thông thấp và thông cao
img_cao = cv2.filter2D(img1,-1,kernel2)
cv2.imshow("Anh Goc1", img)
cv2.imshow('Loc Thong Cao', img_cao)
cv2.waitKey()
cv2.destroyAllWindows()
