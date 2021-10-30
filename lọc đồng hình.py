import cv2
import numpy as np
#Đọc ảnh đầu vào
img1=cv2.imread('img.png', 0)
img= cv2.GaussianBlur(img1,(9,9),0)

#Phần lọc đồng hình
hh, ww = img.shape[:2]

img_log = np.log(np.float64(img), dtype=np.float64)
dft = np.fft.fft2(img_log, axes=(0,1))
dft_shift = np.fft.fftshift(dft)
# tạo vòng tròn màu đen trên nền trắng cho bộ lọc thông cao
radius = 13
mask = np.zeros_like(img, dtype=np.float64)
cy = mask.shape[0] // 2
cx = mask.shape[1] // 2
cv2.circle(mask, (cx,cy), radius, 1, -1)
mask = 1 - mask
# Mặt nạ làm mờ hàm Gauss
mask = cv2.GaussianBlur(mask, (47,47), 0)
# Áp dụng mặt nạ
dft_shift_filtered = np.multiply(dft_shift,mask)
back_ishift = np.fft.ifftshift(dft_shift_filtered)
img_back = np.fft.ifft2(back_ishift, axes=(0,1))
# Kết hợp phần thực và ảo để tạo lại độ lớn cho ảnh gốc
img_back = np.abs(img_back)
# Đảo ngược
img_homomorphic = np.exp(img_back, dtype=np.float64)
# Kết quả lọc đồng hình
img_dh = cv2.normalize(img_homomorphic, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC4)

cv2.imshow("Anh Goc1", img)
cv2.imshow("Loc Dong Hinh", img_dh)
cv2.waitKey(0)
cv2.destroyAllWindows()