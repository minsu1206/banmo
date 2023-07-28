import cv2
import numpy as np
from skimage.metrics import structural_similarity

if __name__ == '__main__':

    path1 = 'database/DAVIS_ama/JPEGImages/Full-Resolution/T_swing1/00136.jpg'
    path2 = 'database/DAVIS_ama/JPEGImages/Full-Resolution/T_swing1/00137.jpg'

    img1 = cv2.imread(path1, 0)
    img2 = cv2.imread(path2, 0)

    diff = cv2.subtract(img2, img1)
    cv2.imwrite("diff_137-136.png", diff)

    (score, diff) = structural_similarity(img1, img2, full=True)
    cv2.imwrite("diff_137-136_ssim.png", diff)
    
