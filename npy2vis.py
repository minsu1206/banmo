"""
This code is only use for active sampled points -> trimesh
"""

import numpy as np
import trimesh
import matplotlib.pyplot as plt

def draw_np_ray(xys, reference_img):

    xys = xys.reshape(2, -1, 2)
    xys = xys[0]
    for (x, y) in xys:
        cv2.circle(reference_img, (int(x),int(y)), 1, (255 ,0, 0), thickness=1)

    cv2.imwrite("visualize/seed3-swing-ft2_epoch_latest_101_xrange_y30_xfrom20to40_active_sampling2.png", reference_img)

if __name__ == "__main__":
    from PIL import Image
    import cv2
    xys = np.load('visualize/seed3-swing-ft2_epoch_latest_101_xrange_y30_xfrom20to40_active_sampling.npy')

    refer = cv2.imread('visualize/seed3-swing-ft2_epoch_latest_101_xrange_y30_xfrom20to40_vis_2d_cv2.png')
    # refer = cv2.resize(refer)
    print(refer.shape)
    refer *= 255
    # cv2.imwrite("temp.png", refer)
    # exit()

    draw_np_ray(xys, refer)

