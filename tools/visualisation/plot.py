import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from tools.dataset_processing.grasp import detect_grasps


def plot_results(fig, rgb_img, grasp_q_img, grasp_angle_img, depth_img=None, no_grasps=1, grasp_width_img=None):
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

     
    plt.ion()
    plt.clf()
    
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(rgb_img)
    ax.set_title('The Image')
    ax.axis('off')

    if depth_img is not None:    
        depth_img = depth_img.reshape(480,480)   
        ax = fig.add_subplot(2, 3, 4)
        ax.imshow(depth_img)
        ax.set_title('Depth')
        ax.axis('off')
     
    
    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
    ax.set_title('Grasp Postition')
    ax.axis('off')

    ax = fig.add_subplot(2, 3, 2)
    plot = ax.imshow(grasp_q_img, cmap='binary', vmin=0, vmax=0.5)
    #ax.set_title('Q')
    ax.axis('On')
    plt.colorbar(plot)


    plt.pause(0.1)
    fig.canvas.draw()