import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from PIL import Image

from hardware.device import get_device
from inference.post_process import post_process_output
from tools.data.Image_data import ImageData
from tools.visualisation.plot import plot_results


# Parameters
model_network = 'trained-models/epoch_48_iou_0.93'
RGB_direc = 'cornell/bottle1.png'
depth_direc = 'cornell/05/pcd0525d.tiff'
no_grasp = 1 
apply_depth = False
apply_RGB = True
    
if __name__ == '__main__':
    #Load image 
   
    image = Image.open(RGB_direc, 'r')
    rgb = np.array(image)
    image = Image.open(depth_direc, 'r')
    depth = np.expand_dims(np.array(image), axis=2)

    # Load Network
    net = torch.load(model_network)

    # Get the compute device
    
    device = get_device(False)

    img_data = ImageData(include_depth = apply_depth, include_rgb = apply_RGB)

    x, depth_img, rgb_img = img_data.get_data(rgb=rgb, depth=depth)
    
    with torch.no_grad():
        xc = x.to(device)
        pred = net.predict(xc)
        
        accu_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
      
        fig = plt.figure(figsize=(10, 10))
        plot_results(fig=fig,
                    rgb_img = img_data.get_rgb(rgb, False),
                    depth_img = np.squeeze(img_data.get_depth(depth)),
                    acc_img = accu_img,
                    grasp_angle_img = ang_img,
                    no_grasps = no_grasp,
                    grasp_width_img = width_img)
        fig.savefig('result.pdf')