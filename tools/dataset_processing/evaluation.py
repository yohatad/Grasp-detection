import matplotlib.pyplot as plt
import numpy as np

from .grasp import GraspRectangles, detect_grasps

# The evaluation section uses IOU (Intersection over union) this one is set at 0.25 
def calculate_iou_match(acc, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None, threshold=0.25):

    if not isinstance(ground_truth_bbs, GraspRectangles):
        gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
    else:
        gt_bbs = ground_truth_bbs
    gs = detect_grasps(acc, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)
    for g in gs:
        if g.max_iou(gt_bbs) > threshold:
            return True
    else:
        return False