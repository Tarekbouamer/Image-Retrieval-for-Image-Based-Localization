import numpy as np
import cv2
from tqdm import tqdm

def warp_points(points, H):
    """
        Warp 2D points by an homography H.
    """

    n_points = points.shape[0]
    
    reproj_points = points.copy()[:, [1, 0]]
    reproj_points = np.concatenate([reproj_points, np.ones((n_points, 1))], axis=1)
    
    reproj_points = H.dot(reproj_points.transpose()).transpose()
    
    reproj_points = reproj_points[:, :2] / reproj_points[:, 2:]
    
    reproj_points = reproj_points[:, [1, 0]]
    
    return reproj_points

def get_desc_dist(features1, features2):
    """ Given two lists of descriptors compute the descriptor distance 
        between each pair of feature. """

    desc_dist = np.linalg.norm(features1['descriptors'][:, None] - features2['descriptors'][None], axis=2)
    
    return desc_dist

def nn_matcher(desc_dist):
    """ Given a matrix of descriptor distances n_points0 x n_points1,
        return a np.array of size n_points0 containing the indices of the
        closest points in img1, and -1 if the nearest neighbor is not mutual.
    """
    nearest1 = np.argmin(desc_dist, axis=1)
    nearest2 = np.argmin(desc_dist, axis=0)
    
    non_mutual = nearest2[nearest1] != np.arange(len(nearest1))
    
    nearest1[non_mutual] = -1
    
    return nearest1

def compute_H_estimation(m_kp1, m_kp2, real_H, img_shape, correctness_thresh=3):
    
    # Estimate the homography between the matches using RANSAC
    H, _ = cv2.findHomography(m_kp1[:, [1, 0]], m_kp2[:, [1, 0]], cv2.RANSAC)

    if H is None:
        return 0.

    # Compute the reprojection error of the four corners of the image
    corners = np.array([[0, 0, 1],
                        [img_shape[1] - 1, 0, 1],
                        [0, img_shape[0] - 1, 1],
                        [img_shape[1] - 1, img_shape[0] - 1, 1]])
    
    warped_corners = np.dot(corners, np.transpose(H))
    warped_corners = warped_corners / warped_corners[:, 2:]

    re_warped_corners = np.dot(warped_corners, np.transpose(np.linalg.inv(real_H)))
    re_warped_corners = re_warped_corners[:, :2] / re_warped_corners[:, 2:]
    
    mean_dist = np.mean(np.linalg.norm(re_warped_corners - corners[:, :2], axis=1))
    
    correctness = float(mean_dist <= correctness_thresh)

    return correctness

def compute_precision(kp_dist1, kp_dist2, correctness_threshold=3):
    """
        Compute the precision for a given threshold, averaged over the two images.
        kp_dist1 is the distance between the matched keypoints in img0 and the
        matched keypoints of img1 warped into img0. And vice-versa for kp_dist2.
    """
    precision = ((kp_dist1 <= correctness_threshold).mean()
                 + (kp_dist2 <= correctness_threshold).mean()) / 2
    
    return precision

def compute_recall(kp_dist, nearest_neighbor, correctness_threshold=3):
    """ 
        Compute the matching recall for a given threshold.
        kp_dist is the distance between all the keypoints of img0
        warped into img1 and all the keypoints of img1. 
    """
    
    mutual_matches = nearest_neighbor != -1
    
    # Get the GT closest point
    closest = np.argmin(kp_dist, axis=1)
    correct_gt = np.amin(kp_dist, axis=1) <= correctness_threshold

    corr_closest = nearest_neighbor[mutual_matches] == closest[mutual_matches]
    corr_matches = corr_closest * correct_gt[mutual_matches]

    if (np.sum(correct_gt) > 0) and (np.sum(mutual_matches) > 0):
        recall = np.sum(corr_matches) / np.sum(correct_gt)
    else:
        recall = 0.
    
    return recall

def run_descriptor_evaluation(config, dataloader):
    
    H_estimation = []
    precision = []
    recall = []
    mma = []

    for it, item in tqdm(enumerate(dataloader), total=len(dataloader)):
    
        H = item['H'][0].numpy().squeeze(0)
        
        features1 = np.load(item["img1_path"][0] + ".npz")
        features2 = np.load(item["img2_path"][0] + ".npz")
        
        # Match the features with mutual nearest neighbor filtering
        desc_dist = get_desc_dist(features1, features2)
            
        nearest_neighbor = nn_matcher(desc_dist)
            
        mutual_matches = nearest_neighbor != -1
        
        m_kp1 = features1['keypoints'][mutual_matches]
        m_kp2 = features2['keypoints'][nearest_neighbor[mutual_matches]]

        # Compute the descriptor metrics
        
        # Homography estimation
        H_estimation.append(compute_H_estimation(m_kp1, m_kp2, H, item['img1_size'][0], config['correctness_threshold']))

        # Precision
        kp_dist1 = np.linalg.norm(m_kp1 - warp_points(m_kp2, np.linalg.inv(H)), axis=1)
        kp_dist2 = np.linalg.norm(m_kp2 - warp_points(m_kp1, H), axis=1)

        precisions = []
        for threshold in range(1, config['max_mma_threshold'] + 1):
            precisions.append(compute_precision(kp_dist1, kp_dist2, threshold))

        precision.append(precisions[config['correctness_threshold']-1])
        mma.append(np.array(precisions))

        # Recall
        kp_dist = np.linalg.norm(warp_points(features1['keypoints'], H)[:, None] - features2['keypoints'][None], axis=2)
            
        recall.append(
            compute_recall(kp_dist, nearest_neighbor, config['correctness_threshold']))

    H_estimation = np.mean(H_estimation)
    precision = np.mean(precision)
    recall = np.mean(recall)
    mma = np.mean(np.stack(mma, axis=1), axis=1)

    return H_estimation, precision, recall, mma













