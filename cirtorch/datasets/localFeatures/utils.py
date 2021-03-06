import numpy as np
import cv2
from numpy.core.records import array


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def random_choice(array, size):
    rand = np.random.RandomState(1234)
    num_data = len(array)
    if num_data > size:
        idx = rand.choice(num_data, size, replace=False)
    else:
        idx = rand.choice(num_data, size, replace=True)
    return array[idx]


def rotateImage(image, angle):
    h, w = image.shape[:2]
    angle_radius = np.abs(angle / 180. * np.pi)
    cos = np.cos(angle_radius)
    sin = np.sin(angle_radius)
    tan = np.tan(angle_radius)
    scale_h = (h / cos + (w - h * tan) * sin) / h
    scale_w = (h / sin + (w - h / tan) * cos) / w
    scale = max(scale_h, scale_w)
    image_center = tuple(np.array(image.shape[1::-1]) / 2.)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    rotation = np.eye(4)
    rotation[:2, :2] = rot_mat[:2, :2]
    return result, rotation


def perspective_transform(img, param=0.001):
    h, w = img.shape[:2]
    random_state = np.random.RandomState(None)
    M = np.array([[1 - param + 2 * param * random_state.rand(),
                   -param + 2 * param * random_state.rand(),
                   -param + 2 * param * random_state.rand()],
                  [-param + 2 * param * random_state.rand(),
                   1 - param + 2 * param * random_state.rand(),
                   -param + 2 * param * random_state.rand()],
                  [-param + 2 * param * random_state.rand(),
                   -param + 2 * param * random_state.rand(),
                   1 - param + 2 * param * random_state.rand()]])

    dst = cv2.warpPerspective(img, M, (w, h))
    return dst, M


def generate_kpts(img, mode, num_pts):
    
    w, h = img.size
    img = np.asarray(img)

    # generate candidate query points
    if mode == 'random':
        kp_x = np.random.rand(num_pts) * (w - 1)
        kp_y = np.random.rand(num_pts) * (h - 1)
        coord = np.stack((kp_x, kp_y)).T

    elif mode == 'sift':
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        sift = cv2.xfeatures2d.SIFT_create(nfeatures=num_pts)
        
        keypoints = sift.detect(gray_img, None)
        
        coord = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
    
    #TODO: Add Superpoints as well alone + mixed 
    
    elif mode == 'mixed':
        kp_x = np.random.rand(1 * int(0.1 * num_pts)) * (w - 1)
        kp_y = np.random.rand(1 * int(0.1 * num_pts)) * (h - 1)
        kp_rand = np.stack((kp_x, kp_y)).T

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        sift = cv2.xfeatures2d.SIFT_create(nfeatures=int(0.9 * num_pts))

        keypoints = sift.detect(gray_img)
        # TODO:add response and superpoints
        kp_sift = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        
        if len(kp_sift) == 0:
            coord = kp_rand
        else:
            coord = np.concatenate((kp_rand, kp_sift), 0)

    else:
        raise Exception('Unknown type of keypoints')

    return coord

#TODO: use this function in algo, using geomtry pkgs
def prune_kpts(kpts, F, img_2_size, k1, k2, P, d_min, d_max):

    # compute the epipolar lines corresponding to img coordinates 1
    kpts_h = np.concatenate([kpts, np.ones_like(kpts[:, [0]])], axis=1).T  # 3xn
    epipolar_line = F.dot(kpts_h)  # 3xn
    epipolar_line /= np.clip(np.linalg.norm(epipolar_line[:2], axis=0), a_min=1e-10, a_max=None)  # 3xn

    # determine whether the epipolar lines intersect with the img 2
    h2, w2 = img_2_size
    corners = np.array([[0, 0, 1], [0, h2 - 1, 1], [w2 - 1, 0, 1], [w2 - 1, h2 - 1, 1]])  # 4x3
    dists = np.abs(corners.dot(epipolar_line))

    # if the epipolar line is far away from any image corners than sqrt(h^2+w^2)
    # it doesn't intersect with the image
    non_intersect = (dists > np.sqrt(w2 ** 2 + h2 ** 2)).any(axis=0)

    # determine if points in coord1 is likely to have correspondence in the other image by the rough depth range
    intrinsic1_4x4 = np.eye(4)
    intrinsic1_4x4[:3, :3] = k1
    intrinsic2_4x4 = np.eye(4)
    intrinsic2_4x4[:3, :3] = k2
    coord1_h_min = np.concatenate([d_min * kpts,
                                   d_min * np.ones_like(kpts[:, [0]]),
                                   np.ones_like(kpts[:, [0]])], axis=1).T
    coord1_h_max = np.concatenate([d_max * kpts,
                                   d_max * np.ones_like(kpts[:, [0]]),
                                   np.ones_like(kpts[:, [0]])], axis=1).T

    coord2_h_min = intrinsic2_4x4.dot(P).dot(np.linalg.inv(intrinsic1_4x4)).dot(coord1_h_min)
    coord2_h_max = intrinsic2_4x4.dot(P).dot(np.linalg.inv(intrinsic1_4x4)).dot(coord1_h_max)

    coord2_min = coord2_h_min[:2] / (coord1_h_min[2] + 1e-10)
    coord2_max = coord2_h_max[:2] / (coord1_h_max[2] + 1e-10)
    out_range = ((coord2_min[0] < 0) & (coord2_max[0] < 0)) | \
                ((coord2_min[1] < 0) & (coord2_max[1] < 0)) | \
                ((coord2_min[0] > w2 - 1) & (coord2_max[0] > w2 - 1)) | \
                ((coord2_min[1] > h2 - 1) & (coord2_max[1] > h2 - 1))

    ind_intersect = ~(non_intersect | out_range)

    return ind_intersect