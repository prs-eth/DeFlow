# code portion from CamLiFlow 
# https://github.com/MCG-NJU/CamLiFlow

import torch
import torchvision
import numpy as np


def color_jitter(image1, image2, brightness, contrast, saturation, hue):
    assert image1.shape == image2.shape
    cj_module = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)

    images = np.concatenate([image1, image2], axis=0)
    images_t = torch.from_numpy(images.transpose([2, 0, 1]).copy())
    images_t = cj_module.forward(images_t / 255.0) * 255.0
    images = images_t.numpy().astype(np.uint8).transpose(1, 2, 0)
    image1, image2 = images[:image1.shape[0]], images[image1.shape[0]:]

    return image1, image2

def flip_sparse_depth(depth, image_h, image_w, flip_mode):
    assert flip_mode in ['lr', 'ud']
    image_x, image_y, depth = depth[..., 0], depth[..., 1], depth[..., 2]

    if flip_mode == 'lr':
        image_x = image_w - 1 - image_x
    else:
        image_y = image_h - 1 - image_y

    depth = np.concatenate([image_x[:, None], image_y[:, None], depth[:, None]], axis=-1)

    return depth

def flip_image(image, flip_mode):
    if flip_mode == 'lr':
        return np.flip(image, axis=2).copy()
    else:
        return np.flip(image, axis=1).copy()
    

def random_flip(input_dict, flip_mode):
    assert flip_mode in ['lr', 'ud']
    image1 = input_dict['imgT_1']
    image2 = input_dict['imgT_2']
    depth1 = input_dict['depth_t1']
    depth2 = input_dict['depth_t2']
    image_h, image_w = input_dict['input_size']
    
    if np.random.rand() < 0.5:  # do nothing
        return input_dict

    # flip images
    flipped_image1 = flip_image(image1, flip_mode)
    flipped_image2 = flip_image(image2, flip_mode)

    # flip point clouds
    flipped_depth1 = flip_sparse_depth(depth1, image_h, image_w, flip_mode)
    flipped_depth2 = flip_sparse_depth(depth2, image_h, image_w, flip_mode)

    input_dict['imgT_1'] = flipped_image1
    input_dict['imgT_2'] = flipped_image2
    input_dict['depth_t1'] = flipped_depth1
    input_dict['depth_t2'] = flipped_depth2
    
    return input_dict


def crop_image_depth(input_dict, crop_window):
    image1 = input_dict['imgT_1']
    image2 = input_dict['imgT_2']
    depth1 = input_dict['depth_t1']
    depth2 = input_dict['depth_t2']
    
    x1, y1, x2, y2 = crop_window
    
    # crop images
    cropped_image1 = image1[:, y1:y2, x1:x2].copy()
    cropped_image2 = image2[:, y1:y2, x1:x2].copy()
    
    # crop sparse depth
    d1_x, d1_y = depth1[..., 0], depth1[..., 1]
    d2_x, d2_y = depth2[..., 0], depth2[..., 1]
    
    crop_mask1 = np.where(np.logical_and(
        np.logical_and(d1_x > x1, d1_x < x2),
        np.logical_and(d1_y > y1, d1_y < y2)
    ))[0]
    crop_mask2 = np.where(np.logical_and(
        np.logical_and(d2_x > x1, d2_x < x2),
        np.logical_and(d2_y > y1, d2_y < y2)
    ))[0]
    
    cropped_depth1 = depth1[crop_mask1]
    cropped_depth2 = depth2[crop_mask2]
    
    cropped_depth1[..., 0] -= x1
    cropped_depth2[..., 0] -= x1
    cropped_depth1[..., 1] -= y1
    cropped_depth2[..., 1] -= y1
    
    input_dict['imgT_1'] = cropped_image1
    input_dict['imgT_2'] = cropped_image2
    input_dict['depth_t1'] = cropped_depth1
    input_dict['depth_t2'] = cropped_depth2
    input_dict['input_size'] = np.array([y2-y1, x2-x1], dtype=np.float32)
    
    return input_dict

def random_crop(input_dict, crop_size):
    
    image_h, image_w = input_dict['input_size']
    crop_w, crop_h = crop_size
    
    assert crop_w <= image_w and crop_h <= image_h
    
    # top left of the cropping window
    x1 = np.random.randint(low=0, high=image_w - crop_w + 1)
    y1 = np.random.randint(low=0, high=image_h - crop_h + 1)
    crop_window = [x1, y1, x1 + crop_w, y1 + crop_h]
    
    return crop_image_depth(input_dict, crop_window)