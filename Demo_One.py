import os
import cv2
import json
import torch
import random
import numpy as np
import torch.nn.functional as F
from models import Get_model
from utils import py_max_match
from sklearn.cluster import KMeans

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def pad(image, stride=32):
    hasChange = False
    stdw = image.shape[1]
    if stdw % stride != 0:
        stdw += stride - (stdw % stride)
        hasChange = True 

    stdh = image.shape[0]
    if stdh % stride != 0:
        stdh += stride - (stdh % stride)
        hasChange = True

    if hasChange:
        newImage = np.zeros((stdh, stdw, 3), np.uint8)
        newImage[:image.shape[0], :image.shape[1], :] = image
        return newImage
    else:
        return image

def detect(model, image, key, threshold=0.3):
    mean = [0.408, 0.447, 0.47]
    std  = [0.289, 0.274, 0.278]

    image = pad(image)
    image = ((image / 255.0 - mean) / std).astype(np.float32)
    image = image.transpose(2, 0, 1)

    torch_image = torch.from_numpy(image)[None]
    torch_image = torch_image.cuda()

    out_dict = model(torch_image)
    hms      = out_dict['heatmap']
    offset   = out_dict['reg'].cpu().squeeze().data.numpy()
    whs      = out_dict['wh'].cpu().squeeze().data.numpy()

    Tscores, Tpoints = [], []
    for ind in range(1):
        hm      = hms[0:1, ind:ind+1]
        hm_pool = F.max_pool2d(hm, 3, 1, 1)
        scores, indices = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(500)
        hm_height, hm_width = hm.shape[2:]
        
        scores  = scores.squeeze()
        indices = indices.squeeze()
        ys      = list((indices / hm_width).int().data.numpy())
        xs      = list((indices % hm_width).int().data.numpy())
        scores  = list(scores.data.numpy())

        stride = 4
        tscores, tpoints = [], []
        for cx, cy, score in zip(xs, ys, scores):
            if score < threshold:
                break

            px   = (cx + offset[0,cy,cx])* stride
            py   = (cy + offset[1,cy,cx])* stride

            tscores.append(score)
            tpoints.append([px, py])
        Tscores.append(tscores)
        Tpoints.append(tpoints)
    return Tscores, Tpoints

def detect_image(model, file, key):
    image   = cv2.imread(file)
    Tscores, Tpoints = detect(model, image, key)

    for points in Tpoints:
        color = (random.randint(1,255), random.randint(1,255), random.randint(1,255))
        
        for t_point in points:
            cv2.circle(image, (int(t_point[0]), int(t_point[1])), 3, color, -1)

    return image

class Cfg_Opts(object):
    def __init__(self,):
        self.Model_Set                = {}
        self.Model_Set['Model_name']  = 'DLA_34'
        self.Model_Set['Head_dict']   = {'heatmap':1, 'reg': 2, 'wh': 2}
        
if __name__ == "__main__":
    cfg   = Cfg_Opts()
    model = Get_model(cfg)
    model.eval()
    model.cuda()
    model.load_param("work_space/DLA_34_2020-10-23-17-20-28/Epoch_best.pth")

    json_data = json.load(open('/data/Dataset/Chart/Task6/Val_Point.json','r'))
    
    for key in json_data.keys():
        if('hbox'  not in key and 'vbox' not in key and 'vertical_box' not in key):
            print(key)
            info_points = json_data[key]
            
            image = detect_image(model, '/data/Dataset/Chart/Task6/' + key, key)

            # for group in info_points:
                # color = (random.randint(1,255), random.randint(1,255), random.randint(1,255))
                # for point in group:
                    # cv2.circle(image, (int(point[0]), int(point[1])), 2, color, -1)
            
            cv2.imwrite('outs/' + key.split('/')[-1],image)
