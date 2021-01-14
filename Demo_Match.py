import os
import cv2
import json
import torch
import numpy as np
import torch.nn.functional as F
from models import Get_model
from cfg import Cfg_Opts
from utils import py_max_match

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

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
    tags     = out_dict['tag'].cpu().squeeze().data.numpy()
    aes      = out_dict['ae'].cpu().squeeze().data.numpy()
    
    # cv2.imwrite('outs/' + key.split('/')[-1] + 'x.jpg', tags[0]*255)
    # cv2.imwrite('outs/' + key.split('/')[-1] + 'y.jpg', tags[1]*255)

    Tscores, Tpoints = [], []
    for ind in range(2):
        hm      = hms[0:1, ind:ind+1]
        hm_pool = F.max_pool2d(hm, 3, 1, 1)
        scores, indices = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
        hm_height, hm_width = hm.shape[2:]
        
        scores  = scores.squeeze()
        indices = indices.squeeze()
        ys      = list((indices / hm_width).int().data.numpy())
        xs      = list((indices % hm_width).int().data.numpy())
        scores  = list(scores.data.numpy())
       # offset  = offset.cpu().squeeze().data.numpy() ###

        stride = 4
        tscores, tpoints = [], []
        for cx, cy, score in zip(xs, ys, scores):
            if score < threshold:
                break

            px   = (cx + offset[0,cy,cx])* stride
            py   = (cy + offset[1,cy,cx])* stride
            tagx = (cx + tags[0,cy,cx])* stride
            tagy = (cx + tags[1,cy,cx])* stride
            ae   = aes[cy, cx]
            
            tscores.append(score)
            tpoints.append([px, py, ae])
        Tscores.append(tscores)
        Tpoints.append(tpoints)
    return Tscores, Tpoints, aes

def detect_image(model, file, G_points,key):
    image = cv2.imread(file)
    Tscores, Tpoints, Tags = detect(model, image, key)
    
    Tpoints_x, Tpoints_y = Tpoints[0], Tpoints[1]
    Gpoints_x, Gpoints_y = [],[]
    
    for point in G_points[0]:
        gx = (point[2]['x0']+point[2]['x1']+point[2]['x2']+point[2]['x3'])/4
        gy = (point[2]['y0']+point[2]['y1']+point[2]['y2']+point[2]['y3'])/4
        ae = Tags[int(gy/4), int(gx/4)]
        Gpoints_x.append([gx, gy, ae])
        
    for point in G_points[1]:
        gx = (point[2]['x0']+point[2]['x1']+point[2]['x2']+point[2]['x3'])/4
        gy = (point[2]['y0']+point[2]['y1']+point[2]['y2']+point[2]['y3'])/4
        ae = Tags[int(gy/4), int(gx/4)]
        Gpoints_y.append([gx, gy, ae])

    Diff_X = []
    for g_point in Gpoints_x:
        g_ae = g_point[2]
        diff_line = []
        for t_point in Tpoints_x:
            diff_line.append(abs(g_ae-t_point[2]))
        Diff_X.append(diff_line)
    X_Indexs = py_max_match(Diff_X)

    Diff_Y = []
    for g_point in Gpoints_y:
        g_ae = g_point[2]
        diff_line = []
        for t_point in Tpoints_y:
            diff_line.append(abs(g_ae-t_point[2]))
        Diff_Y.append(diff_line)
    Y_Indexs = py_max_match(Diff_Y)

    for g_id, t_id in X_Indexs:
        g_point = Gpoints_x[g_id]
        t_point = Tpoints_x[t_id]
        cv2.circle(image, (int(g_point[0]), int(g_point[1])), 2, (255, 0, 0), -1)
        cv2.circle(image, (int(t_point[0]), int(t_point[1])), 2, (0, 0, 255), -1)
        cv2.line(image, (int(g_point[0]), int(g_point[1])), (int(t_point[0]), int(t_point[1])), [0,125,255],1)

    for g_id, t_id in Y_Indexs:
        g_point = Gpoints_y[g_id]
        t_point = Tpoints_y[t_id]
        cv2.circle(image, (int(g_point[0]), int(g_point[1])), 2, (255, 0, 0), -1)
        cv2.circle(image, (int(t_point[0]), int(t_point[1])), 2, (0, 255, 0), -1)
        cv2.line(image, (int(g_point[0]), int(g_point[1])), (int(t_point[0]), int(t_point[1])), [0,125,255],1)
        
    return image

if __name__ == "__main__":
    cfg   = Cfg_Opts()
    model = Get_model(cfg)
    model.eval()
    model.cuda()
    model.load_param("work_space/DLA_34_2020-10-12-10-34-12/Epoch_best.pth")

    json_data = json.load(open('/data/Dataset/Chart/Chart/ATest.json','r'))
    for key in json_data.keys():
        data_dict = json_data[key]
        x_axis    = data_dict['x-axis']
        y_axis    = data_dict['y-axis']
        
        image     = detect_image(model, '/data/Dataset/Chart/Chart/' + key, [x_axis, y_axis], key)
        
        # for point in x_axis:
            # cv2.circle(image, (int(point[0]), int(point[1])), 2, (255, 125, 0), -1)
        # for point in y_axis:
            # cv2.circle(image, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)
        cv2.imwrite('outs/' + key.split('/')[-1],image)
