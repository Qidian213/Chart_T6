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

def dist_points(point_a, point_b):
    dist = ((point_a[0]-point_b[0])**2 + (point_a[1]-point_b[1])**2)**0.5
    return dist
    
def oks_nms(dscores, dpoints, or_h, or_w, THRESHOLD=0.025):
    diag = ((or_h ** 2) + (or_w ** 2)) ** 0.5
    dth  = THRESHOLD * diag
    
    Tscores, Tpoints = [], []
    for scores, points in zip(dscores, dpoints):
        scores = np.array(scores)
        points  = np.array(points)

        sc_indexs = scores.argsort()[::-1]

        scores = scores[sc_indexs]
        points = points[sc_indexs]

        sc_keep = []
        point_keep = []

        flags = [0] * len(scores)
        for index, sc in enumerate(scores):
            if flags[index] != 0:
                continue

            sc_keep.append(scores[index])
            point_keep.append(points[index])

            for j in range(index + 1, len(scores)):
                if flags[j] == 0 and dist_points(points[index], points[j]) < dth:
                    flags[j] = 1
        Tscores.append(sc_keep)
        Tpoints.append(point_keep)
    return Tscores, Tpoints
    
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
    or_h, or_w = image.shape[:2]

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
    for ind in range(5):
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
            w    = whs[0, cy, cx]* stride
            h    = whs[1, cy, cx]* stride
            
            tscores.append(score)
            tpoints.append([px, py, w, h])
        Tscores.append(tscores)
        Tpoints.append(tpoints)
    return oks_nms(Tscores, Tpoints, or_h, or_w)

def detect_image(model, file, key):
    image   = cv2.imread(file)
    Tscores, Tpoints = detect(model, image, key)

    for points in Tpoints:
        color = (random.randint(1,255), random.randint(1,255), random.randint(1,255))
        
        for t_point in points:
            cv2.circle(image, (int(t_point[0]), int(t_point[1])), 3, color, -1)
            if('hbox' in key):
                st_y, et_y = int(t_point[1] - t_point[3]/2), int(t_point[1] + t_point[3]/2)
                cv2.line(image, (int(t_point[0]), st_y), (int(t_point[0]), et_y), color,1)
            else:
                st_x, et_x = int(t_point[0] - t_point[2]/2), int(t_point[0] + t_point[2]/2)
                cv2.line(image, (st_x, int(t_point[1])), (et_x, int(t_point[1])), color,1)

    return Tpoints,image

class Cfg_Opts(object):
    def __init__(self,):
        self.Model_Set                = {}
        self.Model_Set['Model_name']  = 'DLA_34'
        self.Model_Set['Head_dict']   = {'heatmap':5, 'reg': 2, 'tag': 1}

if __name__ == "__main__":
    cfg   = Cfg_Opts()
    model = Get_model(cfg)
    model.eval()
    model.cuda()
    model.load_param("work_space/DLA_34_2020-10-19-14-16-49/Epoch_best.pth")

    Synthetic_Jsons = '/data/Dataset/Chart/ICPR_Train/JSONs/'
    Synthetic_Imgs  = '/data/Dataset/Chart/ICPR_Train/Charts/'
    
    PMC_Jsons = '/data/Dataset/Chart/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/annotations/'
    PMC_Imgs  = '/data/Dataset/Chart/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.21/images/'

    Save_dir  = 'results/'
    
    VAL_DATA = json.load(open('/data/Dataset/Chart/Task6/Val_Point.json','r'))
    
    match_th = 0.02
    for key in VAL_DATA.keys():
        if('Synthetic' in key): 
            if('hbox' in key or 'vbox' in key):   ### line, scatter
                print(key)
                json_path = Synthetic_Jsons + key.split('/')[-2] + '/' + key.split('/')[-1].replace('png', 'json')
                img_path  = Synthetic_Imgs  + key.split('/')[-2] + '/' + key.split('/')[-1]
                
                dst_dir = Save_dir + key.split('/')[0] + '/' + key.split('/')[1]

                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                    
                dst_path = dst_dir + '/' + key.split('/')[-1].replace('png', 'json')
                
                Tpoints, image = detect_image(model, img_path, key)
                or_h, or_w    = image.shape[:2]
                
                First_Points  = Tpoints[0]
                Max_Points    = Tpoints[1]
                Median_Points = Tpoints[2]
                Min_Points    = Tpoints[3]
                Third_Points  = Tpoints[4]
                
                json_data    = json.load(open(json_path,'r'))
                
                if('hbox' in key):
                    result = []
                    task4_inputs = json_data['task6']['input']['task4_output']['axes']['x-axis']
                    for point_dict in task4_inputs:
                        x_base, y_base = point_dict['tick_pt']['x'], point_dict['tick_pt']['y']
                        
                        group_dict = {}
                        
                        min_dist  = 1000
                        min_point = None
                        for point in First_Points:
                            dist = abs(y_base - point[1])
                            if(dist < min_dist):
                                min_dist  = dist
                                min_point = point
                        if(min_dist< or_h*match_th):
                            group_dict['first_quartile'] = {'x': min_point[0],'y':min_point[1]}
                    
                        min_dist  = 1000
                        min_point = None
                        for point in Max_Points:
                            dist = abs(y_base - point[1])
                            if(dist < min_dist):
                                min_dist  = dist
                                min_point = point
                        if(min_dist< or_h*match_th):
                            group_dict['max'] = {'x': min_point[0],'y':min_point[1]}
                    
                        min_dist  = 1000
                        min_point = None
                        for point in Median_Points:
                            dist = abs(y_base - point[1])
                            if(dist < min_dist):
                                min_dist  = dist
                                min_point = point
                        if(min_dist< or_h*match_th):
                            group_dict['median'] = {'x': min_point[0],'y':min_point[1]}

                        min_dist  = 1000
                        min_point = None
                        for point in Min_Points:
                            dist = abs(y_base - point[1])
                            if(dist < min_dist):
                                min_dist  = dist
                                min_point = point
                        if(min_dist< or_h*match_th):
                            group_dict['min'] = {'x': min_point[0],'y':min_point[1]}
                    
                        min_dist  = 1000
                        min_point = None
                        for point in Third_Points:
                            dist = abs(y_base - point[1])
                            if(dist < min_dist):
                                min_dist  = dist
                                min_point = point
                        if(min_dist< or_h*match_th):
                            group_dict['third_quartile'] = {'x': min_point[0],'y':min_point[1]}
                    
                        if(len(group_dict.keys()) >=3):
                            result.append(group_dict)
                    
                    json_data['task6']['output']['visual elements']['boxplots'] = result
                    json.dump(json_data, open(dst_path, 'w'), indent=4)

                if('vbox' in key):
                    result = []
                    task4_inputs = json_data['task6']['input']['task4_output']['axes']['x-axis']
                    for point_dict in task4_inputs:
                        x_base, y_base = point_dict['tick_pt']['x'], point_dict['tick_pt']['y']
                        
                        group_dict = {}
                        
                        min_dist  = 1000
                        min_point = None
                        for point in First_Points:
                            dist = abs(x_base - point[0])
                            if(dist < min_dist):
                                min_dist  = dist
                                min_point = point
                        if(min_dist< or_w*match_th):
                            group_dict['first_quartile'] = {'x': min_point[0],'y':min_point[1]}
                    
                        min_dist  = 1000
                        min_point = None
                        for point in Max_Points:
                            dist = abs(x_base - point[0])
                            if(dist < min_dist):
                                min_dist  = dist
                                min_point = point
                        if(min_dist< or_w*match_th):
                            group_dict['max'] = {'x': min_point[0],'y':min_point[1]}
                    
                        min_dist  = 1000
                        min_point = None
                        for point in Median_Points:
                            dist = abs(x_base - point[0])
                            if(dist < min_dist):
                                min_dist  = dist
                                min_point = point
                        if(min_dist< or_w*match_th):
                            group_dict['median'] = {'x': min_point[0],'y':min_point[1]}

                        min_dist  = 1000
                        min_point = None
                        for point in Min_Points:
                            dist = abs(x_base - point[0])
                            if(dist < min_dist):
                                min_dist  = dist
                                min_point = point
                        if(min_dist< or_w*match_th):
                            group_dict['min'] = {'x': min_point[0],'y':min_point[1]}
                    
                        min_dist  = 1000
                        min_point = None
                        for point in Third_Points:
                            dist = abs(x_base - point[0])
                            if(dist < min_dist):
                                min_dist  = dist
                                min_point = point
                        if(min_dist< or_w*match_th):
                            group_dict['third_quartile'] = {'x': min_point[0],'y':min_point[1]}
                        
                        if(len(group_dict.keys()) >=3):
                            result.append(group_dict)
                    
                    json_data['task6']['output']['visual elements']['boxplots'] = result
                    json.dump(json_data, open(dst_path, 'w'), indent=4)

                cv2.imwrite('outs/' + img_path.split('/')[-1],image)
                
        if('PMC' in key): 
            if('vertical_box' in key):   ### vertical_box
                print(key)
                json_path = PMC_Jsons + key.split('/')[-2] + '/' + key.split('/')[-1].replace('jpg', 'json')
                img_path  = PMC_Imgs  + key.split('/')[-2] + '/' + key.split('/')[-1]
                
                dst_dir = Save_dir + key.split('/')[0] + '/' + key.split('/')[1]
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                dst_path = dst_dir + '/' + key.split('/')[-1].replace('jpg', 'json')
                
                Tpoints, image = detect_image(model, img_path, key)
                or_h, or_w    = image.shape[:2]
                
                First_Points  = Tpoints[0]
                Max_Points    = Tpoints[1]
                Median_Points = Tpoints[2]
                Min_Points    = Tpoints[3]
                Third_Points  = Tpoints[4]
                
                json_data    = json.load(open(json_path,'r'))
                
                if('vertical_box' in key):
                    result = []
                    task4_inputs = json_data['task6']['input']['task4_output']['axes']['x-axis']
                    for point_dict in task4_inputs:
                        x_base, y_base = point_dict['tick_pt']['x'], point_dict['tick_pt']['y']
                        
                        group_dict = {}
                        
                        min_dist  = 1000
                        min_point = None
                        for point in First_Points:
                            dist = abs(x_base - point[0])
                            if(dist < min_dist):
                                min_dist  = dist
                                min_point = point
                        if(min_dist< or_w*match_th):
                            group_dict['first_quartile'] = {'x': min_point[0],'y':min_point[1]}
                    
                        min_dist  = 1000
                        min_point = None
                        for point in Max_Points:
                            dist = abs(x_base - point[0])
                            if(dist < min_dist):
                                min_dist  = dist
                                min_point = point
                        if(min_dist< or_w*match_th):
                            group_dict['max'] = {'x': min_point[0],'y':min_point[1]}
                    
                        min_dist  = 1000
                        min_point = None
                        for point in Median_Points:
                            dist = abs(x_base - point[0])
                            if(dist < min_dist):
                                min_dist  = dist
                                min_point = point
                        if(min_dist< or_w*match_th):
                            group_dict['median'] = {'x': min_point[0],'y':min_point[1]}

                        min_dist  = 1000
                        min_point = None
                        for point in Min_Points:
                            dist = abs(x_base - point[0])
                            if(dist < min_dist):
                                min_dist  = dist
                                min_point = point
                        if(min_dist< or_w*match_th):
                            group_dict['min'] = {'x': min_point[0],'y':min_point[1]}
                    
                        min_dist  = 1000
                        min_point = None
                        for point in Third_Points:
                            dist = abs(x_base - point[0])
                            if(dist < min_dist):
                                min_dist  = dist
                                min_point = point
                        if(min_dist< or_w*match_th):
                            group_dict['third_quartile'] = {'x': min_point[0],'y':min_point[1]}
                        
                        if(len(group_dict.keys()) >=3):
                            result.append(group_dict)
                    
                    json_data['task6']['output']['visual elements']['boxplots'] = result
                    json.dump(json_data, open(dst_path, 'w'), indent=4)

                cv2.imwrite('outs/' + img_path.split('/')[-1],image)

