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

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

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

def detect(model, image, threshold=0.3):
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

def detect_image(model, file, ctype):
    image   = cv2.imread(file)
    Tscores, Tpoints = detect(model, image)

    # for points in Tpoints:
        # color = (random.randint(1,255), random.randint(1,255), random.randint(1,255))
        
        # for t_point in points:
            # cv2.circle(image, (int(t_point[0]), int(t_point[1])), 3, color, -1)
            # if("Horizontal box" == ctype):
                # st_y, et_y = int(t_point[1] - t_point[3]/2), int(t_point[1] + t_point[3]/2)
                # cv2.line(image, (int(t_point[0]), st_y), (int(t_point[0]), et_y), color,1)
            # else:
                # st_x, et_x = int(t_point[0] - t_point[2]/2), int(t_point[0] + t_point[2]/2)
                # cv2.line(image, (st_x, int(t_point[1])), (et_x, int(t_point[1])), color,1)

    return Tpoints,image

class Cfg_Opts(object):
    def __init__(self,):
        self.Model_Set                = {}
        self.Model_Set['Model_name']  = 'DLA_34'
        self.Model_Set['Head_dict']   = {'heatmap':5, 'reg': 2, 'wh': 2}

if __name__ == "__main__":
    # Grouped horizontal bar 66
    # Scatter 56
    # Stacked vertical bar 70
    # Line 66
    # Vertical box 67
    # Stacked horizontal bar 67
    # Grouped vertical bar 73
    # Horizontal box 62

    # scatter 95
    # vertical bar 225
    # line 264
    # horizontal bar 71
    # vertical box 71
    
    cfg   = Cfg_Opts()
    model = Get_model(cfg)
    model.eval()
    model.cuda()
    model.load_param("work_space/DLA_34_2020-10-27-22-46-31/Epoch_best.pth")
    
    match_th = 0.02
    
### SYN
    SYN_Test_Jsons_Dir = '/data/Dataset/Chart/ICPR2020_CHARTINFO_SYNTHETIC_TEST/task_6/Inputs/'
    SYN_Test_Imgs_Dir  = '/data/Dataset/Chart/ICPR2020_CHARTINFO_SYNTHETIC_TEST/task_6/Charts/'
    SYN_Test_Outs_Dir  = '/data/zzg/ICPR_Chart/AE_Chart_T6_Point/results/SYN_Test_Outs/'
    
    Five_SYN = ["Horizontal box", "Vertical box"]
    
    SYN_Json_Files  = os.listdir(SYN_Test_Jsons_Dir)
    for json_file in SYN_Json_Files:
        Img_path  = SYN_Test_Imgs_Dir + json_file.replace('json', 'png')
        Json_path = SYN_Test_Jsons_Dir + json_file

        Json_data = json.load(open(Json_path,'r'))

        Task1_ctype = Json_data['task1_output']['chart_type']  ### ctype
        Task2_boxes = Json_data['task2_output']['text_blocks']
        Task3_roles = Json_data['task3_output']['text_roles']
        Task4_axes  = Json_data['task4_output']['axes']
        Task5_pair  = Json_data['task5_output']['legend_pairs']

        if(Task1_ctype in Five_SYN):
            Tpoints, image = detect_image(model, Img_path, Task1_ctype)
            or_h, or_w    = image.shape[:2]
            
            First_Points  = Tpoints[0]
            Max_Points    = Tpoints[1]
            Median_Points = Tpoints[2]
            Min_Points    = Tpoints[3]
            Third_Points  = Tpoints[4]
            
            if("Horizontal box" == Task1_ctype):
                result = []
                Task4_XS = Task4_axes['x-axis']
                for point_dict in Task4_XS:
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
                
                    if(len(group_dict.keys()) >=2):
                        result.append(group_dict)
                
                Result_Dict = {}
                Result_Dict['task6'] = {}
                Result_Dict['task6']['input'] = {}
                Result_Dict['task6']['input']['task1_output'] = Json_data['task1_output']
                Result_Dict['task6']['input']['task2_output'] = Json_data['task2_output']
                Result_Dict['task6']['input']['task3_output'] = Json_data['task3_output']
                Result_Dict['task6']['input']['task4_output'] = Json_data['task4_output']
                Result_Dict['task6']['input']['task5_output'] = Json_data['task5_output']
                
                Result_Dict['task6']['name'] = "Data Extraction"
                Result_Dict['task6']['output'] = {}
                Result_Dict['task6']['output']['visual elements'] = {}
                Result_Dict['task6']['output']['visual elements']['bars']           = []
                Result_Dict['task6']['output']['visual elements']['legend box']     = []
                Result_Dict['task6']['output']['visual elements']['lines']          = []
                Result_Dict['task6']['output']['visual elements']['scatter points'] = []
                Result_Dict['task6']['output']['visual elements']['boxplots']       = result

                json.dump(Result_Dict, open(SYN_Test_Outs_Dir + json_file, 'w'), indent=4)

                for group_dict in result:
                    color = (random.randint(1,255), random.randint(1,255), random.randint(1,255))
                    if('first_quartile' in group_dict.keys()):
                        cv2.circle(image, (int(group_dict['first_quartile']['x']), int(group_dict['first_quartile']['y'])), 3, color, -1)
                    if('max' in group_dict.keys()):
                        cv2.circle(image, (int(group_dict['max']['x']), int(group_dict['max']['y'])), 3, color, -1)
                    if('median' in group_dict.keys()):
                        cv2.circle(image, (int(group_dict['median']['x']), int(group_dict['median']['y'])), 3, color, -1)
                    if('min' in group_dict.keys()):
                        cv2.circle(image, (int(group_dict['min']['x']), int(group_dict['min']['y'])), 3, color, -1)
                    if('third_quartile' in group_dict.keys()):
                        cv2.circle(image, (int(group_dict['third_quartile']['x']), int(group_dict['third_quartile']['y'])), 3, color, -1)
                cv2.imwrite('outs/' + Img_path.split('/')[-1],image)

            if("Vertical box" == Task1_ctype):
                result = []
                Task4_XS = Task4_axes['x-axis']
                for point_dict in Task4_XS:
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
                    
                    if(len(group_dict.keys()) >=2):
                        result.append(group_dict)
                
                Result_Dict = {}
                Result_Dict['task6'] = {}
                Result_Dict['task6']['input'] = {}
                Result_Dict['task6']['input']['task1_output'] = Json_data['task1_output']
                Result_Dict['task6']['input']['task2_output'] = Json_data['task2_output']
                Result_Dict['task6']['input']['task3_output'] = Json_data['task3_output']
                Result_Dict['task6']['input']['task4_output'] = Json_data['task4_output']
                Result_Dict['task6']['input']['task5_output'] = Json_data['task5_output']
                
                Result_Dict['task6']['name'] = "Data Extraction"
                Result_Dict['task6']['output'] = {}
                Result_Dict['task6']['output']['visual elements'] = {}
                Result_Dict['task6']['output']['visual elements']['bars']           = []
                Result_Dict['task6']['output']['visual elements']['legend box']     = []
                Result_Dict['task6']['output']['visual elements']['lines']          = []
                Result_Dict['task6']['output']['visual elements']['scatter points'] = []
                Result_Dict['task6']['output']['visual elements']['boxplots']       = result

                json.dump(Result_Dict, open(SYN_Test_Outs_Dir + json_file, 'w'), indent=4)
                
                for group_dict in result:
                    color = (random.randint(1,255), random.randint(1,255), random.randint(1,255))
                    if('first_quartile' in group_dict.keys()):
                        cv2.circle(image, (int(group_dict['first_quartile']['x']), int(group_dict['first_quartile']['y'])), 3, color, -1)
                    if('max' in group_dict.keys()):
                        cv2.circle(image, (int(group_dict['max']['x']), int(group_dict['max']['y'])), 3, color, -1)
                    if('median' in group_dict.keys()):
                        cv2.circle(image, (int(group_dict['median']['x']), int(group_dict['median']['y'])), 3, color, -1)
                    if('min' in group_dict.keys()):
                        cv2.circle(image, (int(group_dict['min']['x']), int(group_dict['min']['y'])), 3, color, -1)
                    if('third_quartile' in group_dict.keys()):
                        cv2.circle(image, (int(group_dict['third_quartile']['x']), int(group_dict['third_quartile']['y'])), 3, color, -1)
                cv2.imwrite('outs/' + Img_path.split('/')[-1],image)

### PMC
    # scatter 95
    # vertical bar 225
    # line 264
    # horizontal bar 71
    # vertical box 71

    Plots_V  = {'PMC3407908___ijerph-09-02345-g004.jpg':8, 'PMC3744414___pgen.1003697.g003_part_B.jpg':6, 'PMC3899415___je-21-240-g001.jpg': 9, 
                'PMC4388571___pgen.1004969.g002_panel_1.jpg':9,'PMC4429935___12863_2015_194_Fig1_HTML_panel_7.jpg':1, 
                'PMC4769534___13148_2016_188_Fig3_HTML_panel_5.jpg':6, 'PMC4808930___ijerph-13-00267-g005.jpg': 12,
                'PMC5096291___13148_2016_281_Fig2_HTML_panel_3.jpg':10, 'PMC5334757___ijerph-14-00203-g004_panel_1.jpg':8}
                
    PMC_Test_Jsons_Dir = '/data/Dataset/Chart/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/split_4/annotations/'
    PMC_Test_Imgs_Dir  = '/data/Dataset/Chart/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/split_4/images/'
    PMC_Test_Outs_Dir  = '/data/zzg/ICPR_Chart/AE_Chart_T6_Point/results/PMC_Test_Outs/'
    
    Five_PMC = ["vertical box"]
    
    PMC_Json_Files  = os.listdir(PMC_Test_Jsons_Dir)
    for json_file in PMC_Json_Files:
        Img_path  = PMC_Test_Imgs_Dir + json_file.replace('json', 'jpg')
        Json_path = PMC_Test_Jsons_Dir + json_file

        Json_data = json.load(open(Json_path,'r'))

        Task1_ctype = Json_data['task6']['input']['task1_output']['chart_type']  ### ctype
        Task2_boxes = Json_data['task6']['input']['task2_output']['text_blocks']
        Task3_roles = Json_data['task6']['input']['task3_output']['text_roles']
        Task4_axes  = Json_data['task6']['input']['task4_output']['axes']
        Task5_pair  = Json_data['task6']['input']['task5_output']['legend_pairs']

        if(Task1_ctype in Five_PMC):
            Tpoints, image = detect_image(model, Img_path, Task1_ctype)
            or_h, or_w    = image.shape[:2]
            
            First_Points  = Tpoints[0]
            Max_Points    = Tpoints[1]
            Median_Points = Tpoints[2]
            Min_Points    = Tpoints[3]
            Third_Points  = Tpoints[4]
                
            if("vertical box" == Task1_ctype):
                result = []
                Task4_XS = Task4_axes['x-axis']
                for point_dict in Task4_XS:
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
                    
                    if(len(group_dict.keys()) >=2):
                        result.append(group_dict)

                if(Img_path.split('/')[-1] in Plots_V.keys()):
                    First_Points  = Tpoints[0]
                    Max_Points    = Tpoints[1]
                    Median_Points = Tpoints[2]
                    Min_Points    = Tpoints[3]
                    Third_Points  = Tpoints[4]
                    
                    result  = []
                    # Clu_Num = Plots_V[Img_path.split('/')[-1]]
                    # for id in range(Clu_Num):
                        # result.append({})
                    for point in First_Points:
                        result.append({'first_quartile':{'x': point[0], 'y':point[1]}})
                    
                    for id, point in enumerate(Max_Points):
                        x = point[0]
                        min_ind = 10000
                        min_dist = 10000
                        for id_g, g_dict in enumerate(result):
                            dist = 0
                            for key in g_dict.keys():
                                dist += abs(g_dict[key]['x'] - x)
                            dist = dist/len(list(g_dict.keys()))
                            if(dist < min_dist):
                                min_dist = dist
                                min_ind  = id_g
                        if(min_dist < or_w*match_th):
                            result[min_ind]['max'] = {'x': point[0], 'y':point[1]}
                        else:
                            result.append({'max':{'x': point[0], 'y':point[1]}})

                    for id, point in enumerate(Median_Points):
                        x = point[0]
                        min_ind = 10000
                        min_dist = 10000
                        for id_g, g_dict in enumerate(result):
                            dist = 0
                            for key in g_dict.keys():
                                dist += abs(g_dict[key]['x'] - x)
                            dist = dist/len(list(g_dict.keys()))
                            if(dist < min_dist):
                                min_dist = dist
                                min_ind  = id_g
                        if(min_dist < or_w*match_th):
                            result[min_ind]['median'] = {'x': point[0], 'y':point[1]}
                        else:
                            result.append({'median':{'x': point[0], 'y':point[1]}})

                    for id, point in enumerate(Min_Points):
                        x = point[0]
                        min_ind = 10000
                        min_dist = 10000
                        for id_g, g_dict in enumerate(result):
                            dist = 0
                            for key in g_dict.keys():
                                dist += abs(g_dict[key]['x'] - x)
                            dist = dist/len(list(g_dict.keys()))
                            if(dist < min_dist):
                                min_dist = dist
                                min_ind  = id_g
                        if(min_dist < or_w*match_th):
                            result[min_ind]['min'] = {'x': point[0], 'y':point[1]}
                        else:
                            result.append({'min':{'x': point[0], 'y':point[1]}})

                    for id, point in enumerate(Third_Points):
                        x = point[0]
                        min_ind = 10000
                        min_dist = 10000
                        for id_g, g_dict in enumerate(result):
                            dist = 0
                            for key in g_dict.keys():
                                dist += abs(g_dict[key]['x'] - x)
                            dist = dist/len(list(g_dict.keys()))
                            if(dist < min_dist):
                                min_dist = dist
                                min_ind  = id_g
                        if(min_dist < or_w*match_th):
                            result[min_ind]['min'] = {'x': point[0], 'y':point[1]}
                        else:
                            result.append({'third_quartile':{'x': point[0], 'y':point[1]}})
                            
                Result_Dict = {}
                Result_Dict['visual elements'] = {}
                Result_Dict['visual elements']['bars']           = []
                Result_Dict['visual elements']['legend box']     = []
                Result_Dict['visual elements']['lines']          = []
                Result_Dict['visual elements']['scatter points'] = []
                Result_Dict['visual elements']['boxplots']       = result
                
                Json_data['task6']['output'] = Result_Dict
                json.dump(Json_data, open(PMC_Test_Outs_Dir + json_file, 'w'), indent=4)

                for group_dict in result:
                    color = (random.randint(1,255), random.randint(1,255), random.randint(1,255))
                    if('first_quartile' in group_dict.keys()):
                        cv2.circle(image, (int(group_dict['first_quartile']['x']), int(group_dict['first_quartile']['y'])), 3, color, -1)
                    if('max' in group_dict.keys()):
                        cv2.circle(image, (int(group_dict['max']['x']), int(group_dict['max']['y'])), 3, color, -1)
                    if('median' in group_dict.keys()):
                        cv2.circle(image, (int(group_dict['median']['x']), int(group_dict['median']['y'])), 3, color, -1)
                    if('min' in group_dict.keys()):
                        cv2.circle(image, (int(group_dict['min']['x']), int(group_dict['min']['y'])), 3, color, -1)
                    if('third_quartile' in group_dict.keys()):
                        cv2.circle(image, (int(group_dict['third_quartile']['x']), int(group_dict['third_quartile']['y'])), 3, color, -1)
                cv2.imwrite('outs/' + Img_path.split('/')[-1],image)
