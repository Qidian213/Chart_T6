import os
import cv2
import json
import torch
import random
import numpy as np
import torch.nn.functional as F
from skimage.feature import hog
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage
from models import Get_model

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

def calCHist(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([img_hsv], [0], None, [256], [0, 255])
    hist2 = cv2.calcHist([img_hsv], [1], None, [256], [0, 255])
    hist3 = cv2.calcHist([img_hsv], [2], None, [256], [0, 255])
    hist  = np.hstack((hist1.reshape(-1), hist2.reshape(-1), hist3.reshape(-1)))
    return hist

def calHog(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fd1 = hog(img_gray, 9, [32, 32], [1, 1])
    fd2 = hog(img_gray, 9, [16, 16], [2, 2])
    # fd3 = hog(img_gray, 9, [8, 8], [2, 2])
    fd  = np.hstack((fd1, fd2))
    return fd
    
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

def detect_image(model, file, Task1_ctype):
    image   = cv2.imread(file)
    Tscores, Tpoints = detect(model, image)

    # for points in Tpoints:
        # color = (random.randint(1,255), random.randint(1,255), random.randint(1,255))
        # for t_point in points:
            # cv2.circle(image, (int(t_point[0]), int(t_point[1])), 3, [255,0,0], -1)

    return Tpoints, image

class Cfg_Opts(object):
    def __init__(self,):
        self.Model_Set                = {}
        self.Model_Set['Model_name']  = 'DLA_34'
        self.Model_Set['Head_dict']   = {'heatmap':1, 'reg': 2, 'wh': 2}
        
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
    model.load_param("work_space/DLA_34_2020-10-27-22-49-25/Epoch_best.pth")

### SYN
    SYN_Test_Jsons_Dir = '/data/Dataset/Chart/ICPR2020_CHARTINFO_SYNTHETIC_TEST/task_6/Inputs/'
    SYN_Test_Imgs_Dir  = '/data/Dataset/Chart/ICPR2020_CHARTINFO_SYNTHETIC_TEST/task_6/Charts/'
    SYN_Test_Outs_Dir  = '/data/zzg/ICPR_Chart/AE_Chart_T6_Point/results/SYN_Test_Outs/'
    
    One_SYN = ["Line", "Scatter"]
    
    SYN_Json_Files  = os.listdir(SYN_Test_Jsons_Dir)
    for json_file in SYN_Json_Files:
        Img_path  = SYN_Test_Imgs_Dir + json_file.replace('json', 'png')
        Json_path = SYN_Test_Jsons_Dir + json_file

        Json_data = json.load(open(Json_path,'r'))

        Task1_ctype = Json_data['task1_output']['chart_type']  ### ctype
        Task2_boxes = Json_data['task2_output']['text_blocks']
        Task3_roles = Json_data['task3_output']['text_roles']
        Task4_axes  = Json_data['task4_output']
        Task5_pair  = Json_data['task5_output']['legend_pairs']

        if(Task1_ctype in One_SYN):
            print(Img_path)
            Tpoints, image = detect_image(model, Img_path, Task1_ctype)
            Tpoints = Tpoints[0]
            
            if("Scatter" == Task1_ctype):
                result = []
                plot_bb = Task4_axes['_plot_bb']
                x0, y0, x1, y1 = plot_bb['x0'], plot_bb['y0'], plot_bb['x0'] + plot_bb['width'], plot_bb['y0'] + plot_bb['height']
                
                for point in Tpoints:
                    if(point[0]<x0+10 or point[0]>x1-5 or point[1]<y0+5 or point[1]>y1-10):
                        continue
                    point_dict = {}
                    point_dict['x'] = point[0]
                    point_dict['y'] = point[1]
                    result.append(point_dict)

                    cv2.circle(image, (int(point[0]), int(point[1])), 3, [255,0,0], -1)

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
                Result_Dict['task6']['output']['visual elements']['scatter points'] = result
                Result_Dict['task6']['output']['visual elements']['boxplots']       = []

                json.dump(Result_Dict, open(SYN_Test_Outs_Dir + json_file, 'w'), indent=4)

                cv2.imwrite('outs/' + Img_path.split('/')[-1],image)

            if("Line" == Task1_ctype):
                result = []
                plot_bb = Task4_axes['_plot_bb']
                x0, y0, x1, y1 = plot_bb['x0'], plot_bb['y0'], plot_bb['x0'] + plot_bb['width'], plot_bb['y0'] + plot_bb['height']
                
                Npoints = []
                for point in Tpoints:
                    if(point[0]<x0+10 or point[0]>x1-5 or point[1]<y0+5 or point[1]>y1-10):
                        continue
                    Npoints.append(point)

                Line_Group = []
                if(len(Task5_pair) >0):
                    for id in range(len(Task5_pair)):
                        Line_Group.append([])

                    bws, bhs = [], []
                    for i, bb in enumerate(Task5_pair):
                        bh, bw, x1, y1 = int(bb['bb']['height']), int(bb['bb']['width']), int(bb['bb']['x0']), int(bb['bb']['y0'])
                        bws.append(bw)
                        bhs.append(bh)
                    ebw = max(bws)
                    ebh = max(bhs)

                    hog_features = np.zeros((0, (9+36)))
                    chist_features = np.zeros((0, 256 * 3))
                    for i, bb in enumerate(Task5_pair):
                        bh, bw, x1, y1 = int(bb['bb']['height']), int(bb['bb']['width']), int(bb['bb']['x0']), int(bb['bb']['y0'])
                        cx= x1 + bw//2
                        cy= y1 + bh//2
                        
                        ele_img = image[cy - ebh // 2: cy + ebh // 2, cx - ebh // 2:cx + ebh // 2, :]
                        ele_img = cv2.resize(ele_img, (32, 32))
                        
                        hist_f  = calCHist(ele_img)
                        hist_f  = preprocessing.normalize(hist_f.reshape(1,-1), norm='l2')
                        chist_features = np.vstack((chist_features, hist_f))

                        fd = calHog(ele_img)
                        fd = preprocessing.normalize(fd.reshape(1, -1), norm='l2')
                        hog_features = np.vstack((hog_features, np.hstack((fd))))
                        
                    for point in Npoints:
                        x, y  = point[0], point[1]
                        x1    = int(x - ebh // 2)
                        y1    = int(y - ebh // 2)
                        img_s = image[y1:y1 + ebh, x1:x1 + ebh, :]

                        img_s  = cv2.resize(img_s, (32, 32))
                        hist_f = calCHist(img_s)
                        hist_f = preprocessing.normalize(hist_f.reshape(1,-1), norm='l2')
                        
                        hog_f = calHog(img_s)
                        hog_f = preprocessing.normalize(hog_f.reshape(1, -1), norm='l2')

                        dist = np.sqrt(np.sum(np.asarray(hist_f - chist_features) ** 2, axis=1))
                        mind = np.argmin(dist)
                        Line_Group[mind].append({'x':x,'y':y})
                else:
                    for point in Npoints:
                        Line_Group.append({'x':point[0],'y':point[1]})
                    Line_Group = [Line_Group]

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
                Result_Dict['task6']['output']['visual elements']['lines']          = Line_Group
                Result_Dict['task6']['output']['visual elements']['scatter points'] = []
                Result_Dict['task6']['output']['visual elements']['boxplots']       = []

                json.dump(Result_Dict, open(SYN_Test_Outs_Dir + json_file, 'w'), indent=4)

                for points in Line_Group:
                    color = (random.randint(1,255), random.randint(1,255), random.randint(1,255))
                    for t_point in points:
                        cv2.circle(image, (int(t_point['x']), int(t_point['y'])), 3, color, -1)
                cv2.imwrite('outs/' + Img_path.split('/')[-1],image)

    # scatter 95
    # vertical bar 225
    # line 264
    # horizontal bar 71
    # vertical box 71

### PMC
    SpLine_Lists = {'PMC548283___1471-2458-5-1-3.jpg':2, 'PMC549189___1471-2156-6-4-3_panel_1.jpg':1, 
    'PMC1403762___1471-2458-6-16-1.jpg':2, 'PMC2263030___1471-2458-8-60-2.jpg':4, 'PMC2464733___pgen.1000137.g012.jpg':2, 
    'PMC2800840___1471-2350-10-127-5.jpg':2,'PMC2848222___1471-2350-11-34-2.jpg':2, 'PMC3038975___1471-2156-12-15-2.jpg':2, 
    'PMC3605113___pgen.1003348.g001.jpg':3, 'PMC3605113___pgen.1003348.g002.jpg':3,'PMC3794634___CMMM2013-231762.002.jpg':2,
    'PMC3933275___nihms537020f2.jpg':2, 'PMC4211006___ijerph-11-10790-g002.jpg':3, 'PMC4627007___ijerph-12-12905-g001.jpg':3,
    'PMC4738718___CMMM2016-9343017.004.jpg':4, 'PMC5224609___nanomaterials-06-00130-g004.jpg':4, 'PMC5457097___materials-09-00714-g007.jpg':4,
    'PMC5457097___materials-09-00714-g004.jpg':2, 'PMC5459090___materials-10-00463-g010b.jpg':2, 'PMC5706272___materials-10-01325-g0A4.jpg':3, 
    'PMC5701919___fpubh-05-00289-g003.jpg':2}

    Color_List = ['PMC5701919___fpubh-05-00289-g003.jpg', 'PMC5706272___materials-10-01325-g0A4.jpg','PMC5459090___materials-10-00463-g010b.jpg',
                  'PMC5224609___nanomaterials-06-00130-g004.jpg','PMC4627007___ijerph-12-12905-g001.jpg','PMC2848222___1471-2350-11-34-2.jpg',
                  'PMC1403762___1471-2458-6-16-1.jpg']
                  
    Hof_list  = ['PMC5457097___materials-09-00714-g004.jpg','PMC3794634___CMMM2013-231762.002.jpg', 'PMC2800840___1471-2350-10-127-5.jpg',
                  'PMC548283___1471-2458-5-1-3.jpg']
    
    VLine_Lists = {'PMC2646133___pgen.1000406.g003.jpg': 4, 'PMC3804362___CMMM2013-264809.010.jpg': 3}
    
    PMC_Test_Jsons_Dir = '/data/Dataset/Chart/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/split_4/annotations/'
    PMC_Test_Imgs_Dir  = '/data/Dataset/Chart/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/split_4/images/'
    PMC_Test_Outs_Dir  = '/data/zzg/ICPR_Chart/AE_Chart_T6_Point/results/PMC_Test_Outs/'
    
    One_PMC = ["line", "scatter"]
    match_th = 0.02
    PMC_Json_Files  = os.listdir(PMC_Test_Jsons_Dir)
    for json_file in PMC_Json_Files:
        Img_path  = PMC_Test_Imgs_Dir + json_file.replace('json', 'jpg')
        Json_path = PMC_Test_Jsons_Dir + json_file

        Json_data = json.load(open(Json_path,'r'))

        Task1_ctype = Json_data['task6']['input']['task1_output']['chart_type']  ### ctype
        Task2_boxes = Json_data['task6']['input']['task2_output']['text_blocks']
        Task3_roles = Json_data['task6']['input']['task3_output']['text_roles']
        Task4_axes  = Json_data['task6']['input']['task4_output']
        Task5_pair  = Json_data['task6']['input']['task5_output']['legend_pairs']

        if(Task1_ctype in One_PMC):
            print(Img_path)
            Tpoints, image = detect_image(model, Img_path, Task1_ctype)
            Tpoints = Tpoints[0]
            or_h, or_w = image.shape[:2]
            
            if("scatter" == Task1_ctype):
                result = []
                plot_bb = Task4_axes['_plot_bb']
                x0, y0, x1, y1 = plot_bb['x0'], plot_bb['y0'], plot_bb['x0'] + plot_bb['width'], plot_bb['y0'] + plot_bb['height']
                
                Npoints = []
                for point in Tpoints:
                    if(point[0]<x0+10 or point[0]>x1-5 or point[1]<y0+5 or point[1]>y1-10):
                        continue
                    Npoints.append(point)
                if('PMC3804362___CMMM2013-264809' in Img_path):
                    Npoints = Tpoints

                Scatter_Group = []
                if(len(Task5_pair) >0):
                    for id in range(len(Task5_pair)):
                        Scatter_Group.append([])

                    bws, bhs = [], []
                    for i, bb in enumerate(Task5_pair):
                        bh, bw, x1, y1 = int(bb['bb']['height']), int(bb['bb']['width']), int(bb['bb']['x0']), int(bb['bb']['y0'])
                        bws.append(bw)
                        bhs.append(bh)
                    ebw = max(bws)
                    ebh = max(bhs)

                    hog_features = np.zeros((0, (9+36)))
                    chist_features = np.zeros((0, 256 * 3))
                    for i, bb in enumerate(Task5_pair):
                        bh, bw, x1, y1 = int(bb['bb']['height']), int(bb['bb']['width']), int(bb['bb']['x0']), int(bb['bb']['y0'])
                        cx= x1 + bw//2
                        cy= y1 + bh//2
                        
                        ele_img = image[cy - ebh // 2: cy + ebh // 2, cx - ebh // 2:cx + ebh // 2, :]
                        ele_img = cv2.resize(ele_img, (32, 32))
                        
                        hist_f  = calCHist(ele_img)
                        hist_f  = preprocessing.normalize(hist_f.reshape(1,-1), norm='l2')
                        chist_features = np.vstack((chist_features, hist_f))

                        fd = calHog(ele_img)
                        fd = preprocessing.normalize(fd.reshape(1, -1), norm='l2')
                        hog_features = np.vstack((hog_features, np.hstack((fd))))
                        
                    for point in Npoints:
                        x, y  = point[0], point[1]
                        x1    = int(x - ebh // 2)
                        y1    = int(y - ebh // 2)
                        img_s = image[y1:y1 + ebh, x1:x1 + ebh, :]

                        img_s  = cv2.resize(img_s, (32, 32))
                        hist_f = calCHist(img_s)
                        hist_f = preprocessing.normalize(hist_f.reshape(1,-1), norm='l2')
                        
                        hog_f = calHog(img_s)
                        hog_f = preprocessing.normalize(hog_f.reshape(1, -1), norm='l2')

                        dist = np.sqrt(np.sum(np.asarray(hist_f - chist_features) ** 2, axis=1)) + np.sqrt(
                               np.sum(np.asarray(hog_f-hog_features)**2, axis=1))
                        mind = np.argmin(dist)
                        Scatter_Group[mind].append({'x':x,'y':y})
                else:
                    if(Img_path.split('/')[-1] in VLine_Lists.keys()):
                        for idc in range(VLine_Lists[Img_path.split('/')[-1]]):
                            Scatter_Group.append([])
                            
                        for id, point in enumerate(Npoints):
                            x = point[0]
                            min_ind = 10000
                            min_dist = 10000
                            for id_g, g_points in enumerate(Scatter_Group):
                                dist = 0
                                for g_point in g_points:
                                    dist += abs(g_point['x'] - x)
                                dist = dist/(len(g_points) + 1e-6)
                                if(dist < min_dist and dist>0):
                                    min_dist = dist
                                    min_ind  = id_g
                            if(min_dist < or_w*match_th and min_dist>0):
                                Scatter_Group[min_ind].append({'x':point[0],'y':point[1]})
                            else:
                                for id in range(len(Scatter_Group)):
                                    if(len(Scatter_Group[id]) ==0):
                                        Scatter_Group[id].append({'x':point[0],'y':point[1]})
                                        break
                    else:
                        if(Img_path.split('/')[-1] in SpLine_Lists.keys()):
                            if(len(Npoints) >=2):
                                hist_feats = []
                                hog_feats  = []
                                for point in Npoints:
                                    x, y  = point[0], point[1]
                                    img_s = image[int(y-3):int(y+3), int(x-3):int(x+3), :]

                                    img_s  = cv2.resize(img_s, (12, 12))
                                    hist_f = calCHist(img_s)
                                    hist_f = preprocessing.normalize(hist_f.reshape(1,-1), norm='l2')
                                    
                                    # hog_f = calHog(img_s)
                                    # hog_f = preprocessing.normalize(hog_f.reshape(1, -1), norm='l2')
                                    
                                    hist_feats.append(hist_f[0])
                                    # hog_feats.append(hog_f[0])
                                hist_feats = np.array(hist_feats)
                                #hog_feats = np.array(hog_feats)
                                
                                hist_distmat = cdist(hist_feats,  hist_feats, metric='euclidean')
                               # hog_distmat  = cdist(hog_feats,  hog_feats, metric='euclidean')
                                dist_mat     = hist_distmat# + hog_distmat
                                
                                clu_model  = AgglomerativeClustering(n_clusters=SpLine_Lists[Img_path.split('/')[-1]], affinity='precomputed', linkage='average')
                                clustering = clu_model.fit(dist_mat)
                                Npoints = np.array(Npoints)
                                for la_id in range(SpLine_Lists[Img_path.split('/')[-1]]):
                                    inds = np.where(clustering.labels_ == la_id)
                                    la_points = Npoints[inds]
                                    la_group = []
                                    for point in la_points:
                                        la_group.append({'x':point[0],'y':point[1]})
                                    Scatter_Group.append(la_group)
                            else:
                                for point in Npoints:
                                    Scatter_Group.append({'x':point[0],'y':point[1]})
                                Scatter_Group = [Scatter_Group]
                        else:
                            for point in Npoints:
                                Scatter_Group.append({'x':point[0],'y':point[1]})
                            Scatter_Group = [Scatter_Group]

                Result_Dict = {}
                Result_Dict['task6'] = {}
                Result_Dict['task6']['input'] = {}
                Result_Dict['task6']['input']['task1_output'] = Json_data['task6']['input']['task1_output']
                Result_Dict['task6']['input']['task2_output'] = Json_data['task6']['input']['task2_output']
                Result_Dict['task6']['input']['task3_output'] = Json_data['task6']['input']['task3_output']
                Result_Dict['task6']['input']['task4_output'] = Json_data['task6']['input']['task4_output']
                Result_Dict['task6']['input']['task5_output'] = Json_data['task6']['input']['task5_output']
                
                Result_Dict['task6']['name'] = "Data Extraction"
                Result_Dict['task6']['output'] = {}
                Result_Dict['task6']['output']['visual elements'] = {}
                Result_Dict['task6']['output']['visual elements']['bars']           = []
                Result_Dict['task6']['output']['visual elements']['legend box']     = []
                Result_Dict['task6']['output']['visual elements']['lines']          = []
                Result_Dict['task6']['output']['visual elements']['scatter points'] = Scatter_Group
                Result_Dict['task6']['output']['visual elements']['boxplots']       = []

                json.dump(Result_Dict, open(PMC_Test_Outs_Dir + json_file, 'w'), indent=4)

                for points in Scatter_Group:
                    color = (random.randint(1,255), random.randint(1,255), random.randint(1,255))
                    for t_point in points:
                        cv2.circle(image, (int(t_point['x']), int(t_point['y'])), 3, color, -1)
                cv2.imwrite('outs/' + Img_path.split('/')[-1],image)

            if("line" == Task1_ctype):
                result = []
                plot_bb = Task4_axes['_plot_bb']
                x0, y0, x1, y1 = plot_bb['x0'], plot_bb['y0'], plot_bb['x0'] + plot_bb['width'], plot_bb['y0'] + plot_bb['height']
                
                Npoints = []
                for point in Tpoints:
                    if(point[0]<x0+10 or point[0]>x1-5 or point[1]<y0+5 or point[1]>y1-10):
                        continue
                    Npoints.append(point)

                Line_Group = []
                if(len(Task5_pair) >0):
                    for id in range(len(Task5_pair)):
                        Line_Group.append([])

                    bws, bhs = [], []
                    for i, bb in enumerate(Task5_pair):
                        bh, bw, x1, y1 = int(bb['bb']['height']), int(bb['bb']['width']), int(bb['bb']['x0']), int(bb['bb']['y0'])
                        bws.append(bw)
                        bhs.append(bh)
                    ebw = max(bws)
                    ebh = max(bhs)

                    hog_features = np.zeros((0, (9+36)))
                    chist_features = np.zeros((0, 256 * 3))
                    for i, bb in enumerate(Task5_pair):
                        bh, bw, x1, y1 = int(bb['bb']['height']), int(bb['bb']['width']), int(bb['bb']['x0']), int(bb['bb']['y0'])
                        cx= x1 + bw//2
                        cy= y1 + bh//2
                        
                        ele_img = image[cy - ebh // 2: cy + ebh // 2, cx - ebh // 2:cx + ebh // 2, :]
                        ele_img = cv2.resize(ele_img, (32, 32))
                        
                        hist_f  = calCHist(ele_img)
                        hist_f  = preprocessing.normalize(hist_f.reshape(1,-1), norm='l2')
                        chist_features = np.vstack((chist_features, hist_f))

                        fd = calHog(ele_img)
                        fd = preprocessing.normalize(fd.reshape(1, -1), norm='l2')
                        hog_features = np.vstack((hog_features, np.hstack((fd))))
                        
                    for point in Npoints:
                        x, y  = point[0], point[1]
                        x1    = int(x - ebh // 2)
                        y1    = int(y - ebh // 2)
                        img_s = image[y1:y1 + ebh, x1:x1 + ebh, :]

                        img_s  = cv2.resize(img_s, (32, 32))
                        hist_f = calCHist(img_s)
                        hist_f = preprocessing.normalize(hist_f.reshape(1,-1), norm='l2')
                        
                        hog_f = calHog(img_s)
                        hog_f = preprocessing.normalize(hog_f.reshape(1, -1), norm='l2')

                        dist = np.sqrt(np.sum(np.asarray(hist_f - chist_features) ** 2, axis=1)) + np.sqrt(
                               np.sum(np.asarray(hog_f-hog_features)**2, axis=1))
                        mind = np.argmin(dist)
                        Line_Group[mind].append({'x':x,'y':y})
                else:
                    if(Img_path.split('/')[-1] in SpLine_Lists.keys()):
                        if(len(Npoints) >=2):
                            hist_feats = []
                            hog_feats  = []
                            for point in Npoints:
                                x, y  = point[0], point[1]
                                
                                if(Img_path.split('/')[-1] in Color_List):
                                    img_s  = image[int(y-3):int(y+3), int(x-3):int(x+3), :]
                                    img_s  = cv2.resize(img_s, (12, 12))
                                    hist_f = calCHist(img_s)
                                    hist_f = preprocessing.normalize(hist_f.reshape(1,-1), norm='l2')
                                    hist_feats.append(hist_f[0])
                                elif(Img_path.split('/')[-1] in Hof_list):
                                    img_s = image[int(y-6):int(y+6), int(x-6):int(x+6), :]
                                    img_s  = cv2.resize(img_s, (32, 32))
                                    hog_f = calHog(img_s)
                                    hog_f = preprocessing.normalize(hog_f.reshape(1, -1), norm='l2')
                                    hog_feats.append(hog_f[0])
                                else:
                                    img_s  = image[int(y-3):int(y+3), int(x-3):int(x+3), :]
                                    img_s  = cv2.resize(img_s, (12, 12))
                                    hist_f = calCHist(img_s)
                                    hist_f = preprocessing.normalize(hist_f.reshape(1,-1), norm='l2')
                                    hist_feats.append(hist_f[0])
                                    
                                    img_s = image[int(y-6):int(y+6), int(x-6):int(x+6), :]
                                    img_s  = cv2.resize(img_s, (32, 32))
                                    hog_f = calHog(img_s)
                                    hog_f = preprocessing.normalize(hog_f.reshape(1, -1), norm='l2')
                                    hog_feats.append(hog_f[0])
                                
                            hist_feats = np.array(hist_feats)
                            hog_feats  = np.array(hog_feats)
                            
                            if(Img_path.split('/')[-1] in Color_List):
                                hist_distmat = cdist(hist_feats,  hist_feats, metric='euclidean')
                                dist_mat     = hist_distmat
                            elif(Img_path.split('/')[-1] in Hof_list):
                                hog_distmat  = cdist(hog_feats,  hog_feats, metric='euclidean')
                                dist_mat     = hog_distmat
                            else:
                                hist_distmat = cdist(hist_feats,  hist_feats, metric='euclidean')
                                hog_distmat  = cdist(hog_feats,  hog_feats, metric='euclidean')
                                dist_mat     = hist_distmat + hog_distmat
                            
                            clu_model  = AgglomerativeClustering(n_clusters=SpLine_Lists[Img_path.split('/')[-1]], affinity='precomputed', linkage='average')
                            clustering = clu_model.fit(dist_mat)
                            Npoints = np.array(Npoints)
                            for la_id in range(SpLine_Lists[Img_path.split('/')[-1]]):
                                inds = np.where(clustering.labels_ == la_id)
                                la_points = Npoints[inds]
                                la_group = []
                                for point in la_points:
                                    la_group.append({'x':point[0],'y':point[1]})
                                Line_Group.append(la_group)
                        else:
                            for point in Npoints:
                                Line_Group.append({'x':point[0],'y':point[1]})
                            Line_Group = [Line_Group]
                    else:
                        for point in Npoints:
                            Line_Group.append({'x':point[0],'y':point[1]})
                        Line_Group = [Line_Group]

                Result_Dict = {}
                Result_Dict['task6'] = {}
                Result_Dict['task6']['input'] = {}
                Result_Dict['task6']['input']['task1_output'] = Json_data['task6']['input']['task1_output']
                Result_Dict['task6']['input']['task2_output'] = Json_data['task6']['input']['task2_output']
                Result_Dict['task6']['input']['task3_output'] = Json_data['task6']['input']['task3_output']
                Result_Dict['task6']['input']['task4_output'] = Json_data['task6']['input']['task4_output']
                Result_Dict['task6']['input']['task5_output'] = Json_data['task6']['input']['task5_output']
                
                Result_Dict['task6']['name'] = "Data Extraction"
                Result_Dict['task6']['output'] = {}
                Result_Dict['task6']['output']['visual elements'] = {}
                Result_Dict['task6']['output']['visual elements']['bars']           = []
                Result_Dict['task6']['output']['visual elements']['legend box']     = []
                Result_Dict['task6']['output']['visual elements']['lines']          = Line_Group
                Result_Dict['task6']['output']['visual elements']['scatter points'] = []
                Result_Dict['task6']['output']['visual elements']['boxplots']       = []

                json.dump(Result_Dict, open(PMC_Test_Outs_Dir + json_file, 'w'), indent=4)

                for points in Line_Group:
                    color = (random.randint(1,255), random.randint(1,255), random.randint(1,255))
                    for t_point in points:
                        cv2.circle(image, (int(t_point['x']), int(t_point['y'])), 3, color, -1)
                cv2.imwrite('outs/' + Img_path.split('/')[-1],image)