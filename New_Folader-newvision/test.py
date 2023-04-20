import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
import timeit
import numpy as np
import oyaml as yaml
from torch.utils import data
import cv2
from model.pspnet import pspnet
from loader import get_loader
from metrics import runningScore
from utils import convert_state_dict
from augmentations import get_composed_augmentations
from collections import OrderedDict

import pdb
torch.backends.cudnn.benchmark = True

print(torch.__version__)


def get_class_colors(*args):
    def uint82bin(n, count=8):
        """returns the binary of integer n, count refers to amount of bits"""
        return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])
    label_colours = [
                 # 0!=background
                 (168, 198, 168), (0, 0, 198), (198, 154, 202),
                 (0, 0, 0), (198, 198, 100), (0, 100, 198),
                 (198, 42, 52), (192, 52, 154), (168, 0, 198),
                 (0, 198, 0), (90, 186, 198), (161, 107, 108),
                 (26, 200, 156), (202, 179, 158), (142, 151, 136),
                 (115, 201, 77), (100, 216, 255), (57, 156, 36),
                 (88, 108, 129), (105, 129, 112), (42, 137, 126),
                 (155, 108, 249), (166, 148, 143), (81, 91, 87),
                 (100, 124, 51), (73, 131, 121), (157, 210, 220),
                 (134, 181, 60), (221, 223, 147), (123, 108, 131),
                 (161, 66, 179), (163, 221, 160), (31, 146, 98),
                 (99, 121, 30), (49, 89, 240), (116, 108, 9),
                 (161, 176, 169), (80, 29, 135), (177, 105, 197),
                 (139, 110, 246)]
    return label_colours


def validate(cfg, args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if cfg["data"]["dataset"] == 'LFSyn_dataset':
        cfg["training"]["train_augmentations"]["scale"] = [480,640]
        cfg["training"]["train_augmentations"]["rcrop"] = [480,640]
        cfg["validating"]["val_augmentations"]["scale"] = [480,640]
    else:
        cfg["training"]["train_augmentations"]["scale"] = [432,623]
        cfg["training"]["train_augmentations"]["rcrop"] = [432,623]
        cfg["validating"]["val_augmentations"]["scale"] = [432,623]
        
    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    # Setup augmentations
    val_augmentations = cfg["validating"].get("val_augmentations", None)
    v_data_aug = get_composed_augmentations(val_augmentations)
    
    if cfg["data"]["dataset"] == 'LF_dataset':
        data_path = '/home/crx_pinkpanda1/LF_dataset'
    elif cfg["data"]["dataset"] == 'LFSyn_dataset':
        data_path = '/home/crx_pinkpanda1/UrbanLF_new/UrbanLF_Syn_small_dis'
    else:
        data_path = '/home/crx_pinkpanda1'
    
    v_loader = data_loader(data_path,split='test',augmentations=v_data_aug)
    
    n_classes = v_loader.n_classes
    valloader = data.DataLoader(v_loader, batch_size=cfg["validating"]["batch_size"], num_workers=cfg["validating"]["n_workers"])

    running_metrics = runningScore(n_classes)

    # Setup Model
    model = pspnet(nclass=14,backbone=cfg["model"]["backbone"]).to(device)
    state = torch.load(cfg["validating"]["resume"])["model_state"]  
    new_state = OrderedDict()
    
    
    for k,v in state.items():  
        name = k[7:]
        new_state[name] = v
    model.load_state_dict(new_state, strict=True)
    model.eval()
    model.to(device)
    print("Flip: ", cfg["validating"]["flip"])
    base_h = cfg["validating"]["base_size_h"]
    base_w = cfg["validating"]["base_size_w"]
    with torch.no_grad():
        for (val, labels,img_name_val) in valloader:
            gt = labels.numpy()
            if cfg["validating"]["mult_scale"] == True:
                batch, _, ori_height, ori_width = val.size()
                assert batch == 1, "only supporting batchsize 1."
                image = val.numpy()[0].transpose((1,2,0)).copy()
                stride_h = int(base_h * 1.0)
                stride_w = int(base_w * 1.0)     
                final_pred = torch.zeros([1, 14,ori_height,ori_width]).cuda()
                scales = cfg["validating"]["scales"]
                for scale in scales:     
                    long_size = int(base_w * scale + 0.5)   
                    h, w = image.shape[:2] 
                    if h > w:
                        new_h = long_size
                        new_w = int(w * long_size / h + 0.5)
                    else:
                        new_w = long_size
                        new_h = int(h * long_size / w + 0.5)

                    new_img = cv2.resize(image, (new_w, new_h),interpolation=cv2.INTER_LINEAR)
                    height, width = new_img.shape[:-1]

                    if scale <= 1.0:   
                        new_img = new_img.transpose((2, 0, 1))
                        new_img = np.expand_dims(new_img, axis=0)
                        new_img = torch.from_numpy(new_img)
                        _val = new_img.to(device) 
                        pred = model(_val,gt)
                        if cfg["validating"]["flip"] == True:
                            flip_img = _val.cpu().numpy()[:, :, :, ::-1]
                            flip_preds = model(torch.from_numpy(flip_img.copy()).cuda(),gt)
                            flip_preds = flip_preds.cpu().numpy().copy()
                            flip_preds = torch.from_numpy(flip_preds[:, :, :, ::-1].copy()).cuda()
                            pred += flip_preds
                            pred = pred * 0.5
                        pred = pred.exp()
                        preds = pred[:, :, 0:height, 0:width]
                    else:       
                        new_h, new_w = new_img.shape[:-1]
                        rows = int(np.ceil(1.0 * (new_h - base_h) / stride_h)) + 1
                        cols = int(np.ceil(1.0 * (new_w - base_w) / stride_w)) + 1
                        preds = torch.zeros([1, 14,new_h,new_w]).cuda()
                        count = torch.zeros([1,1, new_h, new_w]).cuda()
                        for r in range(rows):
                            for c in range(cols):
                                h0 = r * stride_h
                                w0 = c * stride_w
                                h1 = min(h0 + base_h, new_h)
                                w1 = min(w0 + base_w, new_w)
                                h0 = max(int(h1 - base_h), 0)
                                w0 = max(int(w1 - base_w), 0)
                                crop_img = new_img[h0:h1, w0:w1, :]
                                crop_img = crop_img.transpose((2, 0, 1))
                                crop_img = np.expand_dims(crop_img, axis=0)
                                crop_img = torch.from_numpy(crop_img)
                                crop_img = crop_img.cuda()
                                
                                pred = model(crop_img,gt)
                                if cfg["validating"]["flip"] == True:
                                    flip_img = crop_img.cpu().numpy()[:, :, :, ::-1]
                                    flip_preds = model(torch.from_numpy(flip_img.copy()).cuda(),gt)
                                    flip_preds = flip_preds.cpu().numpy().copy()
                                    flip_preds = torch.from_numpy(flip_preds[:, :, :, ::-1].copy()).cuda()
                                    pred += flip_preds
                                    pred = pred * 0.5
                                pred = pred.exp()
                                preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                                count[:,:,h0:h1,w0:w1] += 1
                        preds = preds / count 
                        preds = preds[:,:,:height,:width] 

                    preds = F.interpolate(
                        preds, (ori_height, ori_width),
                        mode='bilinear', align_corners=True
                    )
                    final_pred += preds
            
            '''
            _val = val.to(device) 
            outputs = model(_val,gt)
            '''
            pred = final_pred.data.max(1)[1].cpu().numpy()
            print(img_name_val[0].split('/')[-2])
             

            running_metrics.update(gt, pred)
                      
            gt[gt==255]=15
            class_colors = get_class_colors()           
            colored_label = np.vectorize(lambda x: class_colors[int(x)])
            
            for j in range(pred.shape[0]):      
                result_img = np.asarray(colored_label(pred[j])).astype(np.float32)   
                label_img = np.asarray(colored_label(gt[j])).astype(np.float32)   
                col_img = cv2.imread(img_name_val[j])          
                result_img = result_img.transpose(1,2,0)
                label_img = label_img.transpose(1,2,0)
                im_vis = np.concatenate((col_img,label_img, result_img),axis=1).astype(np.uint8)    
                img_name = img_name_val[j].split('/')[-2]
                cv2.imwrite(os.path.join(args.save_dir, img_name)+'.png', im_vis)
                numpy_name = 'numpy_res/sys/' + img_name
                os.mkdir(numpy_name)
                np.save(numpy_name+'/label.npy',(pred[j]+1).astype(np.uint8))
                
                

    score, class_iou,class_ac = running_metrics.get_scores()

    for k, v in score.items():
        print(k, v)
    print("each class iou")
    for i in range(n_classes):
        print(i, class_iou[i])
    print("each class acc")
    for i in range(n_classes):
        print(i, class_ac[i])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s_pascal.yml",
        help="Config file to be used",
    )       
    parser.add_argument(
        "--gpu",
        nargs="?",
        type=str,
        default="0",
        help="GPU ID",
    )      
    parser.add_argument(
        "--dataset",
        type=str,
        help="Configuration file to use",
    )      
    parser.add_argument("--save_dir",nargs="?",type=str,
            default="./output/",help="Path_to_Save",)
    parser.set_defaults(measure_time=True)
    args = parser.parse_args()
    with open(args.config) as fp:   
        cfg = yaml.safe_load(fp)
    cfg["data"]["dataset"] = args.dataset
    args.save_dir = 'output/'+args.dataset+'_'+cfg["model"]["backbone"]
    if not os.path.exists('output'):
        os.mkdir('output')
    if not os.path.exists('output/'+args.dataset):
        os.mkdir('output/'+args.dataset)
    if not os.path.exists(args.save_dir):       
        os.mkdir(args.save_dir)
    validate(cfg, args)
