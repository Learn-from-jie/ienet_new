import os
import yaml as yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np
from torch.nn.parallel.scatter_gather import gather
from torch.utils import data
from tqdm import tqdm
#from encoding.parallel import DataParallelModel, DataParallelCriterion
from model.pspnet import pspnet
from loader import get_loader
from utils import get_logger
from augmentations import get_composed_augmentations
import pdb
from metrics import runningScore, averageMeter
torch.backends.cudnn.enabled = True
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
       
def init_seed(manual_seed, en_cudnn=False):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = en_cudnn
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    random.seed(manual_seed)  ####这是在干嘛？种子是用来干嘛的？

def train(cfg, logger, logdir):
    # Setup seeds
    init_seed(11733, en_cudnn=False)       

    if cfg["data"]["dataset"] == 'LFSyn_dataset':
        cfg["training"]["train_augmentations"]["scale"] = [480,640]
        cfg["training"]["train_augmentations"]["rcrop"] = [480,640]
        cfg["validating"]["val_augmentations"]["scale"] = [480,640]
    else:
        cfg["training"]["train_augmentations"]["scale"] = [432,623]
        cfg["training"]["train_augmentations"]["rcrop"] = [5,10]
        cfg["validating"]["val_augmentations"]["scale"] = [432,623]
    
    # Setup Augmentations
    train_augmentations = cfg["training"].get("train_augmentations", None) 
    t_data_aug = get_composed_augmentations(train_augmentations)    
    val_augmentations = cfg["validating"].get("val_augmentations", None)  
    v_data_aug = get_composed_augmentations(val_augmentations)

    data_loader = get_loader(cfg["data"]["dataset"])      
    if cfg["data"]["dataset"] == 'LF_dataset':
        data_path = 'F:/BaiduNetdiskDownload/semantic_segmentation/UrbanLF-Real'
    elif cfg["data"]["dataset"] == 'LFSyn_dataset':
        data_path = '/data/lj/UrbanLF/semantic_segmentation/UrbanLF-Syn'
    else:
        data_path = '/home/crx_pinkpanda1'
    t_loader = data_loader(data_path,split=cfg["data"]["train_split"],augmentations=t_data_aug)
    v_loader = data_loader(data_path,split=cfg["data"]["val_split"],augmentations=v_data_aug)

    trainloader = data.DataLoader(t_loader,
                                  batch_size=cfg["training"]["batch_size"],
                                  num_workers=cfg["training"]["n_workers"],
                                  shuffle=True,
                                  drop_last=True  )
    valloader = data.DataLoader(v_loader,
                                batch_size=cfg["validating"]["batch_size"],
                                num_workers=cfg["validating"]["n_workers"] )

    logger.info("Using training seting {}".format(cfg["training"]))
    
    running_metrics_val = runningScore(t_loader.n_classes)
    # Setup Model and Loss          
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)     
    model = pspnet(nclass=14, criterion=criterion,backbone=cfg["model"]["backbone"])
    modules_ori = [model.layer0,model.layer01,model.layer02,model.layer03,model.layer04,model.pretrained.layer1, model.pretrained.layer2, model.pretrained.layer3, model.pretrained.layer4,model.pretrained1.layer1,model.pretrained1.layer2,model.pretrained1.layer3,model.pretrained1.layer4]
    # ,model.layer0_2,model.layer0_3,model.layer0_4]
    modules_new = [model.head, model.auxlayer,model.centerimagequan,model.centerimagequan1,model.other,model.shuru]
    # if not args.pretrain is None: 
    #     checkpoint = torch.load(args.pretrain, map_location='cpu')
    #     pretrain_dict = checkpoint['model_state']
    #     state_dict = {'head.conv5.5.weight','head.conv5.5.bias','auxlayer.conv5.4.weight','auxlayer.conv5.4.bias'}  
    #     model_dict = {}
    #     for k, v in pretrain_dict.items():
    #         if k[7:] in state_dict:
    #             continue
    #         else:
    #             model_dict[k[7:]]=v
    #     model.load_state_dict(model_dict,strict=False)
    #     print("load model:",args.pretrain," best_iou:",checkpoint["best_iou"]," epoch:",checkpoint["epoch"])
    # Setup optimizer       
    params_list = []
    for module in modules_ori:
        params_list.append(dict(params=module.parameters(), lr=cfg["training"]["optimizer"]["lr0"]))
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=cfg["training"]["optimizer"]["lr0"] * 10))
    optimizer = torch.optim.SGD(params_list, lr=cfg["training"]["optimizer"]["lr0"], momentum=cfg["training"]["optimizer"]["momentum"], weight_decay=cfg["training"]["optimizer"]["wd"])


    # torch.cuda.set_device(1)
    device_ids = [0]
    model = model.cuda()
    # model = torch.nn.DataParallel(model,device_ids=device_ids)
    # device = torch.device("cuda:1")
    # device = 'cuda'
    # model = model.to(device)
    #Initialize training param
    cnt_iter = 0
    best_iou = 0.0
    best_iter = 0
    while cnt_iter <= cfg["training"]["train_iters"]:
        for (f_img, labels, img_name,disp) in trainloader:
            # import pdb
            # pdb.set_trace()
            cnt_iter += 1
            model.train()     
            f_img[0][0] = f_img[0][0].cuda(device='cuda:0', non_blocking=True)
            f_img[1] = [i.cuda(device='cuda:0', non_blocking=True) for i in f_img[1]] 
            f_img[2] = [i.cuda(device='cuda:0', non_blocking=True) for i in f_img[2]] 
            f_img[3] = [i.cuda(device='cuda:0', non_blocking=True) for i in f_img[3]] 
            f_img[4] = [i.cuda(device='cuda:0', non_blocking=True) for i in f_img[4]] 
            labels = labels.cuda(device='cuda:0', non_blocking=True)
            # f_img[0][0] = f_img[0][0].to(device=device_ids, non_blocking=True)
            # f_img[1] = [i.to(device=device_ids, non_blocking=True) for i in f_img[1]] 
            # f_img[2] = [i.to(device=device_ids, non_blocking=True) for i in f_img[2]] 
            # f_img[3] = [i.to(device=device_ids, non_blocking=True) for i in f_img[3]] 
            # f_img[4] = [i.to(device=device_ids, non_blocking=True) for i in f_img[4]] 
            # labels = labels.to(device=device_ids, non_blocking=True)
            # f_img = f_img.to(device)
            # labels = labels.to(device)
            #f_img = f_img.cuda()
            #labels = labels.cuda()
            output, main_loss, aux_loss = model(f_img, labels,disp)    
            main_loss, aux_loss = torch.mean(main_loss), torch.mean(aux_loss)  
            loss = main_loss + cfg["training"]["loss"]["aux_weight"] * aux_loss

            optimizer.zero_grad()  
            loss.backward()   
            optimizer.step()  
            current_lr = cfg["training"]["optimizer"]["lr0"] * (1-float(cnt_iter)/cfg["training"]["train_iters"]) ** cfg["training"]["optimizer"]["power"]            
            for index in range(0, 12):      
                optimizer.param_groups[index]['lr'] = current_lr
            for index in range(12, len(optimizer.param_groups)):    
                optimizer.param_groups[index]['lr'] = current_lr * 10
            
            if (cnt_iter + 1) % cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f} "
                print_str = fmt_str.format( cnt_iter + 1,
                                            cfg["training"]["train_iters"],
                                            loss.item(), )   

                print(print_str)
                logger.info(print_str)
                
            if (cnt_iter + 1) % cfg["training"]["val_interval"] == 0 or (cnt_iter + 1) == cfg["training"]["train_iters"]:
                model.eval()
                with torch.no_grad():
                    confusion_matrix = np.zeros((14, 14))
                    for (f_img_val, labels_val,img_name_val) in tqdm(valloader):
                        # f_img_val = f_img_val.to(device)
                        # labels_val = labels_val.to(device)
                        f_img_val[0][0] = f_img_val[0][0].cuda(device='cuda:0', non_blocking=True)
                        f_img_val[1] = [i.cuda(device='cuda:0', non_blocking=True) for i in f_img_val[1]] 
                        f_img_val[2] = [i.cuda(device='cuda:0', non_blocking=True) for i in f_img_val[2]] 
                        f_img_val[3] = [i.cuda(device='cuda:0', non_blocking=True) for i in f_img_val[3]] 
                        f_img_val[4] = [i.cuda(device='cuda:0', non_blocking=True) for i in f_img_val[4]] 
                        labels_val = labels_val.cuda(device='cuda:0', non_blocking=True)
                        outputs = model(f_img_val,labels_val)     
                        pred = outputs.data.max(1)[1].cpu().numpy()       
                        gt = labels_val.data.cpu().numpy()
                        running_metrics_val.update(gt, pred)      
                        
                score, class_iou,class_acc = running_metrics_val.get_scores()     
                for k, v in score.items():     
                    print(k, v)
                    logger.info("{}: {}".format(k, v))

                for k, v in class_iou.items(): 
                    logger.info("{}: {}".format(k, v))

                running_metrics_val.reset()  

                if score["Mean IoU : \t"] >= best_iou:  
                    best_iou = score["Mean IoU : \t"]
                    state = {
                        "epoch": cnt_iter + 1,
                        "model_state": model.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_path = os.path.join(logdir,
                        "{}_best_model_{}.pkl".format(cfg["model"]["arch"],cnt_iter+1),
                    )
                    torch.save(state, save_path)
                print("best iou:",best_iou)
    print("Final best iou:",best_iou)
#os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        help="Configuration file to use",
    )
    parser.add_argument(
        "--pretrain",
        nargs="?",
        type=str,
        help="Configuration file to use",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Configuration file to use",
    )
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)
    # pdb.set_trace()
    cfg["pretrain"] = args.pretrain
    cfg["data"]["dataset"] = args.dataset
    run_id = random.randint(1, 100000)  ##run_id是干嘛用的？
    if not os.path.exists("runs"):
        os.mkdir("runs")
    logdir = os.path.join("runs",args.dataset+"_"+cfg["model"]["backbone"])      
    if not os.path.exists(logdir):
        os.mkdir(logdir)    
    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Let the games begin")
    train(cfg, logger, logdir)
