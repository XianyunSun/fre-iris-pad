import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.cuda.amp import GradScaler
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import time
import numpy as np
import wandb
import logging
import pdb

from train_config import cls_args as train_args
from dataset.data_config import *
from dataset.dataloader import BasicDataset, UniformSampler
from model.mfad_mini import FAD_HAM_Net_mini_FFT_SpCh2
from hard_triplet_loss import HardTripletLoss
from utils.loss_meter import AvgrageMeter
from utils.eval import eval_argmax, eval_roc

def log_settings(args, logger):
    for arg in vars(args):  # Convert Namespace to dictionary and iterate
        setting_name = arg
        setting_value = getattr(args, arg)
        logger.info(f"{setting_name}: {setting_value}")

def log_and_print(words, logger):
    print(words)
    logger.info(words)

def train_epoch(model, data, optimizer, criterion, scaler, args):
    model.train()
    loss_total = AvgrageMeter()
    
    result_pad, label_pad = [], []
    for i, batch in enumerate(data):
        torch.cuda.empty_cache()

        img, pad, triplet = batch['img'].cuda(), batch['pad'].cuda(), batch['triplet'].cuda()
        if 'LBP' in batch.keys(): aug = batch['LBP'].cuda()
        elif 'CLAHE' in batch.keys(): aug = batch['CLAHE'].cuda()
        else: aug = None
    
        #output = model(img, aug) # not normalized
        output = model(img, aug, train=True)
        pred_pad_norm = F.softmax(output['pad'], dim=1)

        # loss
        pad_loss = criterion['pad'](output['pad'], pad)
        
        if args.triplet_weight>0:
            triplet_loss = criterion['triplet'](output['pad_feats'], triplet)
        else:
            triplet_loss = 0.

        if args.pixel_weight>0:
            pixel_label = pad.unsqueeze(1).expand(-1, output['pad_feats'].shape[-1]).to(pad.device)
            pixel_loss = criterion['pixel'](output['pad_feats'], pixel_label)
        else:
            pixel_loss = 0.

        loss = pad_loss*args.pad_weight + triplet_loss*args.triplet_weight + pixel_loss*args.pixel_weight
        #print('total loss:', loss.data, 'pad loss:',pad_loss.data)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if not args.wandb_log:
            print(pad_loss.data, triplet_loss.data, loss.data)

        loss_total.update(loss.data, img.shape[0])
        result_pad.extend(pred_pad_norm.data.cpu().numpy())
        label_pad.extend(pad.data.cpu().numpy())

    acc_pad = eval_roc(np.array(result_pad), label_pad)
    acc_argmax = eval_argmax(np.array(result_pad), label_pad)

    return loss_total.avg, acc_pad, acc_argmax

def test_epoch(model, data):
    model.eval()
    
    result_pad, label_pad = [], []
    with torch.no_grad():
        for i, batch in enumerate(data):

            output_g = model(batch['img'].cuda(), batch['LBP'].cuda() if 'LBP' in batch.keys() else None, train=False)
            pred_pad_norm = F.softmax(output_g['pad'], dim=1)

            result_pad.extend(pred_pad_norm.data.cpu().numpy())
            label_pad.extend(batch['pad'].data.numpy())

    acc_pad = eval_roc(np.array(result_pad), label_pad)
    acc_argmax = eval_argmax(np.array(result_pad), label_pad)

    return acc_pad, acc_argmax


def main(data_list, test_list, args):
    # log setting
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG) 
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    save_folder =  './'+args.ckpt_folder+'/'+args.save_name+'/'
    os.makedirs(save_folder, exist_ok=True)
    logname = save_folder+rq+'.log'
    logger.handlers.clear()
    fh = logging.FileHandler(logname, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s"))
    logger.addHandler(fh)
    log_settings(args, logger)
    if args.wandb_log:
        wandb.init(project='LivDet2023', name=args.save_name, reinit=True, mode='offline')
        config = wandb.config

    # build dataset
    train_data_name = [d.get_name() for d in data_list]

    if test_list is None:
        train_dataset = BasicDataset(data_list, img_size=args.input_size, aug=args.augmentation, SLA=args.SLA, train=True, split=0.2)
        test_data_name = ['intra']
        test_dataset = BasicDataset(data_list, img_size=args.input_size, aug=args.augmentation, SLA=args.SLA, train=False, 
                                    indexs=train_dataset.test_index)
    else:
        train_dataset = BasicDataset(data_list, img_size=args.input_size, aug=args.augmentation, SLA=args.SLA, train=True)
        test_data_name = [d.get_name() for d in test_list]
        test_dataset = BasicDataset(test_list, img_size=args.input_size, aug=args.augmentation, SLA=args.SLA)

    if args.pad_sampler:
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, 
                                        sampler=train_dataset.get_weight_sampler(), drop_last=True,
                                        num_workers=args.num_workers, pin_memory=False)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                        num_workers=args.num_workers, pin_memory=False)
    

    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, 
                                    num_workers=args.num_workers, pin_memory=False)

    train_domain_num = len(data_list)
    log_and_print(('train dataset: %s, length: %d' % (', '.join(train_data_name), len(train_dataset))), logger)
    log_and_print(('test dataset: %s, length: %d' % (', '.join(test_data_name), len(test_dataset))), logger)
    if args.wandb_log:
        config.train_set = train_data_name
        config.test_set = test_data_name
        config.args = args

    # load model
    model = FAD_HAM_Net_mini_FFT_SpCh2(pretrain=args.pretrained_ckpt, pad_classes=args.pad_class, variant='efficientnet_b0', fft='dct', image_shape=(3, args.input_size[0], args.input_size[1]), norm='sigmoid', input='aug', cat='para')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
        log_and_print('using multiple GPUs!', logger)
    else:
        model = model.cuda()
    
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt['model_g'])

    # optimizer
    if args.optimizer=='Adam': 
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_g)
    elif args.optimizer=='SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_g, momentum=0.9, weight_decay=5e-4)

    # scheduler
    if args.schedular=='step':
        start_lr = args.lr_g * args.gamma_g**int(args.start_epoch/args.step_g)
        for param_group in optimizer.param_groups:
            param_group['lr'] = start_lr
        lr_scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_g, gamma=args.gamma_g)
    elif args.schedular=='warmup_step':
        warmup_lambda = lambda epoch: (epoch+args.start_epoch+1)/args.warmup_epoch if epoch+args.start_epoch<args.warmup_epoch else 0.01+0.09*(epoch<args.total_epoch*0.5)+0.9*(epoch<args.total_epoch*0.2)
        lr_scheduler_g = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    
    # loss funciton
    class_criterion = nn.CrossEntropyLoss().cuda()
    triplet_criterion = HardTripletLoss(margin=args.triplet_margin, mask_label=None).cuda()
    pixel_criterion = nn.SmoothL1Loss().cuda()
    criterion = {'pad':class_criterion, 'triplet':triplet_criterion, 'pixel':pixel_criterion}

    scaler = GradScaler(enabled=False)

    # train ==================================================================
    log_and_print('------begin training------', logger)
    best_acc = 0.
    for i in range(args.start_epoch, args.total_epoch):
        start = time.time()
        loss, acc_pad, acc_argmax = train_epoch(model, train_dataloader, optimizer, criterion, scaler, args)
        acc_test, acc_argmax_test = test_epoch(model, test_dataloader)

        # logging
        log_and_print(('%02dth epoch train, auc:%.4f,  APCER:%.4f, BPCER:%.4f, EER:%.4f, ACC:%.4f, lr:%.6f, loss:%.4f, time:%.4f' 
                % (i, acc_pad['auc'], acc_pad['APCER'], acc_pad['BPCER'], acc_pad['eer'], acc_argmax, optimizer['g'].param_groups[0]['lr'], loss, time.time()-start)), 
                logger)
        log_and_print(('%02dth epoch test, auc:%.4f,  APCER:%.4f, BPCER:%.4f, EER:%.4f, ACC:%.4f' 
                % (i, acc_test['auc'], acc_test['APCER'], acc_test['BPCER'], acc_test['eer'], acc_argmax_test)), 
                logger)
        if args.wandb_log:
            wandb.log({'train/AUC':acc_pad['auc'], 'train/EER':acc_pad['eer'], 'train/APCER':acc_pad['APCER'], 'train/BPCER':acc_pad['BPCER'], 'train/loss':loss, 'train/ACC':acc_argmax})
            wandb.log({'test/AUC':acc_test['auc'], 'test/EER':acc_test['eer'], 'test/APCER':acc_test['APCER'], 'test/BPCER':acc_test['BPCER'], 'test/ACC':acc_argmax_test})

        # change learning rate
        lr_scheduler_g.step()

        # save model
        if acc_argmax_test>best_acc or (i+1)%5==0:
            save_path = '%sepoch%02d-acc_%.4f.pth' % (save_folder, i, acc_argmax_test)
            save_sota_path = '%ssota.pth' % (save_folder)
            save_dict = {'epoch':i, 'acc_test':acc_argmax_test, 
                         'train_set':train_data_name, 'test_set':test_data_name,
                         'model':model.state_dict()}
            torch.save(save_dict, save_path)
            if acc_argmax_test>best_acc:
                torch.save(save_dict, save_sota_path)
                best_acc = acc_argmax_test
                log_and_print('SOTA model updated', logger)



if __name__=='__main__':
    train_args = train_args()
    data_list = [LivDet2023_Config(train_args.data_type)]
    test_list = [LivDet2023_test_Config(train_args.data_type)]

    main(data_list, test_list, train_args)
