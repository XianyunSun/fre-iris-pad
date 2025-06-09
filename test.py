import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pdb

from train_config import cls_args as train_args
from dataset.data_config import *
from dataset.dataloader import BasicDataset
from model.mfad_mini import FAD_HAM_Net_mini_FFT_SpCh2
from utils.eval import eval_th, eval_roc

def test_epoch(model, data):
    model.eval()
    
    result_pad, label_pad = [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data)):
            img = batch['img'].cuda()
            if 'LBP' in batch.keys(): aug = batch['LBP'].cuda()
            else: aug = None
        
            output = model(img, aug) # not normalized
            pred_pad_norm = F.softmax(output['pad'], dim=1)
            #pred_pad_norm = pred_pad_norm.argmax(dim=1)

            result_pad.extend(pred_pad_norm.data.cpu().numpy())
            label_pad.extend(batch['pad'].data.cpu().numpy())
    
    raw_test_scores = result_pad

    return np.array(raw_test_scores), np.array(label_pad)


def main(test_list, args, save_path, save_name, save_csv=False, save_roc=False, save_hist=False):

    # build dataset
    test_data_name = [d.get_name() for d in test_list]
    test_dataset = BasicDataset(test_list, img_size=args.input_size, aug=args.augmentation, train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                num_workers=args.num_workers, pin_memory=False)
    test_image_list = test_dataset.image_list

    print('test set length', len(test_dataset))

    # load model
    model = FAD_HAM_Net_mini_FFT_SpCh2(pretrain=None, pad_classes=args.pad_class, variant='efficientnet_b0', fft='dct', image_shape=(3, args.input_size[0], args.input_size[1]), norm='sigmoid', input='aug', cat='para')
    model = model.cuda()

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt['model_g'])

    # test ==================================================================
    print('begin testing...')
    raw_test_scores, gt = test_epoch(model, test_dataloader)
    acc_pad = eval_roc(raw_test_scores, gt)
    acc_th = eval_th(raw_test_scores, gt, th=0.5)
    acc_log = 'under threshold {:.4f}, APCER={:.4f}, BPCER={:.4f}, EER={:.4f}, acc={:.4f}, auc={:.4f} \n' \
                .format(acc_pad['th'], acc_pad['APCER'], acc_pad['BPCER'], acc_pad['eer'], acc_pad['acc'], acc_pad['auc'])
    acc_log5 = 'under threshold 0.5, APCER={:.4f}, BPCER={:.4f}, acc={:.4f} \n' \
            .format(acc_th['APCER'], acc_th['BPCER'], acc_th['acc'])
    print(acc_log)
    print(acc_log5)

    # save result ==================================================================
    model_epoch = args.ckpt.split('/')[-1].split('-')[0].replace('epoch', '')
    with open(os.path.join(save_path, 'result_epoch'+model_epoch+'.txt'), 'w') as result_log:
        dataset_log = 'test dataset: {} \n' .format(test_data_name[0])
        result_log.write(dataset_log)
        result_log.write(acc_log)
        result_log.write(acc_log5)
    result_log.close()

    if save_csv:
        df = pd.DataFrame({'image':test_image_list, 'pred0':list(raw_test_scores[:,0]), 'pred1':list(raw_test_scores[:,1]), 'gt':gt})
        df.to_csv(os.path.join(save_path, 'pred_epoch'+model_epoch+'.csv'), index=False)
    
    if save_roc:
        fpr, tpr = acc_pad['fpr'], acc_pad['tpr']
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve of '+ save_name)
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_path, 'roc_epoch'+model_epoch+'.png'))
    
    if save_hist:
        pred = raw_test_scores[:,1]
        pred_att, pred_live = pred[gt==0], pred[gt==1]
        plt.figure()
        plt.hist(pred_att, bins=50, alpha=0.5, label='attack')
        plt.hist(pred_live, bins=50, alpha=0.5, label='live')
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(save_path, 'hist_epoch'+model_epoch+'.png'))



if __name__=='__main__':
    train_args = train_args()
    test_list = [LivDet2023_test_Config(train_args.data_type)]
    save_path = os.path.join(os.path.dirname(train_args.ckpt), 'pred')
    save_name = os.path.dirname(train_args.ckpt).split('/')[-2]
    os.makedirs(save_path, exist_ok=True)
    main(test_list, train_args, save_path, save_name, save_csv=True, save_roc=True, save_hist=True)
