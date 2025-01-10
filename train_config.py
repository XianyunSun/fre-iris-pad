import argparse

def cls_args():

    parser = argparse.ArgumentParser(description='MADiris')

    # dataset
    parser.add_argument('--domain_sampler', action='store_true') # equally sample images from each training domain
    parser.add_argument('--pad_sampler', action='store_true') # equally sampler live and attack images in each domain
    parser.add_argument('--SLA', action='store_true') # if True, consider transparent samples as attack

    # augmentation
    parser.add_argument('--data_type', type=str, default='raw', choices=['norm', 'roi', 'raw'])
    parser.add_argument('--augmentation', type=list, default=['LBP']) # a list contain augmentations e.g. CLAHE, LBP

    # model
    parser.add_argument('--input_size', type=tuple, default=(384, 384))
    parser.add_argument('--pad_class', type=int, default=2)
    parser.add_argument('--pretrained_ckpt', type=str, default="/sdata/xianyun.sun/timm/efficientnet_b0.ra_in1k/pytorch_model.bin")
    #parser.add_argument('--pretrained_ckpt', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default='/sdata/xianyun.sun/iris/MADiris_pub/ckpt/effb0_dctmini384_spch2_sigmoid_sgd_lr1e-4_tri/epoch14-acc_0.6586.pth')
    #parser.add_argument('--ckpt', type=str, default=None)

    # loss
    parser.add_argument('--keep_generator', action='store_true')
    parser.add_argument('--keep_discriminator', action='store_true')
    parser.add_argument('--pad_weight', type=float, default=1.)
    parser.add_argument('--pixel_weight', type=float, default=0.5)
    parser.add_argument('--triplet_weight', type=float, default=1.0)
    parser.add_argument('--triplet_margin', type=float, default=0.5)

    # optimizer
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['Adam', 'SGD'])
    parser.add_argument('--lr_g', type=float, default=1e-4)

    # schedular
    parser.add_argument('--schedular', type=str, default='step', choices=['step', 'warmup_step'])
    parser.add_argument('--gamma_g', type=float, default=0.7)
    parser.add_argument('--step_g', type=float, default=5)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--warmup_epoch', type=int, default=0)
    
    # train
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--total_epoch', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)

    # logging
    parser.add_argument('--ckpt_folder', type=str, default='ckpt')
    parser.add_argument('--save_name', type=str, default='testrun')
    parser.add_argument('--wandb_log', action='store_true')


    args = parser.parse_args()

    return args
