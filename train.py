#!/usr/bin/python
# -*- encoding: utf-8 -*-
from logger import setup_logger
from models.model_stages import BiSeNet
# from cityscapes import CityScapes
from mydata import Mydata
from loss.loss import OhemCELoss
from loss.detail_loss import DetailAggregateLoss
from evaluation import MscEvalV0,MscEvalV_show
from optimizer_loss import Optimizer
from loss.my_loss import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import os.path as osp
import logging
import time
import datetime
import argparse
import os
import wandb
CUDA_LAUNCH_BLOCKING=1
logger = logging.getLogger()  # 创建logger对象，（name=None）


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parse = argparse.ArgumentParser()  # 建立解析对象
    # 给parse对象添加属性
    parse.add_argument(
        '--local_rank',
        dest='local_rank',
        type=int,
        default=0,
    )
    parse.add_argument(
        '--n_workers_train',
        dest='n_workers_train',
        type=int,
        default=4,
    )
    parse.add_argument(
        '--n_workers_val',
        dest='n_workers_val',
        type=int,
        default=1,
    )
    parse.add_argument(
        '--n_img_per_gpu',
        dest='n_img_per_gpu',
        type=int,
        default=8
    )
    parse.add_argument(
        '--max_iter',
        dest='max_iter',
        type=int,
        default=60000,
    )
    parse.add_argument(
        '--save_iter_sep',
        dest='save_iter_sep',
        type=int,
        default=1000,
    )
    parse.add_argument(
        '--warmup_steps',
        dest='warmup_steps',
        type=int,
        default=1000,
    )
    parse.add_argument(
        '--mode',
        dest='mode',
        type=str,
        default='train',
    )
    parse.add_argument(
        '--ckpt',
        dest='ckpt',
        type=str,
        default=None,
    )
    parse.add_argument(
        '--respath',
        dest='respath',
        type=str,
        default='checkpoints/train_STDC1-Seg/',
    )
    parse.add_argument(
        '--backbone',
        dest='backbone',
        type=str,
        default='STDCNet813',
    )
    parse.add_argument(
        '--pretrain_path',
        dest='pretrain_path',
        type=str,
        default='checkpoints/STDCNet813M_73.91.tar',
    )
    parse.add_argument(
        '--use_conv_last',
        dest='use_conv_last',
        type=str2bool,
        default=False,
    )
    parse.add_argument(
        '--use_boundary_2',
        dest='use_boundary_2',
        type=str2bool,
        default=False,
    )
    parse.add_argument(
        '--use_boundary_4',
        dest='use_boundary_4',
        type=str2bool,
        default=False,
    )
    parse.add_argument(
        '--use_boundary_8',
        dest='use_boundary_8',
        type=str2bool,
        default=True,
    )
    parse.add_argument(
        '--use_boundary_16',
        dest='use_boundary_16',
        type=str2bool,
        default=True,
    )

    parse.add_argument(
        '--n_classes',
        dest='n_classes',
        type=int,
        default=2,
    )
    parse.add_argument(
        '--dspth',
        dest='dspth',
        type=str,
        default='fourth_1106/',
    )
    # 把属性返回，或者可以 args = parse.parse_args() args就可以直接使用
    return parse.parse_args()


def train():
    args = parse_args()  # 接收函数的返回值
    print(args)

    # 初始化一个wandb run, 并设置超参数
    # Initialize a new run
    # wandb.init(project="pytorch-intro")
    wandb.init(project='1129',name="fusionv4_183")
    wandb.watch_called = False  # Re-run the model without restarting the runtime, unnecessary after our next release

    # config is a variable that holds and saves hyper parameters and inputs
    config = wandb.config  # Initialize config
    config.batch_size = args.n_img_per_gpu  # input batch size for training (default:64)
    # input batch size for testing(default:1000)
    config.epochs = args.max_iter  # number of epochs to train(default:10)

    save_pth_path = os.path.join(args.respath, 'pths')
    dspth = args.dspth  # 数据集rootpath

    print(save_pth_path)
    # print(osp.exists(save_pth_path))
    # if not osp.exists(save_pth_path) and dist.get_rank()==0: 
    if not osp.exists(save_pth_path):
        os.makedirs(save_pth_path)

    # os.environ['CUDA_VISIBLE_DEVICES'] = 0
    torch.cuda.set_device(args.local_rank)
    # dist.init_process_group(
    #             backend = 'nccl',
    #             init_method = 'tcp://127.0.0.1:33274',
    #             world_size = torch.cuda.device_count(),
    #             rank=args.local_rank
    #             )

    setup_logger(args.respath)  # 保存日志
    ## dataset
    n_classes = args.n_classes
    n_img_per_gpu = args.n_img_per_gpu
    n_workers_train = args.n_workers_train
    n_workers_val = args.n_workers_val
    use_boundary_16 = args.use_boundary_16
    use_boundary_8 = args.use_boundary_8
    use_boundary_4 = args.use_boundary_4
    use_boundary_2 = args.use_boundary_2

    use_ssimloss=False
    use_distanceloss=False
    mode = args.mode
    cropsize = [640,640]
    randomscale = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)

    # if dist.get_rank()==0:
    # logger.setLevel(logging.INFO) 设置logger级别
    logger.info('n_workers_train: {}'.format(n_workers_train))  # log 输出
    logger.info('n_workers_val: {}'.format(n_workers_val))
    logger.info('use_boundary_2: {}'.format(use_boundary_2))
    logger.info('use_boundary_4: {}'.format(use_boundary_4))
    logger.info('use_boundary_8: {}'.format(use_boundary_8))
    logger.info('use_boundary_16: {}'.format(use_boundary_16))
    logger.info('use_ssimloss: {}'.format(use_ssimloss))
    logger.info('use_distanceloss: {}'.format(use_distanceloss))
    logger.info('mode: {}'.format(args.mode))
    logger.info('batchsize:{}'.format(n_img_per_gpu))

    ds = Mydata(dspth, cropsize=cropsize, mode=mode, randomscale=randomscale)
    # sampler = torch.utils.data.distributed.DistributedSampler(ds)
    dl = DataLoader(ds,
                    batch_size=n_img_per_gpu,
                    shuffle=False,
                    # sampler = sampler,
                    num_workers=n_workers_train,
                    pin_memory=False,
                    drop_last=True)
    # exit(0)
    dsval = Mydata(dspth, mode='val', randomscale=randomscale)
    # sampler_val = torch.utils.data.distributed.DistributedSampler(dsval)
    dlval = DataLoader(dsval,
                       batch_size=16,
                       shuffle=False,
                       # sampler = sampler_val,
                       num_workers=n_workers_val,
                       drop_last=False)

    ## model
    ignore_idx = 255
    net = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path,
                  use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4, use_boundary_8=use_boundary_8,
                  use_boundary_16=use_boundary_16, use_conv_last=args.use_conv_last)

    if not args.ckpt is None:
        net.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    net.cuda()
    net.train()
    # net = nn.parallel.DistributedDataParallel(net,
    #         device_ids = [args.local_rank, ],
    #         output_device = args.local_rank,
    #         find_unused_parameters=True
    #         )

    score_thres = 0.7
    n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // 16
    # CE loss
    criteria_p = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_16 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_32 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    # Detail loss
    boundary_loss_func = DetailAggregateLoss()  # detail head+detail loss
    # the STDC Architecture is over.

    ## optimizer
    maxmIOU = 0.
    framecount=271
    # maxmIOU75 = 0.
    momentum = 0.9
    # weight_decay = 5e-4
    weight_decay = 5e-4
    # lr_start = 1e-2
    lr_start = 1e-3
    max_iter = args.max_iter
    save_iter_sep = args.save_iter_sep
    power = 0.9
    warmup_steps = args.warmup_steps
    warmup_start_lr = 1e-5
    logger.info('lr_start:{}'.format(lr_start))
    # if dist.get_rank()==0:
    print('max_iter: ', max_iter)
    print('save_iter_sep: ', save_iter_sep)
    print('warmup_steps: ', warmup_steps)

    optim = Optimizer(
        model=net,
        loss=boundary_loss_func,
        lr0=lr_start,
        momentum=momentum,
        wd=weight_decay,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        max_iter=max_iter,
        power=power)

    ## train loop
    msg_iter = 50
    loss_avg = []
    loss_boundery_bce = []
    loss_boundery_dice = []
    loss_ssim_all =[]
    loss_distance_all =[]
    st = glob_st = time.time()
    diter = iter(dl)
    epoch = 0
    for it in range(max_iter):
        try:
            im, lb = next(diter)
        except StopIteration:
            epoch += 1
            # sampler.set_epoch(epoch)
            diter = iter(dl)
            im, lb = next(diter)
        im = im.cuda()
        lb = lb.cuda()
        H, W = im.size()[2:]
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()

        if use_boundary_2 and use_boundary_4 and use_boundary_8:
            out, out16, out32, detail2, detail4, detail8 = net(im)

        if (not use_boundary_2) and use_boundary_4 and use_boundary_8:
            out, out16, out32, detail4, detail8 = net(im)

        if (not use_boundary_2) and (not use_boundary_4) and use_boundary_8:
            out, out16, out32, detail8 = net(im)

        if (not use_boundary_2) and (not use_boundary_4) and (not use_boundary_8):
            out, out16, out32 = net(im)
        # max_val = torch.max(out)
        # min_val = torch.min(out)
        # max_val16 = torch.max(out16)
        # min_val16 = torch.min(out16)
        # max_val8 = torch.max(detail8)
        # min_val8 = torch.min(detail8)
        lossp = criteria_p(out, lb)
        loss2 = criteria_16(out16, lb)
        loss3 = criteria_32(out32, lb)
        # lossp = distancepunish_loss(out, lb)
        # loss2 = distancepunish_loss(out16, lb)
        # loss3 = distancepunish_loss(out32, lb)

        boundery_bce_loss = 0.
        boundery_dice_loss = 0.

        if use_boundary_2:
            # if dist.get_rank()==0:
            #     print('use_boundary_2')
            boundery_bce_loss2, boundery_dice_loss2 = boundary_loss_func(detail2, lb)
            boundery_bce_loss += boundery_bce_loss2
            boundery_dice_loss += boundery_dice_loss2

        if use_boundary_4:
            # if dist.get_rank()==0:
            #     print('use_boundary_4')
            boundery_bce_loss4, boundery_dice_loss4 = boundary_loss_func(detail4, lb)
            boundery_bce_loss += boundery_bce_loss4
            boundery_dice_loss += boundery_dice_loss4

        if use_boundary_8:
            # if dist.get_rank()==0:
            #     print('use_boundary_8')
            boundery_bce_loss8, boundery_dice_loss8 = boundary_loss_func(detail8, lb)
            boundery_bce_loss += boundery_bce_loss8
            boundery_dice_loss += boundery_dice_loss8
        # if use_boundary_8:
        #     # if dist.get_rank()==0:
        #     #     print('use_boundary_8')
        #     boundery_dice_loss8 = dice_loss(detail8, lb)
        #     boundery_dice_loss += boundery_dice_loss8
        loss_ssim=0.
        if use_ssimloss:
            # Calculate the structural similarity loss
            loss_ssim_out = structural_similarity_loss(out, lb, window_size=11)
            loss_ssim_out16 = structural_similarity_loss(out16,lb,window_size=15)
            loss_ssim_out32 = structural_similarity_loss(out32,lb,window_size=17)
            loss_ssim  = loss_ssim_out+ loss_ssim_out16+ loss_ssim_out32
        loss_distance = 0.
        if use_distanceloss:
            loss_distance_out4= distancepunish_loss(detail4, lb)
            loss_distance_out16 = distancepunish_loss(out16, lb)
            loss_distance_out8 = distancepunish_loss(detail8, lb)

            loss_distance = loss_distance_out4 + loss_distance_out16+loss_distance_out8
            # loss_distance = loss_distance_out8_bce + loss_distance_out8_dice

        # compute the all loss and backward.

        loss = lossp + loss2 + loss3 + boundery_bce_loss + boundery_dice_loss+ loss_ssim+loss_distance
        loss.backward()
        optim.step()

        # the train one step is over.
        # boundery_bce_loss = torch.tensor([0.0]).cuda()
        # boundery_dice_loss = torch.tensor([0.0]).cuda()
        loss_ssim = torch.tensor([0.0]).cuda()
        loss_distance =  torch.tensor([0.0]).cuda()
        loss_avg.append(loss.item())
        loss_boundery_bce.append(boundery_bce_loss.item())
        loss_boundery_dice.append(boundery_dice_loss.item())
        loss_ssim_all.append(loss_ssim.item())
        loss_distance_all.append(loss_distance.item())

        ## print training log message
        if (it + 1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))

            loss_boundery_bce_avg = sum(loss_boundery_bce) / len(loss_boundery_bce)
            loss_boundery_dice_avg = sum(loss_boundery_dice) / len(loss_boundery_dice)
            loss_ssim_avg =sum(loss_ssim_all)/len(loss_ssim_all)
            loss_distance_avg =sum(loss_distance_all)/len(loss_distance_all)
            msg = ', '.join([
                'it: {it}/{max_it}',
                'lr: {lr:4f}',
                'loss: {loss:.4f}',
                'boundery_bce_loss: {boundery_bce_loss:.4f}',
                'boundery_dice_loss: {boundery_dice_loss:.4f}',
                'loss_ssim:{loss_ssim:.4f}',
                'loss_distance:{loss_distance:.4f}',
                'eta: {eta}',
                'time: {time:.4f}',
            ]).format(
                it=it + 1,
                max_it=max_iter,
                lr=lr,
                loss=loss_avg,
                boundery_bce_loss=loss_boundery_bce_avg,
                boundery_dice_loss=loss_boundery_dice_avg,
                loss_ssim=loss_ssim_avg,
                loss_distance=loss_distance_avg,
                time=t_intv,
                eta=eta
            )

            # 用wandb记录
            wandb.log({

                'lr': lr,
                'loss': loss,
                'boundery_bce_loss': boundery_bce_loss,
                'boundery_dice_loss': boundery_dice_loss,
                'loss_ssim':loss_ssim,
                'loss_distance':loss_distance,
            })
            #
            logger.info(msg)
            loss_avg = []
            loss_boundery_bce = []
            loss_boundery_dice = []
            loss_ssim_all = []
            loss_distance_all = []
            st = ed

            # print(boundary_loss_func.get_params())
        if (it + 1) % save_iter_sep == 0:  # and it != 0:

            ## model
            logger.info('evaluating the model ...')
            logger.info('setup and restore model')

            net.eval()

            # ## evaluator
            logger.info('compute the mIOU')
            with torch.no_grad():
                # single_scale1 = MscEvalV_show()
                # mIOU50 = single_scale1(net, dlval, n_classes)

                single_scale2 = MscEvalV_show(scale=1)
                mIOU = single_scale2(net, dlval, n_classes)

            #             save_pth = osp.join(save_pth_path, 'model_iter{}_mIOU50_{}_mIOU75_{}.pth'
            #             .format(it+1, str(round(mIOU50,4)), str(round(mIOU75,4))))

            #             state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
            #             #if dist.get_rank()==0:
            #             torch.save(state, save_pth)

            #             logger.info('training iteration {}, model saved to: {}'.format(it+1, save_pth))

            # if mIOU50 > maxmIOU50:
            #     maxmIOU50 = mIOU50
            #     save_pth = "ckpt/model_maxmIOU50.pth"
            #     state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
            #     # if dist.get_rank()==0:
            #     torch.save(state, save_pth)
            #
            #     logger.info('max mIOU model saved to: {}'.format(save_pth))

            if mIOU > maxmIOU:
                maxmIOU = mIOU
                save_pth = "ckpt/model_maxmIOU.pth"
                state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                # if dist.get_rank()==0:
                torch.save(state, save_pth)
                logger.info('max mIOU model saved to: {}'.format(save_pth))

            #
            wandb.log({
                # 'mIOU50': mIOU50,
                'mIOU': mIOU
            })
            #
            # logger.info('mIOU50 is: {}, mIOU75 is: {}'.format(mIOU50, mIOU75))
            # logger.info('maxmIOU50 is: {}, maxmIOU75 is: {}.'.format(maxmIOU50, maxmIOU75))
            logger.info('mIOU is: {}'.format(mIOU))
            logger.info('maxmIOU is: {}'.format(maxmIOU))

            net.train()

    ## dump the final model
    # save_pth = osp.join(save_pth_path, 'model_final.pth')
    save_pth = "ckpt/model_final.pth"
    net.cpu()
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    # if dist.get_rank()==0:
    torch.save(state, save_pth)
    logger.info('training done, model saved to: {}'.format(save_pth))
    print('epoch: ', epoch)


if __name__ == "__main__":
    train()
