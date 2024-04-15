
import os
import logging
import numpy as np
import torch
import test as test
from bisect import bisect_right

from utils import AverageMeter
from engines.evaluator_ge import Evaluator
from tqdm import tqdm

def train_cot(model, train_dataset, val_dataset, test_dataset, cfg, logger):
    def freeze(m):
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
            p.grad = None

    params_word_embedding = []
    params_encoder = []
    params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'attr_embedder' in name or 'obj_embedder' in name:
            if cfg.lr_word_embedding > 0:
                params_word_embedding.append(p)
        elif name.startswith('feat_extractor'):
            params_encoder.append(p)
        else:
            params.append(p)

    if cfg.lr_word_embedding > 0:
        optimizer = torch.optim.Adam([
            {'params': params_encoder, 'lr': cfg.lr_encoder},
            {'params': params_word_embedding, 'lr': cfg.lr_word_embedding},
            {'params': params, 'lr': cfg.lr},
        ], lr=cfg.lr, weight_decay=cfg.wd)  
        group_lrs = [cfg.lr_encoder, cfg.lr_word_embedding, cfg.lr]
    else:
        optimizer = torch.optim.Adam([
            {'params': params_encoder, 'lr': cfg.lr_encoder},
            {'params': params, 'lr': cfg.lr},
        ], lr=cfg.lr, weight_decay=cfg.wd)
        group_lrs = [cfg.lr_encoder, cfg.lr]

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True, drop_last=False)

    for epoch in range(cfg.start_epoch, cfg.max_epoch):
        if epoch == cfg.aug_epoch:
            print('Start Minority Attribute Augmentation')
            freeze(model.feat_extractor)
            trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.batch_size // 2, shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True, drop_last=False)

        model.train()
        if not cfg.finetune_backbone and cfg.aug_epoch <= epoch:
            freeze(model.feat_extractor)

        list_meters = [
            'loss_total'
        ]
        if cfg.use_obj_loss:
            list_meters.append('loss_aux_obj')
            list_meters.append('acc_aux_obj')
        if cfg.use_attr_loss:
            list_meters.append('loss_aux_attr')
            list_meters.append('acc_aux_attr')
        if cfg.use_emb_pair_loss:
            list_meters.append('emb_loss')
        if cfg.use_composed_pair_loss:
            list_meters.append('composed_unseen_loss')
            list_meters.append('composed_seen_loss')

        dict_meters = { 
            k: AverageMeter() for k in list_meters
        }

        acc_attr_meter = AverageMeter()
        acc_obj_meter = AverageMeter()
        acc_pair_meter = AverageMeter()

        start_iter = (epoch - 1) * len(trainloader)
        for idx, batch in enumerate(trainloader):
            it = start_iter + idx + 1

            for k in batch:
                if isinstance(batch[k], list): 
                    continue
                batch[k] = batch[k].cuda().to(non_blocking=True)
            r = np.random.rand(1)
            if cfg.aug_epoch <= epoch and 0 <= r and r < 0.5:
                out = model(batch, flag=True)
            else:
                out = model(batch)
            loss = out['loss_total']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if 'acc_attr' in out:
                acc_attr_meter.update(out['acc_attr'])
                acc_obj_meter.update(out['acc_obj'])
            acc_pair_meter.update(out['acc_pair'])
            for k in out:
                if k in dict_meters:
                    dict_meters[k].update(out[k].item())

            if (idx + 1) % cfg.disp_interval == 0:
                logger.info(f'Epoch: {epoch} Iter: {idx+1}/{len(trainloader)}, '
                    f'Loss: {dict_meters["loss_total"].avg:.3f}, '
                    f'Acc_Obj: {acc_obj_meter.avg:.2f}, '
                    f'Acc_Attr: {acc_attr_meter.avg:.2f}, '
                    f'Acc_Pair: {acc_pair_meter.avg:.2f}, ')


                acc_pair_meter.reset()
                if 'acc_attr' in out:
                    acc_attr_meter.reset()
                    acc_obj_meter.reset()
                for k in out:
                    if k in dict_meters:
                        dict_meters[k].reset()

        # optimize lr
        def decay_learning_rate_milestones(group_lrs, optimizer, epoch, cfg):
            """Decays learning rate following milestones in cfg.
            """
            milestones = cfg.lr_decay_milestones
            it = bisect_right(milestones, epoch)
            gamma = cfg.decay_factor ** it
            
            gammas = [gamma] * len(group_lrs)
            assert len(optimizer.param_groups) == len(group_lrs)
            i = 0
            for param_group, lr, gamma_group in zip(optimizer.param_groups, group_lrs, gammas):
                param_group["lr"] = lr * gamma_group
                i += 1
                print(f"Group {i}, lr = {lr * gamma_group}")
        
        if cfg.decay_strategy == 'milestone':
            decay_learning_rate_milestones(group_lrs, optimizer, epoch, cfg)

        # validation
        evaluator_val_ge = Evaluator(val_dataset, model)
        evaluator_test_ge = Evaluator(test_dataset, model)

        # valloader = torch.utils.data.DataLoader(
        #     val_dataset, batch_size=cfg.test_batch_size, shuffle=False,
        #     num_workers=cfg.num_workers)
        # auc_val, best_hm_val = validate_ge(epoch, model, valloader, evaluator_val_ge, topk=cfg.topk) 
        testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.test_batch_size, shuffle=False,
        num_workers=cfg.num_workers)
        auc, best_hm = validate_ge(epoch, model, testloader, evaluator_test_ge, logger, topk=cfg.topk) 

        print('auc: ', auc)
        print('hm: ',  best_hm)
        # TODO: add logger codes

def validate_ge(epoch, model, testloader, evaluator, logger, topk=1):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dset = testloader.dataset
    val_attrs, val_objs = zip(*dset.pairs)
    val_attrs = [dset.attr2idx[attr] for attr in val_attrs]
    val_objs = [dset.obj2idx[obj] for obj in val_objs]
    model.val_attrs = torch.LongTensor(val_attrs).cuda()
    model.val_objs = torch.LongTensor(val_objs).cuda()
    model.val_pairs = dset.pairs

    _, _, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []
    for _, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):
        for k in data:
            try:
                data[k] = data[k].to(device, non_blocking=True)
            except AttributeError:
                continue
        out = model(data)
        predictions = out['scores']

        attr_truth, obj_truth, pair_truth = data['attr'], data['obj'], data['pair']

        all_pred.append(predictions)
        all_attr_gt.append(attr_truth)
        all_obj_gt.append(obj_truth)
        all_pair_gt.append(pair_truth)

    all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to(
        'cpu'), torch.cat(all_pair_gt).to('cpu')

    all_pred_dict = {}
    # Gather values as dict of (attr, obj) as key and list of predictions as values
    for k in all_pred[0].keys():
        all_pred_dict[k] = torch.cat(
            [all_pred[i][k].to('cpu') for i in range(len(all_pred))])

    # Calculate best unseen accuracy
    results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=1e3, topk=topk)
    stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict, topk=topk)

    stats['a_epoch'] = epoch

    result = ''
    # write to Tensorboard
    for key in stats:
        result = result + key + '  ' + str(round(stats[key], 4)) + '| '


    logger.info(f'Val Epoch: {epoch}'+result)

    del model.val_attrs
    del model.val_objs

    return stats['AUC'], stats['best_hm']
