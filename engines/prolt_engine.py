
import os
import logging
import numpy as np
import torch
import test as test
from bisect import bisect_right
import torch.nn.functional as F

from utils import AverageMeter
from engines.evaluator_ge import Evaluator
import tqdm

def train_prolt(models, train_dataset, val_dataset, test_dataset, cfg, logger):
    image_extractor, model, image_decoupler = models
    device = cfg.device
    cfg.extractor = image_extractor
    evaluator_val =  Evaluator(test_dataset, model)
    start_epoch = 0

    trainloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers)
    testloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=cfg.test_batch_size,
    shuffle=False,
    num_workers=cfg.num_workers)


    p_log = dict()
    p_log['attr'] = torch.log(torch.ones((1,int(model.train_pairs.shape[0]))))
    p_log['objs'] = torch.log(torch.ones((1,int(model.train_pairs.shape[0]))))

    p_log['test_a'] = torch.log(torch.ones((1,int(model.val_pairs.shape[0]))))
    p_log['test_o'] = torch.log(torch.ones((1,int(model.val_pairs.shape[0]))))

    for epoch in tqdm.tqdm(range(start_epoch, cfg.max_epochs + 1), desc = 'Current epoch'):
        p_log['attr'] = p_log['attr'].to(device)
        p_log['objs'] = p_log['objs'].to(device)
        model.p_log = p_log

        model.args.if_ds = False
        model.freeze_model(model.C_y ,False)
        optimizer = optimizer_builder(cfg, model, image_extractor, image_decoupler)
        p_log = train_epoch(cfg, epoch, image_extractor, model, trainloader, optimizer, p_log, image_decoupler, logger)
        if epoch % cfg.eval_val_every == 0:
            with torch.no_grad(): # todo: might not be needed
                auc = test(epoch, image_extractor, model, testloader, evaluator_val, cfg, image_decoupler, logger)

        if auc > best_val_auc:
            best_val_auc = auc
            counter = 0
            print('new val auc is',"{:.2f}%".format(auc*100))
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    counter = 0
    best_val_auc = 0
    patience = patience*10
    for epoch in tqdm.tqdm(range(start_epoch, cfg.max_epochs + 1), desc = 'Current epoch'):
        p_log['attr'] = p_log['attr'].to(device)
        p_log['objs'] = p_log['objs'].to(device)
        model.p_log = p_log
        model.args.if_ds = True
        model.freeze_model(model.C_y, True)
        optimizer = optimizer_builder(cfg, model, image_extractor, image_decoupler)
        p_log = train_epoch(cfg, epoch, image_extractor, model, trainloader, optimizer, p_log, image_decoupler, logger)
        if epoch % cfg.eval_val_every == 0:
            with torch.no_grad():  # todo: might not be needed
                auc = test(epoch, image_extractor, model, testloader, evaluator_val, cfg, image_decoupler, logger)

        if auc > best_val_auc:
            best_val_auc = auc
            counter = 0
            print('new val auc is', "{:.2f}%".format(auc * 100))
            embedding_save_path = os.path.join(cfg.logdir, 'Best_AUC_Embedding.pth')
            torch.save(model.state_dict(), embedding_save_path)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

def train_epoch(cfg, epoch, image_extractor, model, trainloader, optimizer, p_log, img_decoupler, logger):
    '''
    Runs training for an epoch
    '''
    device = cfg.device
    if image_extractor:
        image_extractor.train()
    model.train()
    train_loss = 0.0
    for idx, data in tqdm.tqdm(enumerate(trainloader), total=len(trainloader), desc = 'Training'):
        data  = [d.to(device) for d in data]

        if image_extractor:
            img = data[0]
            data[0] = image_extractor(data[0])
            data.append(img_decoupler(img))
        loss, pred = model(data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx == 0:
            pred_attr = pred[0].detach().cpu().numpy()
            pred_objs = pred[1].detach().cpu().numpy()

        else:
            pred_attr = np.concatenate((pred_attr, pred[0].detach().cpu().numpy()))
            pred_objs = np.vstack([pred_objs, pred[1].detach().cpu().numpy()])

        train_loss += loss.item()
    train_loss = train_loss/len(trainloader)
    if (idx % len(trainloader)//8) == 0:
        print('Epoch: {}| Loss: {}'.format(epoch, round(train_loss, 2)))
        logger.info('Epoch: {}| Loss: {}'.format(epoch, round(train_loss, 2)))

    '''k(s,o)'''
    pred_attr = np.mean(pred_attr , axis=0)
    pred_objs = np.mean(pred_objs , axis=0)
    p_log['attr'] = torch.from_numpy(pred_objs)
    p_log['objs'] = torch.from_numpy(pred_attr)
    p_log['attr'] = F.softmax(p_log['attr'],dim=-1)
    p_log['objs'] = F.softmax(p_log['objs'],dim=-1)
    p_log['attr'] = np.log(p_log['attr'])
    p_log['objs'] = np.log(p_log['objs'])

    return p_log

def test(epoch, image_extractor, model, testloader, evaluator, cfg, image_decoupler, logger):
    '''
    Runs testing for an epoch
    '''
    global best_auc, best_hm, best_obj,best_attr,best_seen,best_unseen, latest_changes
    device = cfg.device
    if image_extractor:
        image_extractor.eval()
        image_decoupler.eval()

    model.eval()

    accuracies, all_sub_gt, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []

    for idx, data in tqdm.tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):
        data = [d.to(device) for d in data]

        if image_extractor:
            img = data[0]
            data[0] = image_extractor(data[0])
            data.append(image_decoupler(img))

        _, predictions = model(data)

        attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]

        all_pred.append(predictions)
        all_attr_gt.append(attr_truth)
        all_obj_gt.append(obj_truth)
        all_pair_gt.append(pair_truth)


    all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to(
            'cpu'), torch.cat(all_pair_gt).to('cpu')

    all_pred_dict = {}
    # Gather values as dict of (attr, obj) as key and list of predictions as values
    if cfg.cpu_eval:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k].to('cpu') for i in range(len(all_pred))])
    else:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k] for i in range(len(all_pred))])

    # Calculate best unseen accuracy
    results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=cfg.bias, topk=cfg.topk)
    stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict, topk=cfg.topk)
    
    result = ''
    for key in stats:
        result = result + key + '  ' + str(round(stats[key], 4)) + '| '
    
    logger.info(f'Epoch{epoch}: '+result)

    return stats['AUC']


def optimizer_builder(args,model,image_extractor,image_decoupler):
    model_params = [param for name, param in model.named_parameters() if param.requires_grad]
    optim_params = [{'params':model_params}]
    if args.update_image_features:
        ie_parameters = [param for name, param in image_extractor.named_parameters()]
        optim_params.append({'params': ie_parameters,
                            'lr': args.lrg})

        ie_parameters = [param for name, param in image_decoupler.named_parameters()]
        optim_params.append({'params': ie_parameters,
                            'lr': args.lrg})
    optimizer = torch.optim.Adam(optim_params, lr=args.lr, weight_decay=args.wd)
    return optimizer