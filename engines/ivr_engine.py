import os
import torch
from torch.utils.data.dataloader import DataLoader

from engines.evaluator_ge import Evaluator

def train_epoch(cfg, epoch, image_extractor, model, trainloader, optimizer, logger):
    '''
    Runs training for an epoch
    '''
    if cfg.update_image_features: 
        image_extractor.train()
    model = model.train() 
    train_loss = 0.0
    for idx, data in enumerate(trainloader):
        data = [d.to(cfg.device) for d in data]
        if cfg.update_image_features:
            data[0] = image_extractor(data[0])
        loss = model(data)[0]

        if loss == None:
            return

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        train_loss += loss.item()
    
        if (idx % (len(trainloader) // 8) == 0):
            logger.info('|----Train Loss: {:.4f}'.format(loss.item()))

def test(cfg, epoch, image_extractor, model, testloader, evaluator, logger, *best_list):
    '''
    Runs testing for an epoch
    '''
    def save_checkpoint(filename):
        state = {
            'epoch': epoch+1,
            'AUC': stats['AUC']
        }
        state['net'] = model.state_dict()
        if image_extractor:
            state['image_extractor'] = image_extractor.state_dict()
        torch.save(state, os.path.join(cfg.logdir,'ckpt_{}_{}.t7'.format(filename, cfg.dataset)))

    if cfg.update_image_features: image_extractor.eval()
    model = model.eval()

    all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], []
    
    for idx, data in enumerate(testloader):
        data = [d.to(cfg.device) for d in data]
        if cfg.update_image_features:
            data[0] = image_extractor(data[0])
        predictions = model(data)[1]

        attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]
        all_pred.append(predictions)
        all_attr_gt.append(attr_truth)
        all_obj_gt.append(obj_truth)
        all_pair_gt.append(pair_truth)

    del predictions, attr_truth, obj_truth, pair_truth

    all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to(
            'cpu'), torch.cat(all_pair_gt).to('cpu')

    best_attr, best_obj, best_seen, best_unseen, best_auc, best_hm, best_epoch = best_list
    # Gather values as dict of (attr, obj) as key and list of predictions as values
    all_pred_dict = {}
    for k in all_pred[0].keys():
        all_pred_dict[k] = torch.cat([all_pred[i][k].cpu() for i in range(len(all_pred))])
    del all_pred

    # Calculate best unseen accuracy
    results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=cfg.bias, topk=cfg.topk)
    stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict, topk=cfg.topk)

    stats['a_epoch'] = epoch
    
    # print(result)
    attr_acc = stats['closed_attr_match']
    obj_acc = stats['closed_obj_match']
    seen_acc = stats['best_seen']
    unseen_acc = stats['best_unseen']
    HM = stats['best_hm']
    AUC = stats['AUC']
    print('|----Test {} Epoch: Attr Acc: {:.2f}% | Obj Acc: {:.2f}% | Seen Acc: {:.2f}% | Unseen Acc: {:.2f}% | HM: {:.2f}% | AUC: {:.2f}'.\
        format(int(epoch), attr_acc*100, obj_acc*100, seen_acc*100, unseen_acc*100, HM*100, AUC*100))
    logger.info('|----Test {} Epoch: Attr Acc: {:.2f}% | Obj Acc: {:.2f}% | Seen Acc: {:.2f}% | Unseen Acc: {:.2f}% | HM: {:.2f}% | AUC: {:.2f}'.\
        format(int(epoch), attr_acc*100, obj_acc*100, seen_acc*100, unseen_acc*100, HM*100, AUC*100))

    if epoch > 0 and epoch % cfg.save_every == 0:
        save_checkpoint(epoch)
    if AUC > best_auc:
        best_auc = AUC
        best_attr = attr_acc
        best_obj = obj_acc
        best_seen = seen_acc
        best_unseen = unseen_acc
        best_hm = HM
        best_epoch = epoch
        print('|----New Best AUC {:.2f}. SAVE to local disk!'.format(best_auc*100))
        logger.info('|----New Best AUC {:.2f}. SAVE to local disk!'.format(best_auc*100))
        save_checkpoint('best_auc')
    return best_attr ,best_obj ,best_seen ,best_unseen ,best_auc ,best_hm ,best_epoch

def train_ivr(models, train_dataset, val_dataset, test_dataset, config, logger):
    image_extractor, model = models
    trainloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers)
    testloader = DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.num_workers) 
    evaluator = Evaluator(test_dataset, model)

    # Initialize optimizer
    model_params = [param for _, param in model.named_parameters() if param.requires_grad]
    optim_params = [{'params':model_params}]
    if config.update_image_features:
        ie_parameters = [param for _, param in image_extractor.named_parameters()]
        optim_params.append({'params': ie_parameters, 'lr': config.lrg})
    optimizer = torch.optim.Adam(optim_params, lr=config.lr, weight_decay=config.wd)

    # 
    best_attr = best_obj = best_seen = best_unseen = best_auc = best_hm = best_epoch = 0.0
    for epoch in range(config.max_epochs):
        print('Epoch {} | Best Attr: {:.2f}% | Best Obj: {:.2f}% | Best Seen: {:.2f}% | Best Unseen: {:.2f}% | Best HM: {:.2f}% | Best AUC: {:.2f} | Best Epoch: {:.0f}'.\
            format(epoch+1, best_attr*100, best_obj*100, best_seen*100, best_unseen*100, best_hm*100, best_auc*100, best_epoch))
        train_epoch(config, epoch, image_extractor, model, trainloader, optimizer, logger)
        with torch.no_grad():
            best_attr ,best_obj ,best_seen ,best_unseen ,best_auc ,best_hm ,best_epoch=test(
                config, epoch, image_extractor, model, testloader, evaluator, logger, 
                best_attr ,best_obj ,best_seen ,best_unseen ,best_auc ,best_hm ,best_epoch)    
        
    logger.info('======>The train and test pipeline of CANet is done.')