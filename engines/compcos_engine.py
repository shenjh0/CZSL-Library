import os

import torch
import test as test
from engines.evaluator_ge import Evaluator
from tqdm import tqdm


def train_compcos(models, train_dataset, val_dataset, test_dataset, config, logger):
    image_extractor, model = models
    config.extractor = image_extractor

    train = train_epoch

    evaluator_val =  Evaluator(test_dataset, model)

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers)
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.num_workers)

    model_params = [param for name, param in model.named_parameters() if param.requires_grad]
    optim_params = [{'params':model_params}]
    if config.update_image_features:
        ie_parameters = [param for name, param in image_extractor.named_parameters()]
        optim_params.append({'params': ie_parameters,
                            'lr': config.lrg})
    optimizer = torch.optim.Adam(optim_params, lr=config.lr, weight_decay=config.wd)


    start_epoch = 0
    # Load checkpoint
    if config.load is not None:
        checkpoint = torch.load(config.load)
        if image_extractor:
            try:
                image_extractor.load_state_dict(checkpoint['image_extractor'])
                if config.freeze_features:
                    print('Freezing image extractor')
                    image_extractor.eval()
                    for param in image_extractor.parameters():
                        param.requires_grad = False
            except:
                print('No Image extractor in checkpoint')
        model.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        print('Loaded model from ', config.load)
    
    best_auc = best_hm = 0
    for epoch in tqdm(range(start_epoch, config.max_epochs + 1), desc = 'Current epoch'):
        train(config, epoch, image_extractor, model, trainloader, optimizer, logger)
        if epoch % config.eval_val_every == 0:
            with torch.no_grad(): # todo: might not be needed
                best_auc, best_hm = test(epoch, image_extractor, model, testloader, evaluator_val, config, logger,
                     best_auc, best_hm)
    print('Best AUC achieved is ', best_auc)
    print('Best HM achieved is ', best_hm)


def train_epoch(cfg, epoch, image_extractor, model, trainloader, optimizer, logger):
    '''
    Runs training for an epoch
    '''

    if image_extractor:
        image_extractor.train()
    model.train() # Let's switch to training
    device = cfg.device
    train_loss = 0.0 
    for idx, data in tqdm(enumerate(trainloader), total=len(trainloader), desc = 'Training'):
        data  = [d.to(device) for d in data]

        if image_extractor:
            data[0] = image_extractor(data[0])
        
        loss, _ = model(data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        train_loss += loss.item()
        if (idx % (len(trainloader) // 8) == 0):
            logger.info('|----Train Loss: {:.4f}'.format(loss.item()))

    train_loss = train_loss/len(trainloader)
    print('Epoch: {}| Loss: {}'.format(epoch, round(train_loss, 2)))
    logger.info('Epoch: {}| Loss: {}'.format(epoch, round(train_loss, 2)))


def test(epoch, image_extractor, model, testloader, evaluator, config, logger, best_auc, best_hm):
    '''
    Runs testing for an epoch
    '''
    device = config.device

    def save_checkpoint(filename):
        state = {
            'net': model.state_dict(),
            'epoch': epoch,
            'AUC': stats['AUC']
        }
        if image_extractor:
            state['image_extractor'] = image_extractor.state_dict()
        torch.save(state, os.path.join(config.logdir, 'ckpt_{}.t7'.format(filename)))

    if image_extractor:
        image_extractor.eval()

    model.eval()

    accuracies, all_sub_gt, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []

    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing'):
        data = [d.to(device) for d in data]

        if image_extractor:
            data[0] = image_extractor(data[0])

        _, predictions = model(data)

        attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]

        all_pred.append(predictions)
        all_attr_gt.append(attr_truth)
        all_obj_gt.append(obj_truth)
        all_pair_gt.append(pair_truth)

    if config.cpu_eval:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt), torch.cat(all_obj_gt), torch.cat(all_pair_gt)
    else:
        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to(
            'cpu'), torch.cat(all_pair_gt).to('cpu')

    all_pred_dict = {}
    # Gather values as dict of (attr, obj) as key and list of predictions as values
    if config.cpu_eval:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k].to('cpu') for i in range(len(all_pred))])
    else:
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k] for i in range(len(all_pred))])

    # Calculate best unseen accuracy
    results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=config.bias, topk=config.topk)
    stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict, topk=config.topk)

    stats['a_epoch'] = epoch

    result = ''
    # write to Tensorboard
    for key in stats:
        result = result + key + '  ' + str(round(stats[key], 4)) + '| '
    
    logger.info(f'epoch{epoch}: '+ result)

    result = result + config.name
    print(f'Test Epoch: {epoch}')
    print(result)
    if epoch > 0 and epoch % config.save_every == 0:
        save_checkpoint(epoch)
    if stats['AUC'] > best_auc:
        best_auc = stats['AUC']
        print('New best AUC ', best_auc)
        save_checkpoint('best_auc')

    if stats['best_hm'] > best_hm:
        best_hm = stats['best_hm']
        print('New best HM ', best_hm)
        save_checkpoint('best_hm')
    return best_auc, best_hm