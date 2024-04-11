
import os
import logging
import numpy as np
import torch
import tqdm
from torch.utils.data.dataloader import DataLoader
import test as test
from bisect import bisect_right

from utils import AverageMeter
from util import evaluator_ge


def train(model, optimizer, config, train_dataset, val_dataset, test_dataset):

    if config.model_type.upper() == "COT":
        train_cot(model, optimizer, train_dataset, val_dataset, test_dataset, config)
    else:
        train_dataloader = DataLoader(
            train_dataset,
            num_workers = 16,
            batch_size=config.train_batch_size,
            shuffle=True
        )

        model.train()
        best_loss = 1e5
        best_metric = 0
        
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        attr2idx = train_dataset.attr2idx
        obj2idx = train_dataset.obj2idx

        train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                    for attr, obj in train_dataset.train_pairs]).cuda()

        train_losses = []

        for i in range(config.epoch_start, config.epochs):
            progress_bar = tqdm.tqdm(
                total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
            )
            epoch_train_losses = []
            for bid, batch in enumerate(train_dataloader):
                # if bid > 1:
                #     break
                predict, loss = model(batch, train_pairs)

                # normalize loss to account for batch accumulation
                loss = loss / config.gradient_accumulation_steps

                # backward pass
                loss.backward()

                # weights update
                if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_train_losses.append(loss.item())
                progress_bar.set_postfix({"train loss": np.mean(epoch_train_losses[-50:])})
                progress_bar.update()
            # scheduler.step()
            progress_bar.close()
            progress_bar.write(f"epoch {i +1} train loss {np.mean(epoch_train_losses)}")
            train_losses.append(np.mean(epoch_train_losses))

            if (i + 1) % config.save_every_n == 0:
                torch.save(model.state_dict(), os.path.join(config.save_path, f"epoch_{i}.pt"))
            if config.update == True:
                model.update_status(i)

            if i < config.jump_epoch:
                continue

            print("Evaluating val dataset:")
            logging.info("Evaluating val dataset:")
            loss_avg, val_result = evaluate(model, val_dataset, config)
            print("Now status is {}".format(model.train_status))
            logging.info("Now status is {}".format(model.train_status))

            print("Loss average on val dataset: {}".format(loss_avg))
            print("Evaluating test dataset:")
            logging.info("Evaluating test dataset:")
            evaluate(model, test_dataset, config)
            if config.best_model_metric == "best_loss":
                if loss_avg.cpu().float() < best_loss:
                    best_loss = loss_avg.cpu().float()
                    print("Evaluating test dataset:")
                    evaluate(model, test_dataset, config)
                    torch.save(model.state_dict(), os.path.join(
                    config.save_path, f"best.pt"
                ))
            else:
                if val_result[config.best_model_metric] > best_metric:
                    best_metric = val_result[config.best_model_metric]
                    print("Evaluating test dataset:")
                    evaluate(model, test_dataset, config)
                    torch.save(model.state_dict(), os.path.join(
                    config.save_path, f"best.pt"
                ))
            if i + 1 == config.epochs:
                print("Evaluating test dataset on Closed World")
                model.load_state_dict(torch.load(os.path.join(config.save_path, f"best.pt")))
                evaluate(model, test_dataset, config)

        if config.save_model:
            torch.save(model.state_dict(), os.path.join(config.save_path, f'final_model.pt'))


def evaluate(model, dataset, config):
    model.eval()
    evaluator = test.Evaluator(dataset, model=None)
    all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = test.predict_logits(
            model, dataset, config)
    test_stats = test.test(
            dataset,
            evaluator,
            all_logits,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt,
            config
        )
    result = ""
    key_set = ["best_seen", "best_unseen", "AUC", "best_hm", "attr_acc", "obj_acc"]
    for key in test_stats:
        if key in key_set:
            result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
    print(result)   
    logging.info(result)
    model.train()
    return loss_avg, test_stats

def train_cot(model, optimizer, train_dataset, val_dataset, test_dataset, cfg):
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
            if cfg.TRAIN.lr_word_embedding > 0:
                params_word_embedding.append(p)
        elif name.startswith('feat_extractor'):
            params_encoder.append(p)
        else:
            params.append(p)

    if cfg.lr_word_embedding > 0:
        optimizer = torch.optim.Adam([
            {'params': params_encoder, 'lr': cfg.TRAIN.lr_encoder},
            {'params': params_word_embedding, 'lr': cfg.TRAIN.lr_word_embedding},
            {'params': params, 'lr': cfg.TRAIN.lr},
        ], lr=cfg.TRAIN.lr, weight_decay=cfg.TRAIN.wd)  
        group_lrs = [cfg.TRAIN.lr_encoder, cfg.TRAIN.lr_word_embedding, cfg.TRAIN.lr]
    else:
        optimizer = torch.optim.Adam([
            {'params': params_encoder, 'lr': cfg.TRAIN.lr_encoder},
            {'params': params, 'lr': cfg.TRAIN.lr},
        ], lr=cfg.TRAIN.lr, weight_decay=cfg.TRAIN.wd)
        group_lrs = [cfg.TRAIN.lr_encoder, cfg.TRAIN.lr]

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
        for idx, batch in enumerate(tqdm(trainloader)):
            it = start_iter + idx + 1

            for k in batch:
                if isinstance(batch[k], list): 
                    continue
                batch[k] = batch[k].to(non_blocking=True)
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
                print(
                    f'Epoch: {epoch} Iter: {idx+1}/{len(trainloader)}, '
                    f'Loss: {dict_meters["loss_total"].avg:.3f}, '
                    f'Acc_Obj: {acc_obj_meter.avg:.2f}, '
                    f'Acc_Attr: {acc_attr_meter.avg:.2f}, '
                    f'Acc_Pair: {acc_pair_meter.avg:.2f}, ',
                    flush=True)

                # TODO: add logger
                # for k in out:
                #     if k in dict_meters:
                #         logger.add_scalar('train/%s' % k, dict_meters[k].avg, it)
                # logger.add_scalar('train/acc_pair', acc_pair_meter.avg, it)

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
            milestones = cfg.TRAIN.lr_decay_milestones
            it = bisect_right(milestones, epoch)
            gamma = cfg.TRAIN.decay_factor ** it
            
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
        evaluator_val_ge = evaluator_ge.Evaluator(val_dataset, model)
        evaluator_test_ge = evaluator_ge.Evaluator(test_dataset, model)

        testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.test_batch_size, shuffle=False,
        num_workers=cfg.num_workers)
        valloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.test_batch_size, shuffle=False,
            num_workers=cfg.num_workers)

        auc_val, best_hm_val = evaluator_ge.validate_ge(epoch, model, valloader, evaluator_val_ge, topk=cfg.topk) 
        auc, best_hm = evaluator_ge.validate_ge(epoch, model, testloader, evaluator_test_ge, topk=cfg.topk) 

        print('auc: ', auc)
        print('hm: ',  best_hm)
        # TODO: add logger codes
