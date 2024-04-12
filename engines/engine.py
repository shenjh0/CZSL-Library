
import os
import logging
import numpy as np
import torch
import tqdm
from torch.utils.data.dataloader import DataLoader
import test as test
from bisect import bisect_right

from utils import AverageMeter
from engines import evaluator_ge
from engines.cot_engine import train_cot
from engines.canet_engine import train_canet
from engines.scen_engine import train_scen
from engines.ivr_engine import train_ivr


def enginer(model, config, train_dataset, val_dataset, test_dataset, logger):    
    if config.model_type.upper() == "COT":
        train_cot(model,  train_dataset, val_dataset, test_dataset, config, logger)
    if config.model_type.upper() == "CANET":
        train_canet(model, train_dataset, val_dataset, test_dataset, config, logger)
    if config.model_type.upper() == "SCEN":
        train_scen(model, train_dataset, val_dataset, test_dataset, config, logger)
    if config.model_type.upper() == "IVR":
        train_ivr(model, train_dataset, val_dataset, test_dataset, config, logger)

    else:
        os.makedirs(config.save_path, exist_ok=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
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
