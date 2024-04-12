
def test(cfg, epoch, image_extractor, model, testloader, evaluator, logger, *best_list):
    print(cfg)
    print(epoch)
    print(image_extractor)
    print(model)
    print(best_list)
    best_attr, best_obj, best_seen, best_unseen, best_auc, best_hm, best_epoch = best_list

    print(best_attr)
    print(best_auc)


best_attr = best_obj = best_seen = best_unseen = best_auc = best_hm = best_epoch = 0.0
best_attr = 'ss'
best_auc = '12'
test(1,2,3,4,5,6,7,
     best_attr ,best_obj ,best_seen ,best_unseen ,best_auc ,best_hm ,best_epoch)