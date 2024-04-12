from itertools import product
import os
import random
from random import choice
import numpy as np
import torch
import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, RandomHorizontalFlip,
                                    RandomPerspective, RandomRotation, Resize,
                                    ToTensor)
from torchvision import transforms, models
import torch.nn as nn
from torchvision.transforms.transforms import RandomResizedCrop

BICUBIC = InterpolationMode.BICUBIC
n_px = 224


def transform_image(split="train", imagenet=False):
    if imagenet:
        # from czsl repo.
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = Compose(
            [
                RandomResizedCrop(n_px),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(
                    mean,
                    std,
                ),
            ]
        )
        return transform

    if split == "test" or split == "val":
        transform = Compose(
            [
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    else:
        transform = Compose(
            [
                # RandomResizedCrop(n_px, interpolation=BICUBIC),
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                RandomHorizontalFlip(),
                RandomPerspective(),
                RandomRotation(degrees=5),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    return transform

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = '%s/%s' % (self.img_dir, img)
        img = Image.open(file).convert('RGB')
        return img

class CompositionDataset(Dataset):
    def __init__(
            self,
            root,
            phase,
            split='compositional-split-natural',
            open_world=False,
            imagenet=False
            # inductive=True
    ):
        self.root = root
        self.phase = phase
        self.split = split
        self.open_world = open_world

        # new addition
        # if phase == 'train':
        #     self.inductive = inductive
        # else:
        #     self.inductive = False

        self.feat_dim = None
        self.transform = transform_image(phase, imagenet=imagenet)
        self.loader = ImageLoader(self.root + '/images/')

        self.attrs, self.objs, self.pairs, \
                self.train_pairs, self.val_pairs, \
                self.test_pairs = self.parse_split()

        if self.open_world:
            self.pairs = list(product(self.attrs, self.objs))

        self.train_data, self.val_data, self.test_data = self.get_split_info()
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        else:
            self.data = self.test_data

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        print('# train pairs: %d | # val pairs: %d | # test pairs: %d' % (len(
            self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('# train images: %d | # val images: %d | # test images: %d' %
              (len(self.train_data), len(self.val_data), len(self.test_data)))

        self.train_pair_to_idx = dict(
            [(pair, idx) for idx, pair in enumerate(self.train_pairs)]
        )

        if self.open_world:
            mask = [1 if pair in set(self.train_pairs) else 0 for pair in self.pairs]
            self.seen_mask = torch.BoolTensor(mask) * 1.

            self.obj_by_attrs_train = {k: [] for k in self.attrs}
            for (a, o) in self.train_pairs:
                self.obj_by_attrs_train[a].append(o)

            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.attrs_by_obj_train = {k: [] for k in self.objs}
            for (a, o) in self.train_pairs:
                self.attrs_by_obj_train[o].append(a)

    def get_split_info(self):
        data = torch.load(self.root + '/metadata_{}.t7'.format(self.split))
        train_data, val_data, test_data = [], [], []
        for instance in data:
            image, attr, obj, settype = instance['image'], instance[
                'attr'], instance['obj'], instance['set']

            if attr == 'NA' or (attr,
                                obj) not in self.pairs or settype == 'NA':
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue

            data_i = [image, attr, obj]
            if settype == 'train':
                train_data.append(data_i)
            elif settype == 'val':
                val_data.append(data_i)
            else:
                test_data.append(data_i)

        return train_data, val_data, test_data

    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                # pairs = [t.split() if not '_' in t else t.split('_') for t in pairs]
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            '%s/%s/train_pairs.txt' % (self.root, self.split))
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            '%s/%s/val_pairs.txt' % (self.root, self.split))
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            '%s/%s/test_pairs.txt' % (self.root, self.split))

        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
                list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def __getitem__(self, index):
        image, attr, obj = self.data[index]
        img = self.loader(image)
        img = self.transform(img)

        if self.phase == 'train':
            data = [img, self.attr2idx[attr], self.obj2idx[obj], self.train_pair_to_idx[(attr, obj)]]
        else:
            data = [img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]]

        return data

    def __len__(self):
        return len(self.data)

###                COT DATASET 

def imagenet_transform(phase):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if phase == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
   
    elif phase == 'test' or phase == 'val':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    return transform

class COTDataset(Dataset):
    def __init__(
        self,
        phase,
        split='compositional-split',
        open_world=False,
        cfg=None
    ):
        self.phase = phase
        self.cfg = cfg
        self.split = split
        self.open_world = open_world

        #num_negs
        self.num_negs = 128

        self.transform = imagenet_transform(phase)
        self.loader = ImageLoader(f'{cfg.dataset_path}/images')
        
        self.attrs, self.objs, self.pairs, \
            self.train_pairs, self.val_pairs, \
            self.test_pairs = self.parse_split()

        self.train_data, self.val_data, self.test_data = self.get_split_info()
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        else:
            self.data = self.test_data

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}
        self.train_pair2idx = {pair: idx for idx, pair in enumerate(self.train_pairs)}

        print('# train pairs: %d | # val pairs: %d | # test pairs: %d' % (len(
            self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('# train images: %d | # val images: %d | # test images: %d' %
              (len(self.train_data), len(self.val_data), len(self.test_data)))


        self.obj_affordance = {} # -> contains objects compatible with an attribute.
        for _obj in self.objs:
            candidates = [
                attr
                for (_, attr, obj) in self.train_data
                if obj == _obj
            ]
            self.obj_affordance[_obj] = sorted(list(set(candidates)))

        # Keeping a list of all pairs that occur with each object

        self.sample_indices = list(range(len(self.data)))
        self.sample_pairs = self.train_pairs

        # Images that contain an object.
        self.image_with_obj = {}
        self.image_with_obj_hasattr = {}
        for i, instance in enumerate(self.train_data):
            obj = instance[2]
            if obj not in self.image_with_obj:
                self.image_with_obj[obj] = []
                self.image_with_obj_hasattr[obj] = []
            self.image_with_obj[obj].append(i)
            self.image_with_obj_hasattr[obj].append(self.attr2idx[instance[1]])
        
        # Images that contain an attribute.
        self.image_with_attr = {}
        for i, instance in enumerate(self.train_data):
            attr = instance[1]
            if attr not in self.image_with_attr:
                self.image_with_attr[attr] = []
            self.image_with_attr[attr].append(i)

        # Images that contain a pair.
        self.image_with_pair = {}
        for i, instance in enumerate(self.train_data):
            attr, obj = instance[1], instance[2]
            if (attr, obj) not in self.image_with_pair:
                self.image_with_pair[(attr, obj)] = []
            self.image_with_pair[(attr, obj)].append(i)
        
        unseen_pairs = set()
        for pair in self.val_pairs + self.test_pairs:
            if pair not in self.train_pair2idx:
                unseen_pairs.add(pair)
        self.unseen_pairs = list(unseen_pairs)
        self.unseen_pair2idx = {pair: idx for idx, pair in enumerate(self.unseen_pairs)}
            
    def get_split_info(self):
        data = torch.load(f'{self.cfg.dataset_path}/metadata_{self.split}.t7')
        train_data, val_data, test_data = [], [], []

        for instance in data:
            image, attr, obj, settype = \
                instance['image'], instance['attr'], instance['obj'], instance['set']
            if attr == 'NA' or (attr, obj) not in self.pairs or settype == 'NA':
                continue
            data_i = [image, attr, obj]
            if settype == 'train':
                train_data.append(data_i)
            elif settype == 'val':
                val_data.append(data_i)
            else:
                test_data.append(data_i)

        return train_data, val_data, test_data

    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs
        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            f'{self.cfg.dataset_path}/{self.split}/train_pairs.txt')
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            f'{self.cfg.dataset_path}/{self.split}/val_pairs.txt')
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            f'{self.cfg.dataset_path}/{self.split}/test_pairs.txt')

        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
                list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def __getitem__(self, index):
        image, attr, obj = self.data[index]
        if self.cfg.use_precomputed_features:
            img = self.activations[image]
        else:
            img = self.loader(image)
            img = self.transform(img)
        unseen = 0
        if (attr, obj) in self.unseen_pairs:
            unseen = 1
        if self.phase == 'train':
            data = {
                'img': img,
                'attr': self.attr2idx[attr],
                'obj': self.obj2idx[obj],
                'pair': self.train_pair2idx[(attr, obj)],
                'unseen': unseen,
                'img_name': self.data[index][0]
            }
            
            # Object task.
            i3 = self.sample_same_object2(attr, obj)
            img1, attr1_o, obj1 = self.data[i3]

            if self.cfg.use_precomputed_features:
                img1 = self.activations[img1]
            else:
                img1 = self.loader(img1)
                img1 = self.transform(img1)
            data['img1_o'] = img1
            data['attr1_o'] = self.attr2idx[attr1_o]
            data['obj1_o'] = self.obj2idx[obj1]
            data['idx1_o'] = self.train_pair2idx[(attr1_o, obj1)]
            data['img1_name_o'] = self.data[i3][0]
        else:
            # Testing mode.

            data = {
                'img': img,
                'attr': self.attr2idx[attr],
                'obj': self.obj2idx[obj],
                'pair': self.pair2idx[(attr, obj)],
                'unseen': unseen,
                'name': (attr, obj)
            }
        return data

    def __len__(self):
        return len(self.data)

    def sample_negative(self, attr, obj):
        '''
        Inputs
            attr: String of valid attribute
            obj: String of valid object
        Returns
            Tuple of a different attribute, object indexes
        '''
        new_attr, new_obj = self.sample_pairs[np.random.choice(
            len(self.sample_pairs))]

        while new_attr == attr and new_obj == obj:
            new_attr, new_obj = self.sample_pairs[np.random.choice(
            len(self.sample_pairs))]

        return (self.attr2idx[new_attr], self.obj2idx[new_obj])

    def sample_same_object2(self, attr, obj):
        if len(self.obj_affordance[obj]) == 1:
            i2 = np.random.choice(self.image_with_obj[obj])
            return i2
        else:
            i2 = np.random.choice(self.image_with_obj[obj])
            ind, count = np.unique(self.image_with_obj_hasattr[obj], return_counts=True)
            weight = 1.0 / count[ind!=self.attr2idx[attr]]
            weight = weight / np.sum(weight)
            idx = np.random.choice(ind[ind!=self.attr2idx[attr]], p=weight)
            _, attr1, _ = self.data[i2]
            while self.attr2idx[attr1] != idx:
                i2 = np.random.choice(self.image_with_obj[obj])
                _, attr1, _ = self.data[i2]
            return i2


class CANetDataset(CompositionDataset):
    def __init__(
            self,
            config,
            root,
            phase,
            split='compositional-split',
            model='resnet18',
            update_image_features=False,
            train_only=False,
            **var
    ):
        super().__init__(
            root=root,
            phase=phase,
            split=split,
            open_world=config.open_world,
            imagenet=config.imagenet
        )
        self.config = config
        self.root = root
        self.phase = phase
        self.split = split
        self.update_image_features = update_image_features
        self.feat_dim = 512 if 'resnet18' in model else 2048  # todo, unify this with models
        self.device = config.device
        # attrs [115], objs [245], pairs [1962], train_pairs [1262], val_pairs [600], test_pairs [800]
        self.attrs, self.objs, self.pairs, self.train_pairs, \
        self.val_pairs, self.test_pairs = self.parse_split()
        self.train_data, self.val_data, self.test_data = self.get_split_info()
        self.full_pairs = list(product(self.attrs, self.objs))

        # Clean only was here
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.all_pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}  # all pairs [1962], for val in training

        if train_only and self.phase == 'train':
            print('  Using only train pairs during training')
            self.pair2idx = {pair: idx for idx, pair in enumerate(self.train_pairs)}  # train pairs [1262]
        else:
            print('  Using all pairs as classification classes during {} process'.format(self.phase))
            self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}  # all pairs [1962]

        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        elif self.phase == 'test':
            self.data = self.test_data
        elif self.phase == 'all':
            self.data = self.train_data + self.val_data + self.test_data
        else:
            raise ValueError('  Invalid training phase')
        print('Use data from {} set'.format(self.phase))

        self.all_data = self.train_data + self.val_data + self.test_data

        # Keeping a list of all pairs that occur with each object
        self.obj_affordance = {}
        self.train_obj_affordance = {}
        for _obj in self.objs:
            candidates = [attr for (_, attr, obj) in self.train_data + self.test_data if obj == _obj]
            self.obj_affordance[_obj] = list(set(candidates))

            candidates = [attr for (_, attr, obj) in self.train_data if obj == _obj]
            self.train_obj_affordance[_obj] = list(set(candidates))

        self.sample_indices = list(range(len(self.data)))
        self.sample_pairs = self.train_pairs

        # Load based on what to output
        self.transform = transform_image(self.phase)
        self.loader = ImageLoader(os.path.join(self.root, 'images'))

        if not self.update_image_features:
            feat_file = os.path.join(root, model + '_feature_vectors.t7')
            if not os.path.exists(feat_file):
                print('  Feature file not found. Now get one!')
                with torch.no_grad():
                    self.generate_features(feat_file, model)
            self.phase = phase
            print(f'  Using {model} and feature file {feat_file}')
            activation_data = torch.load(feat_file)
            self.activations = dict(
                zip(activation_data['files'], activation_data['features']))
            self.feat_dim = activation_data['features'].size(1)
  

    def generate_features(self, out_file, model):
        '''
        Inputs
            out_file: Path to save features
            model: String of extraction model
        '''
        import tqdm
        import glob
        from models.CANET.image_extractor import get_image_extractor
        # data = self.all_data
        data = os.path.join(self.root, 'images')
        files_before = glob(os.path.join(data, '**', '*.jpg'), recursive=True)
        files_all = []
        for current in files_before:
            parts = current.split('/')
            if "cgqa" in self.root:
                files_all.append(parts[-1])
            else:
                files_all.append(os.path.join(parts[-2], parts[-1]))
        transform = transform_image('test') # Do not use any image augmentation, because we have a trained image backbone
        feat_extractor = get_image_extractor(arch=model).eval()
        if not self.config.extract_feature_vectors:
            from torch.nn import Sequential
            feat_extractor = Sequential(*list(feat_extractor.children())[:-1])
        feat_extractor = feat_extractor.to(self.device)

        image_feats = []
        image_files = []
        for chunk in tqdm(chunks(files_all, 1024), total=len(files_all) // 1024, desc=f'Extracting features {model}'):
            files = chunk
            imgs = list(map(self.loader, files))
            imgs = list(map(transform, imgs))
            feats = feat_extractor(torch.stack(imgs, 0).to(self.device))
            image_feats.append(feats.data.cpu())
            image_files += files
        image_feats = torch.cat(image_feats, dim=0)
        print('features for %d images generated' % (len(image_files)))

        torch.save({'features': image_feats, 'files': image_files}, out_file)

    def __getitem__(self, index):
        '''
        Call for getting samples
        '''
        index = self.sample_indices[index]

        image, attr, obj = self.data[index]

        # Decide what to output
        if not self.update_image_features:
            if self.config.dataset == 'mit-states':
                pair, img = image.split('/')
                pair = pair.replace('_', ' ')
                image = pair + '/' + img
            img = self.activations[image]
        else:
            img = self.loader(image)
            img = self.transform(img)

        data = [img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]]

        return data

    def __len__(self):
        '''
        Call for length
        '''
        return len(self.sample_indices)

class SCENDataset(CANetDataset):
    def reset_dropout(self):
        '''
        Helper function to sample new subset of data containing a subset of pairs of objs and attrs
        '''
        self.pair_dropout = 0.0
        self.sample_indices = list(range(len(self.data)))
        self.sample_pairs = self.train_pairs

        # Using sampling from random instead of 2 step numpy
        n_pairs = int((1 - self.pair_dropout) * len(self.train_pairs))

        self.sample_pairs = random.sample(self.train_pairs, n_pairs)
        print('Sampled new subset')
        print('Using {} pairs out of {} pairs right now'.format(
            n_pairs, len(self.train_pairs)))

        self.sample_indices = [ i for i in range(len(self.data))
            if (self.data[i][1], self.data[i][2]) in self.sample_pairs
        ]
        print('Using {} images out of {} images right now'.format(
            len(self.sample_indices), len(self.data)))

    def sample_negative(self, attr, obj):
        '''
        Inputs
            attr: String of valid attribute
            obj: String of valid object
        Returns
            Tuple of a different attribute, object indexes
        '''
        new_attr, new_obj = self.sample_pairs[np.random.choice(
            len(self.sample_pairs))]

        while new_attr == attr and new_obj == obj:
            new_attr, new_obj = self.sample_pairs[np.random.choice(
            len(self.sample_pairs))]

        return (self.attr2idx[new_attr], self.obj2idx[new_obj])

    def sample_affordance(self, attr, obj):
        '''
        Inputs
            attr: String of valid attribute
            obj: String of valid object
        Return
            Idx of a different attribute for the same object
        '''
        new_attr = np.random.choice(self.obj_affordance[obj])

        while new_attr == attr:
            new_attr = np.random.choice(self.obj_affordance[obj])

        return self.attr2idx[new_attr]

    def sample_train_affordance(self, attr, obj):
        '''
        Inputs
            attr: String of valid attribute
            obj: String of valid object
        Return
            Idx of a different attribute for the same object from the training pairs
        '''
        new_attr = np.random.choice(self.train_obj_affordance[obj])

        while new_attr == attr:
            new_attr = np.random.choice(self.train_obj_affordance[obj])

        return self.attr2idx[new_attr]

    def __getitem__(self, index):
        '''
        Call for getting samples
        '''
        index = self.sample_indices[index]

        image, attr, obj = self.data[index]

        # Decide what to output
        if not self.update_image_features:
            img = self.activations[image]
        else:
            img = self.loader(image)
            img = self.transform(img)


        data = [img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]]
        # data = [img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]]

        if self.phase == 'train':

            img_pos_obj = [_img for (_img, _, _obj) in self.train_data if _obj == obj]
            img_pos_att = [_img for (_img, _att, _) in self.train_data if _att == attr]

            for i in range(len(img_pos_obj)):
                img_pos_obj[i] = self.activations[img_pos_obj[i]]
            if len(img_pos_obj) > 10:
                img_pos_obj_feats = random.sample(img_pos_obj, 10)
            else:
                if len(img_pos_obj) != 0:
                    img_pos_obj_feats = []
                    while len(img_pos_obj_feats) < 10:
                        for i in range(len(img_pos_obj)):
                            img_pos_obj_feats.append(img_pos_obj[i])
                            if len(img_pos_obj_feats) == 10:
                                break
                    # img_pos_obj_feats = img_pos_obj.repeat(math.ceil(10 / len(img_pos_obj)), 1)[:10]
                else:
                    img_pos_obj_feats = torch.Tensor(10, len(img_pos_obj[0]))


            for i in range(len(img_pos_att)):
                img_pos_att[i] = self.activations[img_pos_att[i]]
            if len(img_pos_att) > 10:
                img_pos_att_feats = random.sample(img_pos_att, 10)
            else:
                if len(img_pos_att) != 0:
                    img_pos_att_feats = []
                    while len(img_pos_att_feats) < 10:
                        for i in range(len(img_pos_att)):
                            img_pos_att_feats.append(img_pos_att[i])
                            if len(img_pos_att_feats) == 10:
                                break
                    # img_pos_obj_feats = img_pos_obj.repeat(math.ceil(10 / len(img_pos_obj)), 1)[:10]
                else:
                    img_pos_att_feats = torch.Tensor(10, len(img_pos_att[0]))

            img_pos_obj_feats = torch.tensor([item.cpu().detach().numpy() for item in img_pos_obj_feats])
            img_pos_att_feats = torch.tensor([item.cpu().detach().numpy() for item in img_pos_att_feats])

            all_neg_attrs = []
            all_neg_objs = []

            for curr in range(1):
                neg_attr, neg_obj = self.sample_negative(attr, obj) # negative for triplet lose
                all_neg_attrs.append(neg_attr)
                all_neg_objs.append(neg_obj)

            neg_attr, neg_obj = torch.LongTensor(all_neg_attrs), torch.LongTensor(all_neg_objs)

            #note here
            if len(self.train_obj_affordance[obj])>1:
                  inv_attr = self.sample_train_affordance(attr, obj) # attribute for inverse regularizer
            else:
                  inv_attr = (all_neg_attrs[0])

            comm_attr = self.sample_affordance(inv_attr, obj) # attribute for commutative regularizer

            data += [neg_attr, neg_obj, inv_attr, comm_attr, img_pos_obj_feats, img_pos_att_feats]

        return data

class IVRDataset(CANetDataset):
    def __init__(
        self,
        config,
        root,
        phase,
        split = 'compositional-split',
        model = 'resnet18',
        update_image_features = False,
    ):
        self.root = root
        self.phase = phase
        self.split = split
        self.update_image_features = update_image_features
        self.feat_dim = 512 if 'resnet18' in model else 2048
        self.open_world = False

        self.attrs, self.objs, self.pairs, self.train_pairs, self.val_pairs, self.test_pairs = self.parse_split()
        self.train_data, self.val_data, self.test_data = self.get_split_info()
 
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr : idx for idx, attr in enumerate(self.attrs)}

        if self.phase == 'train':
            self.pair2idx = {pair : idx for idx, pair in enumerate(self.train_pairs)}
        else:
            self.pair2idx = {pair : idx for idx, pair in enumerate(self.pairs)}
        self.all_pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}
        
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        elif self.phase == 'test':
            self.data = self.test_data
        
        self.all_data = self.train_data + self.val_data + self.test_data

        print('Dataset loaded')
        print('Train pairs: {}, Validation pairs: {}, Test Pairs: {}'.format(
            len(self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('Train images: {}, Validation images: {}, Test images: {}'.format(
            len(self.train_data), len(self.val_data), len(self.test_data)))

        self.sample_indices = list(range(len(self.data)))


        self.transform = transform_image(self.phase)
        self.loader = ImageLoader(os.path.join(self.root, 'images'))
        if not self.update_image_features:
            feat_file = os.path.join(root, model + '_feature_vectors.t7')
            if not os.path.exists(feat_file):
                print('  Feature file not found. Now get one!')
                with torch.no_grad():
                    self.generate_features(feat_file, model)
            self.phase = phase
            print(f'  Using {model} and feature file {feat_file}')
            activation_data = torch.load(feat_file)
            self.activations = dict(
                zip(activation_data['files'], activation_data['features']))
            self.feat_dim = activation_data['features'].size(1)

    def __getitem__(self, index):
        index = self.sample_indices[index]
        image, attr, obj = self.data[index]
        if self.phase == 'train':
            positive_attr = self.same_A_diff_B(label_A=attr, label_B=obj, phase='attr')
            same_attr_image = positive_attr[0]
            one_obj=positive_attr[2]
            one_attr = positive_attr[1]
            positive_obj = self.same_A_diff_B(label_A=obj, label_B=attr, phase='obj')
            same_obj_image = positive_obj[0]
            two_attr=positive_obj[1]
            two_obj= positive_obj[2]

        if not self.update_image_features:
            img = self.activations[image]
            if self.phase == 'train':
                same_attr_img = self.activations[same_attr_image]
                same_obj_img = self.activations[same_obj_image]
        else:
            img = self.loader(image)
            img = self.transform(img)
            if self.phase == 'train':
                same_attr_img = self.loader(same_attr_image)
                same_attr_img = self.transform(same_attr_img)
                same_obj_img = self.loader(same_obj_image)
                same_obj_img = self.transform(same_obj_img)

        data = [img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]]

        if self.phase == 'train':
            data += [same_attr_img, self.obj2idx[one_obj], same_obj_img, self.attr2idx[two_attr],
                self.attr2idx[one_attr], self.obj2idx[two_obj],
                self.pair2idx[(attr, one_obj)], self.pair2idx[(two_attr, obj)]]

        return data


    def same_A_diff_B(self, label_A, label_B, phase='attr'):
        data1 = []
        for i in range(len(self.train_data)):
            if phase=='attr':
                if (self.train_data[i][1]== label_A) & (self.train_data[i][2] != label_B):
                    data1.append(self.train_data[i])
            else:
                if (self.train_data[i][2]== label_A) & (self.train_data[i][1] != label_B):
                    data1.append(self.train_data[i])
            
        data2 = choice(data1)
        return data2

    def __len__(self):
        return len(self.sample_indices)

class PROLTDataset(CompositionDataset):
    '''
    Inputs
        root: String of base dir of dataset
        phase: String train, val, test
        split: String dataset split
        subset: Boolean if true uses a subset of train at each epoch
        num_negs: Int, numbers of negative pairs per batch
        pair_dropout: Percentage of pairs to leave in current epoch
    '''
    def __init__(
        self,
        config,
        phase,
        split = 'compositional-split',
        model = 'resnet18',
        norm_family = 'imagenet',
        subset = False,
        num_negs = 1,
        pair_dropout = 0.0,
        update_image_features = False,
        return_images = False,
        train_only = False,
        open_world=False
    ):
        super().__init__(
            root=config.dataset_path,
            phase=phase,
            split=split,
            open_world=config.open_world,
            imagenet=config.imagenet
        )
        self.num_negs = num_negs
        self.pair_dropout = pair_dropout
        self.norm_family = norm_family
        self.return_images = return_images
        self.update_image_features = update_image_features
        self.feat_dim = 512 if 'resnet18' in model else 2048
        self.open_world = open_world

        self.attrs, self.objs, self.pairs, self.train_pairs, \
            self.val_pairs, self.test_pairs = self.parse_split()
        self.train_data, self.val_data, self.test_data = self.get_split_info()
        self.full_pairs = list(product(self.attrs,self.objs))
        
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr : idx for idx, attr in enumerate(self.attrs)}
        if self.open_world:
            self.pairs = self.full_pairs

        self.all_pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        if train_only and self.phase == 'train':
            print('Using only train pairs')
            self.pair2idx = {pair : idx for idx, pair in enumerate(self.train_pairs)}
        else:
            print('Using all pairs')
            self.pair2idx = {pair : idx for idx, pair in enumerate(self.pairs)}
        
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        elif self.phase == 'test':
            self.data = self.test_data
        elif self.phase == 'all':
            print('Using all data')
            self.data = self.train_data + self.val_data + self.test_data
        else:
            raise ValueError('Invalid training phase')
        
        self.all_data = self.train_data + self.val_data + self.test_data
        print('Dataset loaded')
        print('Train pairs: {}, Validation pairs: {}, Test Pairs: {}'.format(
            len(self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('Train images: {}, Validation images: {}, Test images: {}'.format(
            len(self.train_data), len(self.val_data), len(self.test_data)))

        # Keeping a list of all pairs that occur with each object
        self.obj_affordance = {}
        self.train_obj_affordance = {}
        for _obj in self.objs:
            candidates = [attr for (_, attr, obj) in self.train_data+self.test_data if obj==_obj]
            self.obj_affordance[_obj] = list(set(candidates))

            candidates = [attr for (_, attr, obj) in self.train_data if obj==_obj]
            self.train_obj_affordance[_obj] = list(set(candidates))

        self.sample_indices = list(range(len(self.data)))
        self.sample_pairs = self.train_pairs

        # Load based on what to output
        self.transform = transform_image(self.phase,imagenet=False)
        self.loader = ImageLoader(os.path.join(self.root, 'images'))
        if not self.update_image_features:
            feat_file = os.path.join(self.root, model+'_featurers_map.t7')
            print(feat_file)
            print(f'Using {model} and feature file {feat_file}')
            if not os.path.exists(feat_file):
                with torch.no_grad():
                    self.generate_features(feat_file,model,feat_avgpool=False)
            self.phase = phase
            activation_data = torch.load(feat_file)
            self.activations = dict(
                zip(activation_data['files'], activation_data['features']))
            self.feat_dim = activation_data['features'].size(1)
            print('{} activations loaded'.format(len(self.activations)))

    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                pairs = [line.split() for line in pairs]
                pairs = list(map(tuple, pairs))

            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            os.path.join(self.root, self.split, 'train_pairs.txt')
        )
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            os.path.join(self.root, self.split, 'val_pairs.txt')
        )
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            os.path.join(self.root, self.split, 'test_pairs.txt')
        )
        
        #now we compose all objs, attrs and pairs
        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
                list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def get_split_info(self):
        '''
        Helper method to read image, attrs, objs samples

        Returns
            train_data, val_data, test_data: List of tuple of image, attrs, obj
        '''
        data = torch.load(os.path.join(self.root, 'metadata_{}.t7'.format(self.split)))

        train_data, val_data, test_data = [], [], []

        for instance in data:
            image, attr, obj, settype = instance['image'], instance['attr'], \
                instance['obj'], instance['set']
            curr_data = [image, attr, obj]

            if attr == 'NA' or (attr, obj) not in self.pairs or settype == 'NA':
                # Skip incomplete pairs, unknown pairs and unknown set
                continue

            if settype == 'train':
                train_data.append(curr_data)
            elif settype == 'val':
                val_data.append(curr_data)
            else:
                test_data.append(curr_data)

        return train_data, val_data, test_data

    def get_dict_data(self, data, pairs):
        data_dict = {}
        for current in pairs:
            data_dict[current] = []

        for current in data:
            image, attr, obj = current
            data_dict[(attr, obj)].append(image)
        
        return data_dict


    def reset_dropout(self):
        ''' 
        Helper function to sample new subset of data containing a subset of pairs of objs and attrs
        '''
        self.sample_indices = list(range(len(self.data)))
        self.sample_pairs = self.train_pairs

        # Using sampling from random instead of 2 step numpy
        n_pairs = int((1 - self.pair_dropout) * len(self.train_pairs))

        self.sample_pairs = random.sample(self.train_pairs, n_pairs)
        print('Sampled new subset')
        print('Using {} pairs out of {} pairs right now'.format(
            n_pairs, len(self.train_pairs)))

        self.sample_indices = [ i for i in range(len(self.data))
            if (self.data[i][1], self.data[i][2]) in self.sample_pairs
        ]
        print('Using {} images out of {} images right now'.format(
            len(self.sample_indices), len(self.data)))

    def sample_negative(self, attr, obj):
        '''
        Inputs
            attr: String of valid attribute
            obj: String of valid object
        Returns
            Tuple of a different attribute, object indexes
        '''
        new_attr, new_obj = self.sample_pairs[np.random.choice(
            len(self.sample_pairs))]

        while new_attr == attr and new_obj == obj:
            new_attr, new_obj = self.sample_pairs[np.random.choice(
            len(self.sample_pairs))]
        
        return (self.attr2idx[new_attr], self.obj2idx[new_obj])

    def sample_affordance(self, attr, obj):
        '''
        Inputs
            attr: String of valid attribute
            obj: String of valid object
        Return
            Idx of a different attribute for the same object
        '''
        new_attr = np.random.choice(self.obj_affordance[obj])
        
        while new_attr == attr:
            new_attr = np.random.choice(self.obj_affordance[obj])
        
        return self.attr2idx[new_attr]

    def sample_train_affordance(self, attr, obj):
        '''
        Inputs
            attr: String of valid attribute
            obj: String of valid object
        Return
            Idx of a different attribute for the same object from the training pairs
        '''
        new_attr = np.random.choice(self.train_obj_affordance[obj])
        
        while new_attr == attr:
            new_attr = np.random.choice(self.train_obj_affordance[obj])
        
        return self.attr2idx[new_attr]

    def generate_features(self, out_file, model, feat_avgpool=True):
        data = self.train_data + self.val_data + self.test_data
        transform = imagenet_transform('test')

        if model == 'resnet18':
            model = models.resnet18(pretrained=True)
        elif model == 'resnet50':
            model = models.resnet50(pretrained=True)

        feat_extractor = model
        feat_extractor.fc = nn.Sequential()
        feat_extractor.eval().cuda()

        image_feats = []
        image_files = []
        for chunk in tqdm.tqdm(
                chunks(data, 512), total=len(data) // 512):
            files, attrs, objs = zip(*chunk)
            imgs = list(map(self.loader, files))
            imgs = list(map(transform, imgs))
            imgs = torch.stack(imgs, 0).cuda()
            if feat_avgpool:
                feats = feat_extractor(imgs)
            else:
                feats = feat_extractor.conv1(imgs)
                feats = feat_extractor.bn1(feats)
                feats = feat_extractor.relu(feats)
                feats = feat_extractor.maxpool(feats)
                feats = feat_extractor.layer1(feats)
                feats = feat_extractor.layer2(feats)
                feats = feat_extractor.layer3(feats)
                feats = feat_extractor.layer4(feats)
                assert feats.shape[-3:] == (512, 7, 7), feats.shape
            image_feats.append(feats.data.cpu())
            image_files += files
        image_feats = torch.cat(image_feats, 0)
        print('features for %d images generated' % (len(image_files)))
        torch.save({'features': image_feats, 'files': image_files}, out_file)

    def __getitem__(self, index):
        '''
        Call for getting samples
        '''
        index = self.sample_indices[index]

        image, attr, obj = self.data[index]

        # Decide what to output
        if not self.update_image_features:
            img = self.activations[image]
        else:
            img = self.loader(image)
            img = self.transform(img)

            #fix_img = self.activations[image]

        data = [img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]]
        
        if self.phase == 'train':
            all_neg_attrs = []
            all_neg_objs = []

            for curr in range(self.num_negs):
                neg_attr, neg_obj = self.sample_negative(attr, obj) # negative for triplet lose
                all_neg_attrs.append(neg_attr)
                all_neg_objs.append(neg_obj)

            neg_attr, neg_obj = torch.LongTensor(all_neg_attrs), torch.LongTensor(all_neg_objs)
            
            #note here
            if len(self.train_obj_affordance[obj])>1:
                  inv_attr = self.sample_train_affordance(attr, obj) # attribute for inverse regularizer
            else:
                  inv_attr = (all_neg_attrs[0]) 

            comm_attr = self.sample_affordance(inv_attr, obj) # attribute for commutative regularizer
            

            data += [neg_attr, neg_obj, inv_attr, comm_attr]

        # Return image paths if requested as the last element of the list
        if self.return_images and self.phase != 'train':
            data.append(image)

        return data
    
    def __len__(self):
        '''
        Call for length
        '''
        return len(self.sample_indices)