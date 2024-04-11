from itertools import product

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, RandomHorizontalFlip,
                                    RandomPerspective, RandomRotation, Resize,
                                    ToTensor)
from torchvision import transforms
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
            data = [
                img, self.attr2idx[attr], self.obj2idx[obj], self.train_pair_to_idx[(attr, obj)]
            ]
        else:
            data = [
                img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]
            ]

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

def imagenet_transform_zappos(phase, cfg):
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

        ## TODO: ut-zap50k is utzappos ??
        if 'ut-zap50k' in cfg.dataset:
            self.transform = imagenet_transform_zappos(phase, cfg)
        else:
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
