from itertools import product
from turtle import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import argparse
import clip
from collections import OrderedDict
from clip_modules.model_loader import load
from models.common import *
import numpy as np

class FusionTextImageBlock(nn.Module):
    def __init__(self, width_img: int, width_txt: int, attributes: int, classes: int, layers: int, attn_mask: torch.Tensor = None, context_length: int = 8, fusion: str = "BiFusion"):
        super().__init__()
        self.fusion = fusion
        self.width_img = width_img
        self.width_txt = width_txt
        self.layers = layers
        self.context_length = context_length
        self.attributes = attributes
        self.classes = classes
        self.img2txt_transform_layer1 = nn.Linear(width_img, width_txt)
        self.img2txt_transform_layer2 = nn.Linear(257, context_length * (attributes + classes))
        self.txt2img_transform_layer1 = nn.Linear(width_txt, width_img)
        self.txt2img_transform_layer2 = nn.Linear(context_length * (attributes + classes), 257)
        self.dropout = nn.Dropout(0.3)
        self.crossblock_img = CrossResidualAttentionBlock(width_img, width_img//64, attn_mask)
        self.crossblock_txt = CrossResidualAttentionBlock(width_txt, width_txt//64, attn_mask)
        self.resblocks_img = nn.Sequential(*[ResidualAttentionBlock(width_img, width_img//64, attn_mask) for _ in range(layers)])
        self.resblocks_txt = nn.Sequential(*[ResidualAttentionBlock(width_txt, width_txt//64, attn_mask) for _ in range(layers)])
        self.txt_fine_tune = nn.Linear(self.width_txt, self.width_txt)


    def decompose(self, text_feature, idx):
        t, l, c = text_feature.shape
        att_idx, obj_idx = idx[:, 0].cpu().numpy(), idx[:, 1].cpu().numpy()
        text_att = torch.zeros(t, self.attributes, c).cuda()
        text_obj = torch.zeros(t, self.classes, c).cuda()
        for i in range(self.attributes):
            text_att[:, i, :] = text_feature[:, np.where(att_idx==i)[0], :].mean(-2)
        for i in range(self.classes):
            text_obj[:, i, :] = text_feature[:, np.where(obj_idx==i)[0], :].mean(-2)    
        text_decom_feature = torch.cat([text_att, text_obj], dim=1)
        return text_decom_feature


    def compose(self, text_feature, idx):
        t, l, c = text_feature.shape
        att_idx, obj_idx = idx[:, 0].cpu().numpy(), idx[:, 1].cpu().numpy()
        text_com_feature = torch.zeros(t, len(idx), c).cuda()
        text_com_feature = text_feature[:, att_idx, :] * text_feature[:, obj_idx + self.attributes, :]
        text_com_feature = self.txt_fine_tune(text_com_feature)
        return text_com_feature



    def img2txt(self, x: torch.Tensor):
        x = self.img2txt_transform_layer1(x)
        x = x.permute(2,1,0)
        x = self.img2txt_transform_layer2(x)
        x = x.permute(2,1,0).reshape(-1, (self.attributes + self.classes), self.width_txt)
        x = self.dropout(x)
        return x

    def txt2img(self, x:torch.Tensor, idx, b: int):
        x = self.decompose(x, idx)
        x = self.txt2img_transform_layer1(x)
        x = rearrange(x, 't l c -> c (t l)')
        x = self.txt2img_transform_layer2(x)
        x = self.dropout(x)
        x = x.permute(1,0).unsqueeze(1).repeat(1,b,1)
        return x
        

    def forward(self, x_image: torch.Tensor, x_text: torch.Tensor, idx, b: int):
        if self.fusion == "BiFusion":
            x_img = self.crossblock_img(x_image, self.txt2img(x_text, idx, b))
            x_txt = self.img2txt(x_image)
            x_text = self.decompose(x_text, idx)
            x_txt = self.crossblock_txt(x_text.repeat(b, 1, 1), x_txt)
            x_txt = self.resblocks_txt(x_txt)
            x_txt = self.compose(x_txt, idx)
            x_txt = x_txt.reshape(b, self.context_length, -1, self.width_txt)
            x_img = self.resblocks_img(x_img)
            return x_img, x_txt
        elif self.fusion == "img2txt":
            x_txt = self.img2txt(x_image)
            x_text = self.decompose(x_text, idx)
            x_txt = self.crossblock_txt(x_text.repeat(b, 1, 1), x_txt)
            x_txt = self.resblocks_txt(x_txt)
            x_txt = self.compose(x_txt, idx)
            x_txt = x_txt.reshape(b, self.context_length, -1, self.width_txt)
            x_img = self.resblocks_img(x_image)
            return x_img, x_txt
        elif self.fusion == "txt2img":
            x_img = self.crossblock_img(x_image, self.txt2img(x_text, idx, b))
            x_img = self.resblocks_img(x_img)
            x_txt = self.resblocks_txt(x_text)
            return x_img, x_txt
        elif self.fusion == "OnlySPM":
            return x_image, x_text

class DFSP(nn.Module):
    def __init__(self, config, attributes, classes, offset):
        super().__init__()
        clip_model, _ = load(config.clip_model, context_length=config.context_length)
        self.clip = clip_model
        self.config = config
        self.attributes = attributes
        self.classes = classes
        self.attr_dropout = nn.Dropout(config.attr_dropout)
        self.token_ids, self.soft_att_obj, ctx_vectors = self.construct_soft_prompt()
        self.offset = offset
        self.enable_pos_emb = True
        dtype = None
        if dtype is None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.text_encoder = CustomTextEncoder(self.clip, self.dtype)
        for p in self.parameters():
            p.requires_grad=False

        self.soft_att_obj = nn.Parameter(self.soft_att_obj)
        self.soft_prompt = nn.Parameter(ctx_vectors).cuda()
        self.fusion = FusionTextImageBlock(config.width_img, config.width_txt, len(self.attributes), len(self.classes), config.SA_K, context_length=self.config.context_length, fusion=self.config.fusion)
        self.weight = config.res_w


    def construct_soft_prompt(self):
        token_ids = clip.tokenize("a photo of x x",
                              context_length=self.config.context_length).cuda()

        tokenized = torch.cat(
            [
                clip.tokenize(tok, context_length=self.config.context_length)
                for tok in self.attributes + self.classes
            ]
        )
        orig_token_embedding = self.clip.token_embedding(tokenized.cuda())

        # with torch.no_grad():
        soft_att_obj = torch.zeros(
            (len(self.attributes) + len(self.classes), orig_token_embedding.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        ctx_init = "a photo of "
        n_ctx = len(ctx_init.split())
        prompt = clip.tokenize(ctx_init,
                            context_length=self.config.context_length).cuda()
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        return token_ids, soft_att_obj, ctx_vectors


    def construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)
        token_tensor = self.clip.token_embedding(
            class_token_ids.cuda()
        ).type(self.clip.dtype)
        soft_att_obj = self.attr_dropout(self.soft_att_obj)
        eos_idx = int(self.token_ids[0].argmax())
        token_tensor[:, eos_idx - 2, :] = soft_att_obj[
            attr_idx
        ].type(self.clip.dtype)
        token_tensor[:, eos_idx - 1, :] = soft_att_obj[
            obj_idx + self.offset
        ].type(self.clip.dtype)

        # adding the correct learnable context
        token_tensor[
            :, 1 : len(self.soft_prompt) + 1, :
        ] = self.soft_prompt.type(self.clip.dtype)
        return token_tensor



    def visual(self, x: torch.Tensor):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        img_feature = self.clip.visual.transformer(x)

        x = img_feature.permute(1, 0, 2)  # LND -> NLD

        x = self.clip.visual.ln_post(x[:, 0, :])
        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj
        return x, img_feature


    def ft_to_logit(self, img, txt):
        img_feature = img.permute(1, 0, 2)  # LND -> NLD

        img_feature = self.clip.visual.ln_post(img_feature[:, 0, :])
        if self.clip.visual.proj is not None:
            img_feature = img_feature @ self.clip.visual.proj
        
        if self.config.fusion in ["BiFusion", "img2txt"]: 
            txt_feature = txt.permute(0, 2, 1, 3)
            txt_feature = self.text_encoder.ln_final(txt_feature)
            txt_tf = (
                txt_feature[
                    :, torch.arange(txt_feature.shape[1]), self.token_ids.argmax(dim=-1)
                ]  # POS of <EOS>
                @ self.text_encoder.text_projection
            )
        else:
            txt_feature = txt.permute(1, 0, 2)
            txt_feature = self.text_encoder.ln_final(txt_feature)
            txt_tf = (
                txt_feature[
                    torch.arange(txt_feature.shape[0]), self.token_ids.argmax(dim=-1)
                ]  # POS of <EOS>
                @ self.text_encoder.text_projection
            )
        return img_feature, txt_tf

    def decompose_logits(self, logits, idx):
        att_idx, obj_idx = idx[:, 0].cpu().numpy(), idx[:, 1].cpu().numpy()
        logits_att = torch.zeros(logits.shape[0], len(self.attributes)).cuda()
        logits_obj = torch.zeros(logits.shape[0], len(self.classes)).cuda()
        for i in range(len(self.attributes)):
            logits_att[:, i] = logits[:, np.where(att_idx==i)[0]].mean(-1)
        for i in range(len(self.classes)):
            logits_obj[:, i] = logits[:, np.where(obj_idx==i)[0]].mean(-1)        
        return logits_att, logits_obj


    def forward(self, batch_img, idx):
        b = batch_img.shape[0]
        l, _ = idx.shape
        batch_img, img_ft = self.visual(batch_img.type(self.clip.dtype))   ## bs * 768
        token_tensors = self.construct_token_tensors(idx)
        text_features, text_ft = self.text_encoder(
            self.token_ids,
            token_tensors,
            enable_pos_emb=self.enable_pos_emb,
        )  
        batch_img_soft_prompt = batch_img / batch_img.norm(dim=-1, keepdim=True)
        text_features_soft_prompt = text_features / text_features.norm(dim=-1, keepdim=True)
        img_ft, text_ft = self.fusion(img_ft.type(torch.float), text_ft.type(torch.float), idx, b)
        img_ft, text_ft = self.ft_to_logit(img_ft.type(self.clip.dtype), text_ft.type(self.clip.dtype))
        batch_img = self.weight * batch_img + (1 - self.weight) * img_ft
        normalized_img = batch_img / batch_img.norm(dim=-1, keepdim=True)
        if self.config.fusion in ["BiFusion", "img2txt"]: 
            text_features = self.weight * text_features.repeat(b, 1, 1) + (1 - self.weight) * text_ft
        else:
            text_features = self.weight * text_features + (1 - self.weight) * text_ft
        idx_text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )
        if self.config.fusion in ["BiFusion", "img2txt"]: 
            logits = (
                self.clip.logit_scale.exp()
                * normalized_img.unsqueeze(1)
                @ idx_text_features.permute(0,2,1)
            ).squeeze()     ###     48 * 1262
        else:
            logits = (
                self.clip.logit_scale.exp()
                * normalized_img
                @ idx_text_features.t()
            )   

        logits_soft_prompt = (
            self.clip.logit_scale.exp()
            * batch_img_soft_prompt
            @ text_features_soft_prompt.t()
        )     

        logits_att, logits_obj = self.decompose_logits(logits_soft_prompt, idx)

        return (logits, logits_att, logits_obj, logits_soft_prompt)
