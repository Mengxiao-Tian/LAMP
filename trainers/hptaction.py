import os.path as osp

import json
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from collections import OrderedDict

from clip import clip
import numpy as np
import ast
from transformers import CLIPVisionModelWithProjection, AutoProcessor, CLIPVisionModel, CLIPTextModel, AutoTokenizer, CLIPModel

def load_clip_to_cpu(cfg):
    
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    
    model = clip.build_model(state_dict or model.state_dict())

    return model

class VisionEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        visual = clip_model.visual
        self.ln_pre = visual.ln_pre
        self.transformer = visual.transformer.resblocks
        self.ln_post = visual.ln_post
        self.proj = visual.proj
        self.dtype = clip_model.dtype
        self.n_vpro = cfg.TRAINER.HPT.N_VPRO

    def forward(self, x):
        x = self.ln_pre(x).type(self.dtype)
        x = x.permute(1, 0, 2)

        x = self.transformer(x)
        
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        x = x @ self.proj
        
        return x


class Adapter(nn.Module):
    def __init__(self, c_in, c_out, reduction):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_out, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class PositionWiseFeedForward(nn.Module):
    def __init__(self, embed_dim, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, d_ff)
        self.fc2 = nn.Linear(d_ff, embed_dim)
        self.relu = QuickGELU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class AdapterLayer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, d_ff=2048, dropout=0.1):
        super(AdapterLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.feed_forward = PositionWiseFeedForward(embed_dim, d_ff)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, y=None):
        x = x.reshape(-1, x.shape[0], x.shape[1]).float()
        y = y.reshape(-1, y.shape[0], y.shape[1]).float()

        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.cross_attn(self.norm2(x), y, y)[0]
        x = x + self.feed_forward(self.norm3(x))
        x = torch.squeeze(x, dim=0)
        return x


class VisionPromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.pro_dim = clip_model.visual.ln_pre.weight.shape[0]
        self.dtype = clip_model.dtype
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.layers = len(clip_model.visual.transformer.resblocks)
        self.proj = Adapter(self.ctx_dim, self.pro_dim, 1).to(clip_model.dtype)

    def forward(self, x, prefix):
        x = x.type(self.dtype)
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], 
                                                                      dtype=x.dtype, device=x.device), x], dim=1) 
        x = x + self.positional_embedding.to(x.dtype)
        prefix = prefix.half()
        p_input = self.proj(prefix)
        p_input = p_input.reshape(x.shape[0],-1,x.shape[2])
        x = torch.cat([x, p_input], dim=1)
        return x


class TextEncoderZS(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer.resblocks
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class myTransformer(nn.Module):
    def __init__(self, vocab_size=50000, width=512, num_layer=4, num_head=8):
        super().__init__()
        self.context_length = width
        self.vocab_size = vocab_size
        self.width = width
        self.transformer = Transformer(width=width,
                                       layers=num_layer,
                                       heads=num_head,
                                       attn_mask=None)
        self.subject_embedding = torch.nn.Embedding(self.vocab_size, self.width)
        self.rel_embedding = torch.nn.Embedding(self.vocab_size, self.width)
        self.object_embedding = torch.nn.Embedding(self.vocab_size, self.width)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.subject_embedding.weight, std=0.001)
        nn.init.normal_(self.rel_embedding.weight, std=0.001)
        nn.init.normal_(self.object_embedding.weight, std=0.001)
    
    def forward(self, actions, device):
        know = []
        for action in actions:
            triplet_list = ast.literal_eval(action)
            heads = []
            relations = []
            tails = []

            for item in triplet_list:
                heads.append(item[1])
                relations.append(item[2])
                tails.append(item[3])

            heads = torch.Tensor(heads).long().to(device)
            relations = torch.Tensor(relations).long().to(device)
            tails = torch.Tensor(tails).long().to(device)

            head_emb = self.subject_embedding(heads)
            rel_emb = self.rel_embedding(relations)
            tail_emb = self.object_embedding(tails)
            knowledge_emb = head_emb + rel_emb + tail_emb
            knowledge_emb = knowledge_emb.reshape(-1, knowledge_emb.shape[0], knowledge_emb.shape[1])
            knowledge_emb = self.transformer(knowledge_emb)
            knowledge_emb = knowledge_emb.mean(dim=1)
            knowledge_emb = knowledge_emb / knowledge_emb.norm()
            know.append(knowledge_emb)

        know = torch.cat(know, dim=0)

        return know


class TextEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer.resblocks
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.n_tpro = cfg.TRAINER.HPT.N_TPRO
        self.n_set = cfg.TRAINER.HPT.N_SET 

    def forward(self, x, p_ins, p_uni, tokenized_prompts, attn, flag):
        (l, c, d) = p_ins.shape
        p_ins = p_ins.reshape(l, c//self.n_set, self.n_set, d)

        if not flag:
            p_ins = p_ins.unsqueeze(2).expand(-1, -1, self.n_set, -1, -1)
            p_ins = torch.flatten(p_ins, 1, 2) 
            
        p_ins = p_ins.permute(0, 2, 1, 3).type(self.dtype)
        x = (x + self.positional_embedding).type(self.dtype)
        x = x.permute(1, 0, 2)

        for layer_idx, layer in enumerate(self.transformer):
            if layer_idx > 0:                
                prefix = x[:1]
                suffix = x[1+self.n_tpro+self.n_set:]
                ctx_g = p_uni[layer_idx - 1].unsqueeze(1).expand(self.n_tpro, prefix.shape[1], -1)
                ctx_h = p_ins[layer_idx - 1]
                x = torch.cat([prefix, ctx_g, ctx_h, suffix], dim=0)
                x = layer(x, attn[:, layer_idx])
            elif layer_idx == 0:
                x = layer(x, attn[:, layer_idx])
            else:
                x = layer(x)

        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        
        if not flag:
            x = x.reshape(x.shape[0]//5, 5, -1)

        return x

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VisionPromptLearner(cfg, clip_model)
        self.image_encoder = VisionEncoder(cfg, clip_model)
        self.text_encoder_zs = TextEncoderZS(cfg, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.clip_model = clip_model
        self.triplet_trans = myTransformer()
        self.gpt_joint = AdapterLayer()

    def get_features(self, text, device):
        texts = clip.tokenize(text).to(device)
        text_features = self.text_encoder_zs(texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def get_gpt_features(self, texts, device):
        features = []
        for text in texts:
            triplet_list = ast.literal_eval(text)
            gpt_features = self.get_features(triplet_list, device)
            gpt_features = torch.unsqueeze(gpt_features, 0)
            gpt_features = gpt_features.mean(dim=1)
            gpt_features = gpt_features / gpt_features.norm()
            features.append(gpt_features)

        features = torch.cat(features, dim=0)
        return features


    def ceshi(self, image, caption, actions, caption_gpt, device):
        text_features = self.get_features(caption, device)
        act_features = self.triplet_trans(actions, device)
        gpt_features = self.get_gpt_features(caption_gpt, device)

        act_features_huizong = self.gpt_joint(act_features, gpt_features)

        x = self.prompt_learner(image, act_features_huizong)
        image_features = self.image_encoder(x)
        return image_features, text_features

    def forward(self, image, caption, actions, caption_gpt, device):
        logit_scale = self.logit_scale.exp()

        text_features = self.get_features(caption, device)
        act_features = self.triplet_trans(actions, device)
        gpt_features = self.get_gpt_features(caption_gpt, device)

        act_features_huizong = self.gpt_joint(act_features, gpt_features)

        x = self.prompt_learner(image, act_features_huizong)
        image_features = self.image_encoder(x)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


@TRAINER_REGISTRY.register()
class HPT(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.HPT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.n_class = len(self.dm.dataset.classnames)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg).cuda()

        if cfg.TRAINER.HPT.PREC == "fp32" or cfg.TRAINER.HPT.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")

        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
            if "prompt_learner" in name:
                param.requires_grad_(True)
            if "triplet_trans" in name:
                param.requires_grad_(True)
            if "gpt_joint" in name:
                param.requires_grad_(True)
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("Model", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.HPT.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        print(device_count)
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model).cuda()

    def forward_backward(self, batch):

        image, caption, actions, caption_gpt, device = self.parse_batch_train(batch)
        logits_per_image, logits_per_text = self.model(image, caption, actions, caption_gpt, device)

        ground_truth = torch.arange(len(image),dtype=torch.long).cuda()

        loss_i = F.cross_entropy(logits_per_image, ground_truth)
        loss_t = F.cross_entropy(logits_per_text, ground_truth)

        loss = (loss_i + loss_t) / 2

        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        actions = batch['domain']
        input = input.to(self.device)
        caption = batch['caption']
        actions = batch['actions']
        caption_gpt = batch['gpts']
        device = self.device
        return input, caption, actions, caption_gpt, device

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
