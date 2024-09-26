# Adapted from https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/gpt.py
import os
import itertools
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl

from src.videogpt.resnet import resnet34, ResidualBlock
from src.videogpt.attention import AttentionStack, LayerNorm, AddBroadcastPosEmbed


class VideoGPT(pl.LightningModule):
    def __init__(self, args, vqvae_ckpt=None, desc=None):
        super().__init__()
        self.args = args
        self.desc = desc

        # Load VQ-VAE and set all parameters to no grad
        from src.videogpt.vqvae import VQVAE

        self.vqvae = (
            VQVAE.load_from_checkpoint(args.vqvae)
            if vqvae_ckpt is None
            else VQVAE.load_from_checkpoint(vqvae_ckpt)
        )
        for p in self.vqvae.parameters():
            p.requires_grad = False
        self.vqvae.codebook._need_init = False
        self.vqvae.eval()

        # ResNet34 for frame conditioning
        self.use_frame_cond = args.n_cond_frames > 0
        if self.use_frame_cond:
            cond_shape = (args.n_cond_frames, args.resolution // 4, args.resolution // 4, 240)
            self.resnet = resnet34(1, (1, 4, 4), resnet_dim=240)
            self.cond_pos_embd = AddBroadcastPosEmbed(
                shape=cond_shape[:-1], embd_dim=cond_shape[-1]
            )
        else:
            frame_cond_shape = None

        # ResNet for embedding conditioning
        self.use_emb_cond = hasattr(args, "emb_cond") and args.emb_cond  # backward compatibility
        if self.use_emb_cond:
            cond_shape = (*self.vqvae.latent_shape, 240)
            self.resnet = ResidualBlock(
                self.vqvae.embedding_dim, 240, stride=1, use_projection=True
            )
            self.cond_pos_embd = AddBroadcastPosEmbed(
                shape=cond_shape[:-1], embd_dim=cond_shape[-1]
            )

        # VideoGPT transformer
        self.shape = self.vqvae.latent_shape  # dowsampled (t, h, w)

        self.fc_in = nn.Linear(self.vqvae.embedding_dim, args.hidden_dim, bias=False)
        self.fc_in.weight.data.normal_(std=0.02)

        self.attn_stack = AttentionStack(
            self.shape,
            args.hidden_dim,
            args.heads,
            args.layers,
            args.dropout,
            args.attn_type,
            args.attn_dropout,
            args.class_cond_dim,
            cond_shape,
        )

        self.norm = LayerNorm(args.hidden_dim, args.class_cond_dim)

        self.fc_out = nn.Linear(args.hidden_dim, self.vqvae.n_codes, bias=False)
        self.fc_out.weight.data.copy_(torch.zeros(self.vqvae.n_codes, args.hidden_dim))

        # caches for faster decoding (if necessary)
        self.cond_cache = None

        self.save_hyperparameters()

    def get_reconstruction(self, videos):
        return self.vqvae.decode(self.vqvae.encode(videos))

    def sample(self, n, batch=None):
        device = self.fc_in.weight.device

        cond = dict()
        if self.use_frame_cond or self.args.class_cond or self.use_emb_cond:
            assert batch is not None
            video = batch

            if self.args.class_cond:
                label = batch[1]
                cond["class_cond"] = F.one_hot(label, self.args.class_cond_dim).type_as(video)
            if self.use_frame_cond:
                cond["frame_cond"] = video[:, :, : self.args.n_cond_frames]
            if self.use_emb_cond:
                cond["emb_cond"] = video

        samples = torch.zeros((n,) + self.shape).long().to(device)
        idxs = list(itertools.product(*[range(s) for s in self.shape]))

        with torch.no_grad():
            prev_idx = None
            for i, idx in enumerate(tqdm(idxs)):
                batch_idx_slice = (slice(None, None), *[slice(i, i + 1) for i in idx])
                batch_idx = (slice(None, None), *idx)
                embeddings = self.vqvae.codebook.dictionary_lookup(samples)

                if prev_idx is None:
                    # set arbitrary input values for the first token
                    # does not matter what value since it will be shifted anyways
                    embeddings_slice = embeddings[batch_idx_slice]
                    samples_slice = samples[batch_idx_slice]
                else:
                    embeddings_slice = embeddings[prev_idx]
                    samples_slice = samples[prev_idx]

                logits = self(embeddings_slice, samples_slice, cond, decode_step=i, decode_idx=idx)[
                    1
                ]
                # squeeze all possible dim except batch dimension
                logits = logits.squeeze().unsqueeze(0) if logits.shape[0] == 1 else logits.squeeze()
                probs = F.softmax(logits, dim=-1)
                samples[batch_idx] = torch.multinomial(probs, 1).squeeze(-1)

                prev_idx = batch_idx_slice
            samples = self.vqvae.decode(samples)
            samples = torch.clamp(samples, -0.5, 0.5)

        return samples  # BCTHW in [-0.5, 0.5]

    def forward(self, x, targets, cond, decode_step=None, decode_idx=None):
        if self.use_frame_cond:
            if decode_step is None:
                cond["frame_cond"] = self.cond_pos_embd(self.resnet(cond["frame_cond"]))
            elif decode_step == 0:
                self.cond_cache = self.cond_pos_embd(self.resnet(cond["frame_cond"]))
                cond["frame_cond"] = self.cond_cache
            else:
                cond["frame_cond"] = self.cond_cache

        elif self.use_emb_cond:
            if decode_step is None:
                cond["emb_cond"] = self.vqvae.encode(cond["emb_cond"], True)[1]
                cond["emb_cond"] = self.cond_pos_embd(self.resnet(cond["emb_cond"]).movedim(-4, -1))
            elif decode_step == 0:
                cond["emb_cond"] = self.vqvae.encode(cond["emb_cond"], True)[1]
                self.cond_cache = self.cond_pos_embd(self.resnet(cond["emb_cond"]).movedim(-4, -1))
                cond["emb_cond"] = self.cond_cache
            else:
                cond["emb_cond"] = self.cond_cache

        h = self.fc_in(x)
        h = self.attn_stack(h, cond, decode_step, decode_idx)
        h = self.norm(h, cond)
        logits = self.fc_out(h)

        loss = F.cross_entropy(logits.movedim(-1, 1), targets)

        return loss, logits

    '''    
    def training_step(self, batch, batch_idx):
        self.vqvae.eval()
        x = batch[0] if self.args.class_cond else batch

        cond = dict()
        if self.args.class_cond:
            label = batch[1]
            cond["class_cond"] = F.one_hot(label, self.args.class_cond_dim).type_as(x)
        if self.use_frame_cond:
            cond["frame_cond"] = x[:, :, : self.args.n_cond_frames]
        if self.use_emb_cond:
            cond["emb_cond"] = x[:, :, :self.vqvae.sequence_length]
            x = x[:, :, -self.vqvae.sequence_length:]

        with torch.no_grad():
            targets, x = self.vqvae.encode(x, include_embeddings=True)
            x = x.movedim(1, -1)

        loss, accuracy = self(x, targets, cond)

        return {"loss": loss, "accuracy": accuracy}

    def validation_step(self, batch, batch_idx):
        metrics = self.training_step(batch, batch_idx)
        loss = metrics["loss"]
        accuracy = metrics["accuracy"]

        self.log('val/loss', loss, prog_bar=True)
        self.log('val/accuracy', accuracy, prog_bar=True)
 
    '''
    
    def training_step(self, batch, batch_idx):
        self.vqvae.eval()
        x = batch[0] if self.args.class_cond else batch

        cond = dict()
        if self.args.class_cond:
            label = batch[1]
            cond["class_cond"] = F.one_hot(label, self.args.class_cond_dim).type_as(x)
        if self.use_frame_cond:
            cond["frame_cond"] = x[:, :, : self.args.n_cond_frames]
        if self.use_emb_cond:
            cond["emb_cond"] = x[:, :, : self.vqvae.sequence_length]
            x = x[:, :, -self.vqvae.sequence_length :]

        with torch.no_grad():
            targets, x = self.vqvae.encode(
                x, include_embeddings=True
            )  # target = encodings, x = embeddings
            x = x.movedim(1, -1)

        
        #loss, _ = self(x, targets, cond)
        loss, accuracy = self(x, targets, cond)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True)


        #loss, _ = self(x, targets, cond)


        #loss, accuracy = self(x, targets, cond)
        #self.log("tng/loss", loss)
        #self.log("tng/accuracy", accuracy)

        # Return a dictionary of metrics
        #return {"loss": loss, "accuracy": accuracy}
        #return loss

    
    

    #def validation_step(self, batch, batch_idx):
        #loss = self.training_step(batch, batch_idx)
        #metrics = self.training_step(batch, batch_idx)

        #loss = metrics["loss"]
        #accuracy = metrics["accuracy"]

        #self.log("val/loss", metrics["loss"], prog_bar=True)
        #self.log("val/accuracy", metrics["accuracy"], prog_bar=True)

        #self.log("val/loss", loss, prog_bar=True)
        #self.log('val/accuracy', accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, betas=(0.9, 0.999))
        assert (
            hasattr(self.args, "max_steps") and self.args.max_steps is not None
        ), f"Must set max_steps argument"
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_steps)
        return [optimizer], [dict(scheduler=scheduler, interval="step", frequency=1)]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--vqvae", type=str, help="path to vqvae ckpt")
        parser.add_argument("--n_cond_frames", type=int, default=0)
        parser.add_argument("--class_cond", action="store_true")
        parser.add_argument("--emb_cond", action="store_true")

        # VideoGPT hyperparmeters
        parser.add_argument("--hidden_dim", type=int, default=576)
        parser.add_argument("--heads", type=int, default=4)
        parser.add_argument("--layers", type=int, default=8)
        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument("--attn_type", type=str, default="full", choices=["full", "sparse"])
        parser.add_argument("--attn_dropout", type=float, default=0.3)

        return parser
