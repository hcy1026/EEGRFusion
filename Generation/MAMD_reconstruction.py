import os
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'
from itertools import combinations

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from eegdatasets_leaveone import EEGDataset

from einops.layers.torch import Rearrange, Reduce

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import random
from util import wandb_logger
import csv
from torch import Tensor
import itertools
import math
import re
from subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from subject_layers.Embed import DataEmbedding
from subject_layers.mamba2 import Mamba2Encoder
from subject_layers.MontageAwareEmbedding import MontageAwareEmbedding

import numpy as np
from loss import ClipLoss
import argparse
from torch import nn
from torch.optim import AdamW


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model + 1, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term[:d_model // 2 + 1])
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:x.size(0), :].unsqueeze(1).repeat(1, x.size(1), 1)
        x = x + pe
        return x


def build_coords_from_ch_names(ch_names, montage_name="standard_1020"):
    """
    ch_names: list[str]，来自你的预处理产物
    return: torch.FloatTensor [C,3]
    """
    try:
        import mne
    except Exception as e:
        raise RuntimeError("mne 未安装或不可用，但 montage-aware 需要它来生成标准电极坐标") from e

    montage = mne.channels.make_standard_montage(montage_name)
    ch_pos = montage.get_positions()["ch_pos"]  # dict: name -> (x,y,z)

    # 做一个大小写/前缀鲁棒的映射
    def norm(n):
        n = n.strip()
        n = n.replace("EEG ", "").replace("eeg ", "")
        return n

    ch_pos_lut = {k.lower(): v for k, v in ch_pos.items()}

    coords = []
    missing = []
    for name in ch_names:
        key = norm(name)
        v = ch_pos_lut.get(key.lower(), None)
        if v is None:
            # 尝试常见大小写形式（Fp1 vs FP1）
            v = ch_pos_lut.get(key.capitalize().lower(), None)
        if v is None:
            missing.append(name)
            v = (0.0, 0.0, 0.0)  # 找不到就置零，不让程序崩
        coords.append(v)

    if len(missing) > 0:
        print(f"[MontageAware] WARNING: {len(missing)} channels not found in {montage_name}, set to zeros. "
              f"Examples: {missing[:5]}")

    coords = torch.tensor(coords, dtype=torch.float32)  # [C,3]
    return coords


class Config:
    def __init__(self):
        self.task_name = 'classification'  # Example task name
        self.seq_len = 250  # Sequence length
        self.pred_len = 250  # Prediction length
        self.output_attention = False  # Whether to output attention weights
        self.d_model = 250  # Model dimension
        self.embed = 'timeF'  # Time encoding method
        self.freq = 'h'  # Time frequency
        self.dropout = 0.25  # Dropout rate
        self.factor = 1  # Attention scaling factor
        self.n_heads = 4  # Number of attention heads
        self.e_layers = 1  # Number of encoder layers
        self.d_ff = 256  # Feedforward network dimension
        self.activation = 'gelu'  # Activation function
        self.enc_in = 63  # Encoder input dimension (example value)


class iTransformer(nn.Module):
    def __init__(self, configs, joint_train=False, num_subjects=10, backbone="mamba2"):
        super(iTransformer, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        #self.enc_embedding = DataEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq,
        #                                   configs.dropout, joint_train=False, num_subjects=num_subjects)
        self.enc_embedding = MontageAwareEmbedding(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout,
            joint_train=False, num_subjects=num_subjects,
            coord_dim=3, coord_scale=1.0, use_coords=True
        )
        self.pos_encoder = PositionalEncoding(configs.enc_in)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=configs.enc_in, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        # Encoder

        if backbone == "mamba2":
            self.encoder = Mamba2Encoder(
                d_model=configs.d_model,      # 250
                n_layers=4,    # 与 baseline 对齐 configs.e_layers 为 4 改为 2 又改回
                dropout=0.2,    # 与 baseline 对齐 configs.dropout 为 0.25 改为 0.2
                d_state=128,
                d_conv=4,
                expand=1,  # 2 改 1
                headdim=25,                  # 50 改 25
                use_mem_eff_path=True,
                chunk_size=128,      # 256 改 128
            )
            print("configs.d_model =", configs.d_model)

        else:
            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=configs.output_attention),
                            configs.d_model, configs.n_heads
                        ),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model)
            )

    def forward(self, x_enc, x_mark_enc, subject_ids=None):
        #embedding
        # ---- apply spatial permutation BEFORE embedding/encoder (for Mamba locality) ----
        x_enc1 = x_enc
        x_enc1 = x_enc1.permute(2, 0, 1)  # Change shape to [time_length, batch_size, channel]
        x_enc1 = self.pos_encoder(x_enc1)
        out_enc1 = self.transformer_encoder(x_enc1)
        out_enc1 = out_enc1.permute(1, 2, 0)

        perm = getattr(self.enc_embedding, "perm", None)
        inv_perm = getattr(self.enc_embedding, "inv_perm", None)

        if perm is not None and perm.numel() == x_enc.size(1):
            perm = perm.to(x_enc.device)
            x_enc = x_enc[:, perm, :]  # [B,63,T]

        enc_out = self.enc_embedding(x_enc, x_mark_enc, subject_ids)  # [B,64,250]
        enc_out, _ = self.encoder(enc_out, attn_mask=None)  # [B,64,250]


        if getattr(self.enc_embedding, "subject_embedding", None) is not None:
            enc_out = enc_out[:, :63, :]  # drop subject token -> [B,63,250]

        # if hasattr(self.enc_embedding, "graph_refine"):
        #     enc_out = self.enc_embedding.graph_refine(enc_out)
            # print("graph_refine finish.")

        # ---- restore original channel order for downstream compatibility ----
        if inv_perm is not None and inv_perm.numel() == enc_out.size(1):
            inv_perm = inv_perm.to(enc_out.device)
            enc_out = enc_out[:, inv_perm, :]


        assert enc_out.size(1) == 63
        return enc_out + out_enc1
        # print("enc_out", enc_out.shape)


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # Revised from ShallowNet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = x.unsqueeze(1)
        # print("x", x.shape)
        x = self.tsconv(x)
        # print("tsconv", x.shape)
        x = self.projection(x)
        # print("projection", x.shape)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )


class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class ATMS(nn.Module):
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=2, num_features=64, num_latents=1024,
                 num_blocks=1):
        super(ATMS, self).__init__()
        default_config = Config()
        self.encoder = iTransformer(default_config, backbone="mamba2")
        self.subject_wise_linear = nn.ModuleList(
            [nn.Linear(default_config.d_model, sequence_length) for _ in range(num_subjects)])
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x, subject_ids):
        x = self.encoder(x, None, subject_ids)
        # print(f'After attention shape: {x.shape}')
        # print("x", x.shape)
        # x = self.subject_wise_linear[0](x)
        # print(f'After subject-specific linear transformation shape: {x.shape}')
        eeg_embedding = self.enc_eeg(x)

        out = self.proj_eeg(eeg_embedding)
        return out


def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None


def train_model(sub, eeg_model, dataloader, optimizer, device, text_features_all, img_features_all, config):
    eeg_model.train()
    text_features_all = text_features_all.to(device).float()  # (n_cls, d)
    img_features_all = (img_features_all[::10]).to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    alpha = 0.9
    features_list = []  # List to store features
    save_features = True
    mse_loss_fn = nn.MSELoss()
    for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
        eeg_data = eeg_data.to(device)
        text_features = text_features.to(device).float()
        img_features = img_features.to(device).float()
        labels = labels.to(device)

        optimizer.zero_grad()

        batch_size = eeg_data.size(0)
        subject_id = extract_id_from_string(sub)
        # eeg_data = eeg_data.permute(0, 2, 1)
        subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)
        # if not config.insubject:
        #     subject_ids = torch.full((batch_size,), -1, dtype=torch.long).to(device)
        eeg_features = eeg_model(eeg_data, subject_ids).float()

        features_list.append(eeg_features)
        logit_scale = eeg_model.logit_scale

        img_loss = eeg_model.loss_func(eeg_features, img_features, logit_scale)
        text_loss = eeg_model.loss_func(eeg_features, text_features, logit_scale)
        # loss = img_loss + text_loss
        # print("text_loss", text_loss)
        # print("img_loss", img_loss)

        # 回归项用 L2-normalize 后再算（关键）
        # eeg_n = F.normalize(eeg_features, dim=-1)
        # img_n = F.normalize(img_features, dim=-1)
        regress_loss = mse_loss_fn(eeg_features, img_features)

        loss = (alpha * regress_loss * 10 + (1 - alpha) * img_loss * 10)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

        # Compute the corresponding logits
        logits_img = logit_scale * eeg_features @ img_features_all.T
        # logits_text = logit_scale * eeg_features @ text_features_all.T
        # logits_single = (logits_text + logits_img) / 2.0
        # logits_text = logit_scale * eeg_features @ text_features_all.T
        logits_single = logits_img
        predicted = torch.argmax(logits_single, dim=1)  # (n_batch, ) in {0, 1, ..., n_cls-1}

        batch_size = predicted.shape[0]
        total += batch_size
        correct += (predicted == labels).sum().item()
        del eeg_data, eeg_features, img_features
    average_loss = total_loss / (batch_idx + 1)
    accuracy = correct / total
    return average_loss, accuracy, torch.cat(features_list, dim=0)


def evaluate_model(sub, eeg_model, dataloader, device, text_features_all, img_features_all, k, config):
    eeg_model.eval()

    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    alpha = 0.9
    top5_correct = 0
    top5_correct_count = 0
    # Get all unique classes
    all_labels = set(range(text_features_all.size(0)))
    top5_acc = 0
    mse_loss_fn = nn.MSELoss()
    with torch.no_grad():
        for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()

            batch_size = eeg_data.size(0)
            subject_id = extract_id_from_string(sub)
            # eeg_data = eeg_data.permute(0, 2, 1)
            subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)
            # if not config.insubject:
            #     subject_ids = torch.full((batch_size,), -1, dtype=torch.long).to(device)
            eeg_features = eeg_model(eeg_data, subject_ids)

            logit_scale = eeg_model.logit_scale
            # print(eeg_features.type, text_features.type, img_features.type)
            img_loss = eeg_model.loss_func(eeg_features, img_features, logit_scale)
            text_loss = eeg_model.loss_func(eeg_features, text_features, logit_scale)
            # 回归项用 L2-normalize 后再算（关键）
            # eeg_n = F.normalize(eeg_features, dim=-1)
            # img_n = F.normalize(img_features, dim=-1)

            regress_loss = mse_loss_fn(eeg_features, img_features)
            loss = (alpha * regress_loss * 10 + (1 - alpha) * img_loss*10)

            total_loss += loss.item()

            for idx, label in enumerate(labels):
                # First, select k-1 classes excluding the correct class
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k - 1) + [label.item()]
                selected_img_features = img_features_all[selected_classes]
                selected_text_features = text_features_all[selected_classes]

                if k == 200:
                    # Compute the corresponding logits
                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    # print("logits_single", logits_single.shape)
                    # Get the predicted class
                    # predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    predicted_label = selected_classes[
                        torch.argmax(logits_single).item()]  # (n_batch, ) in {0, 1, ..., n_cls-1}
                    if predicted_label == label.item():
                        # print("predicted_label", predicted_label)
                        correct += 1

                    # logits_single is the model's output, shape (n_batch, n_classes)
                    # label is the true label, shape (n_batch,)
                    # Get the indices of the top-5 predictions
                    # print("logits_single", logits_single)
                    _, top5_indices = torch.topk(logits_single, 5, largest=True)

                    # Check if the true label is in the top-5 predictions
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:
                        top5_correct_count += 1
                    total += 1
                elif k == 50 or k == 100:
                    # For k=50 or 100, select k classes for evaluation
                    selected_classes = random.sample(possible_classes, k - 1) + [label.item()]

                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img

                    predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    if predicted_label == label.item():
                        correct += 1
                    _, top5_indices = torch.topk(logits_single, 5, largest=True)

                    # Check if the true label is in the top-5 predictions
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:
                        top5_correct_count += 1
                    total += 1
                elif k == 2 or k == 4 or k == 10:
                    selected_classes = random.sample(possible_classes, k - 1) + [label.item()]
                    # Compute the corresponding logits
                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    # logits_text = logit_scale * eeg_features[idx] @ selected_text_features.T
                    # logits_single = (logits_text + logits_img) / 2.0
                    logits_single = logits_img
                    # print("logits_single", logits_single.shape)
                    # Get the predicted class
                    # predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    predicted_label = selected_classes[
                        torch.argmax(logits_single).item()]  # (n_batch, ) in {0, 1, ..., n_cls-1}
                    if predicted_label == label.item():
                        correct += 1
                    total += 1
                else:
                    print("Error.")
            del eeg_data, eeg_features, img_features
    average_loss = total_loss / (batch_idx + 1)
    accuracy = correct / total
    top5_acc = top5_correct_count / total
    return average_loss, accuracy, top5_acc


def main_train_loop(sub, current_time, eeg_model, train_dataloader, test_dataloader, optimizer, device,
                    text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all,
                    config, logger=None):
    logger = wandb_logger(config) if logger else None
    logger.watch(eeg_model, logger)
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    v2_accs = []
    v4_accs = []
    v10_accs = []

    best_accuracy = 0.0
    best_v2_acc = 0.0
    best_model_weights = None
    best_epoch_info = {}
    results = []  # List to store results for each epoch

    for epoch in range(config.epochs):
        # Train the model
        train_loss, train_accuracy, features_tensor = train_model(sub, eeg_model, train_dataloader, optimizer, device,
                                                                  text_features_train_all, img_features_train_all,
                                                                  config=config)
        if (epoch + 1) % 5 == 0:
            # Get the current time and format it as a string (e.g., '2024-01-17_15-30-00')
            if config.insubject==True:
                # os.makedirs(f"./models/contrast/{config.encoder_type}/{sub}/{current_time}", exist_ok=True)
                # file_path = f"./models/contrast/{config.encoder_type}/{sub}/{current_time}/{epoch+1}.pth"
                os.makedirs(f"/home/diaoyueqin/hcy/Generation/models/contrast/{config.encoder_type}/mamba_token/{sub}/{current_time}", exist_ok=True)
                file_path = f"/home/diaoyueqin/hcy/Generation/models/contrast/{config.encoder_type}/mamba_token/{sub}/{current_time}/{epoch+1}.pth"
                torch.save(eeg_model.state_dict(), file_path)
            else:
                # os.makedirs(f"./models/contrast/across/{config.encoder_type}/{current_time}", exist_ok=True)
                # file_path = f"./models/contrast/across/{config.encoder_type}/{current_time}/{epoch+1}.pth"
                os.makedirs(f"/home/diaoyueqin/hcy/Generation/models/contrast/across/{config.encoder_type}/mamba_token/{current_time}", exist_ok=True)
                file_path = f"/home/diaoyueqin/hcy/Generation/models/contrast/across/{config.encoder_type}/mamba_token/{current_time}/{epoch+1}.pth"
                torch.save(eeg_model.state_dict(), file_path)
            print(f"Model saved in {file_path}!")
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluate the model
        test_loss, test_accuracy, top5_acc = evaluate_model(sub, eeg_model, test_dataloader, device,
                                                            text_features_test_all, img_features_test_all, k=200,
                                                            config=config)
        _, v2_acc, _ = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all,
                                      img_features_test_all, k=2, config=config)
        _, v4_acc, _ = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all,
                                      img_features_test_all, k=4, config=config)
        _, v10_acc, _ = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all,
                                       img_features_test_all, k=10, config=config)
        _, v50_acc, v50_top5_acc = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all,
                                                  img_features_test_all, k=50, config=config)
        _, v100_acc, v100_top5_acc = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all,
                                                    img_features_test_all, k=100, config=config)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        v2_accs.append(v2_acc)
        v4_accs.append(v4_acc)
        v10_accs.append(v10_acc)

        # Append results for this epoch
        epoch_results = {
            "epoch": epoch + 1,
            # "train_loss": train_loss,
            # "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "v2_acc": v2_acc,
            "v4_acc": v4_acc,
            "v10_acc": v10_acc,
            "top5_acc": top5_acc,
            "v50_acc": v50_acc,
            "v100_acc": v100_acc,
            "v50_top5_acc": v50_top5_acc,
            "v100_top5_acc": v100_top5_acc
        }

        results.append(epoch_results)
        # If the test accuracy of the current epoch is the best, save the model and related information
        # if test_accuracy > best_accuracy:
            # best_accuracy = test_accuracy
            # best_model_weights = model.state_dict().copy()
        if v2_acc > best_v2_acc:
            best_v2_acc = v2_acc

            best_epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "v2_acc": v2_acc,
                "v4_acc": v4_acc,
                "v10_acc": v10_acc
            }
        logger.log({
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy,
            "v2 Accuracy": v2_acc,
            "v4 Accuracy": v4_acc,
            "v10 Accuracy": v10_acc,
            "Epoch": epoch
        })

        print(
            f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Top5 Accuracy: {top5_acc:.4f}")
        print(
            f"Epoch {epoch + 1}/{config.epochs} - v2 Accuracy:{v2_acc} - v4 Accuracy:{v4_acc} - v10 Accuracy:{v10_acc} - v50 Accuracy:{v50_acc} - v100 Accuracy:{v100_acc}")
    # # Load the best model weights
    # model.load_state_dict(best_model_weights)

    # # # Save the best model
    # torch.save(model.state_dict(), '{train_pos_img_text}.pth')

    # Create 5 subplots
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    # Loss curve
    axs[0, 0].plot(train_losses, label='Train Loss')
    axs[0, 0].plot(test_losses, label='Test Loss')
    axs[0, 0].legend()
    axs[0, 0].set_title("Loss Curve")

    # Overall accuracy curve
    axs[0, 1].plot(train_accuracies, label='Train Accuracy')
    axs[0, 1].plot(test_accuracies, label='Test Accuracy')
    axs[0, 1].legend()
    axs[0, 1].set_title("Accuracy Curve")

    # The following are the three new plots you added, assuming you've already calculated the corresponding accuracies
    # 2-class accuracy plot
    axs[1, 0].plot(v2_accs, label='2-class Accuracy')
    axs[1, 0].legend()
    axs[1, 0].set_title("2-Class Accuracy Curve")

    # 4-class accuracy plot
    axs[1, 1].plot(v4_accs, label='4-class Accuracy')
    axs[1, 1].legend()
    axs[1, 1].set_title("4-Class Accuracy Curve")

    # 10-class accuracy plot
    axs[2, 0].plot(v10_accs, label='10-class Accuracy')
    axs[2, 0].legend()
    axs[2, 0].set_title("10-Class Accuracy Curve")

    # Construct the string information for annotation
    info_text = (f"Best Model Info (from Epoch {best_epoch_info['epoch']}):\n"
                 f"Train Loss: {best_epoch_info['train_loss']:.4f}\n"
                 f"Train Accuracy: {best_epoch_info['train_accuracy']:.4f}\n"
                 f"Test Loss: {best_epoch_info['test_loss']:.4f}\n"
                 f"Test Accuracy: {best_epoch_info['test_accuracy']:.4f}\n"
                 f"v2_acc:{best_epoch_info['v2_acc']:.4f}\n"
                 f"v4_acc:{best_epoch_info['v4_acc']:.4f}\n"
                 f"v10_acc:{best_epoch_info['v10_acc']:.4f}")

    axs[2, 1].axis('off')
    axs[2, 1].text(0.5, 0.5, info_text, fontsize=10, ha='center', va='center', transform=axs[2, 1].transAxes)

    plt.tight_layout()

    # Add main title
    plt.suptitle('pos_img_text', fontsize=16, y=1.05)
    plt.savefig('pos_img_text')
    logger.finish()
    return results

import datetime


def count_params(module: nn.Module, trainable_only: bool = False):
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def format_millions(n: int):
    return f"{n/1e6:.3f}M"


def main():
    # Use argparse to parse the command-line arguments
    parser = argparse.ArgumentParser(description='EEG Transformer Training Script')
    parser.add_argument('--data_path', type=str, default="/home/diaoyueqin/hcy/Preprocessed_data_250Hz", help='Path to the EEG dataset')
    parser.add_argument('--output_dir', type=str, default='/home/diaoyueqin/hcy/Generation/outputs/contrast', help='Directory to save output results')
    parser.add_argument('--project', type=str, default="train_pos_img_text_rep", help='WandB project name')
    parser.add_argument('--entity', type=str, default="sustech_rethinkingbci", help='WandB entity name')
    parser.add_argument('--name', type=str, default="lr=1e-4_img_pos_pro_eeg", help='Experiment name')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--logger', type=bool, default=True, help='Enable WandB logging')
    parser.add_argument('--gpu', type=str, default='cuda:0', help='GPU device to use')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu', help='Device to run on (cpu or gpu)')
    parser.add_argument('--insubject', type=bool, default=True, help='In-subject mode or cross-subject mode')
    parser.add_argument('--encoder_type', type=str, default='ATMS', help='Encoder type')
    parser.add_argument('--subjects', nargs='+', default=['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'], help='List of subject IDs (default: sub-01 to sub-10)')
    args = parser.parse_args()

    # Set device based on the argument
    if args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device(args.gpu)
    else:
        device = torch.device('cpu')

    subjects = args.subjects
    current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")

    for sub in subjects:
        eeg_model = globals()[args.encoder_type]()
        eeg_model.to(device)

        optimizer = AdamW(itertools.chain(eeg_model.parameters()), lr=args.lr)  # 不加weight-decay

        if args.insubject:
            train_dataset = EEGDataset(args.data_path, subjects=[sub], train=True)
            test_dataset = EEGDataset(args.data_path, subjects=[sub], train=False)
        else:
            train_dataset = EEGDataset(args.data_path, exclude_subject=sub, subjects=subjects, train=True)
            test_dataset = EEGDataset(args.data_path, exclude_subject=sub, subjects=subjects, train=False)

        # dataset 已经创建完成后：
        ch_names = train_dataset.ch_names  # 你 eegdatasets_leaveone.py 里确实保存了 self.ch_names

        coords = build_coords_from_ch_names(ch_names, montage_name="standard_1020").to(device)

        # 把 coords 注入到模型的 embedding 前端
        # 你需要保证你的 enc_embedding 有一个 set_coords() 方法
        eeg_model.encoder.enc_embedding.set_coords(coords)
        print("[MontageAware] coords injected into encoder embedding:", coords.shape)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                  drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

        text_features_train_all = train_dataset.text_features
        text_features_test_all = test_dataset.text_features
        img_features_train_all = train_dataset.img_features
        img_features_test_all = test_dataset.img_features

        results = main_train_loop(sub, current_time, eeg_model, train_loader, test_loader, optimizer, device,
                                  text_features_train_all, text_features_test_all, img_features_train_all,
                                  img_features_test_all, config=args, logger=args.logger)

        # Save results to a CSV file
        results_dir = os.path.join(args.output_dir, args.encoder_type, "mamba_token", sub, current_time)
        os.makedirs(results_dir, exist_ok=True)

        if args.insubject:
            results_file = f"{results_dir}/{args.encoder_type}_{sub}.csv"
        else:
            results_file = f"{results_dir}/{args.encoder_type}_cross_exclude_{sub}.csv"

        with open(results_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            print(f'Results saved to {results_file}')


if __name__ == '__main__':

    main()
