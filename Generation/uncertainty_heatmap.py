# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from torchvision import transforms
#
# def resize_batch(x, imsize):
#     # 兼容不同 torchvision 版本的 antialias 参数
#     try:
#         r = transforms.Resize((imsize, imsize), antialias=True)
#     except TypeError:
#         r = transforms.Resize((imsize, imsize))
#     return r(x)
#
# def main(
#     pt_path,
#     out_dir,
#     t=96,            # 选第几个样本(0~199)
#     K=10,            # 每个样本生成张数
#     imsize=256,
#     alpha=0.45,
#     cmap="magma"
# ):
#     os.makedirs(out_dir, exist_ok=True)
#
#     x = torch.load(pt_path, map_location="cpu")
#
#     # 期望: [2000,H,W,3] 或 [2000,3,H,W]
#     if isinstance(x, dict):
#         # 如果你 pt 以后是 dict，可在这里改 key
#         raise ValueError(f"Loaded a dict. Please pick the tensor key. Keys={list(x.keys())}")
#
#     if x.dim() == 4 and x.shape[-1] == 3:
#         x = x.permute(0, 3, 1, 2).contiguous()  # [N,3,H,W]
#     elif x.dim() == 4 and x.shape[1] in (1,3):
#         pass
#     else:
#         raise ValueError(f"Unexpected tensor shape: {tuple(x.shape)}")
#
#     x = x.float()
#     if x.max() > 1.5:
#         x = x / 255.0
#     x = x.clamp(0, 1)
#
#     x = resize_batch(x, imsize)  # [N,3,imsize,imsize]
#     N_total = x.shape[0]
#     assert N_total % K == 0, f"Total images {N_total} not divisible by K={K}."
#     N = N_total // K
#     assert 0 <= t < N, f"t must be in [0, {N-1}], got t={t}."
#
#     stack = x[t*K:(t+1)*K]                 # [K,3,H,W]
#     mean_img = stack.mean(dim=0)           # [3,H,W]
#     var_map = stack.var(dim=0, unbiased=False).mean(dim=0)  # [H,W]
#
#     # robust normalize uncertainty to [0,1]
#     # lo = torch.quantile(var_map, 0.02)
#     # hi = torch.quantile(var_map, 0.98)
#     # u = ((var_map - lo) / (hi - lo + 1e-8)).clamp(0, 1)     # [H,W]
#     # 只高亮最不确定的 top_p 区域（例如 top 10%）
#     top_p = 0.13
#     thr = torch.quantile(var_map, 1.0 - top_p)  # 90% 分位作为阈值
#     hi = torch.quantile(var_map, 0.995)  # 上限取更高分位，避免过曝
#
#     u = (var_map - thr) / (hi - thr + 1e-8)  # 低于阈值的会变成负数
#     u = u.clamp(0, 1)
#
#     mean_np = mean_img.permute(1,2,0).numpy()  # [H,W,3]
#     u_np = u.numpy()
#
#     base_path = os.path.join(out_dir, f"sample_{t:03d}_mean.png")
#     heat_path = os.path.join(out_dir, f"sample_{t:03d}_uncertainty.png")
#     over_path = os.path.join(out_dir, f"sample_{t:03d}_overlay.png")
#
#     # 1) 保存均值图
#     plt.imsave(base_path, mean_np)
#
#     # 2) 保存纯不确定性热图（带 colormap）
#     plt.imsave(heat_path, u_np, cmap="hot")
#
#     # 3) 保存叠加图
#     plt.figure(figsize=(6.6, 6))
#     plt.imshow(mean_np)
#     im = plt.imshow(u_np, alpha=0.45, cmap="magma")
#     plt.axis("off")
#     cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
#     cbar.set_label("Uncertainty (higher = less reliable)")
#     plt.tight_layout()
#     plt.savefig(over_path, dpi=250, bbox_inches="tight", pad_inches=0.02)
#     plt.close()
#
#     print("Saved:")
#     print(" -", base_path)
#     print(" -", heat_path)
#     print(" -", over_path)
#
# if __name__ == "__main__":
#     pt_path = "/home/diaoyueqin/hcy/Generation/RF_mamto_generated_imgs_tensor/all_images.pt"
#     out_dir = "/home/diaoyueqin/hcy/Generation/uncertainty_png/rf"
#     main(pt_path, out_dir, t=47, K=10, imsize=256, alpha=0.45, cmap="magma")

# import os
# import glob
# from PIL import Image, ImageDraw, ImageFont
#
# # -----------------------
# # Config (edit if needed)
# # -----------------------
# gen_root = "/home/diaoyueqin/hcy/Generation/RF_CA_mamto_generated_imgs/sub-08"
# gt_root  = "/home/diaoyueqin/hcy/images_set/test_images"
# unc_root = "/home/diaoyueqin/hcy/Generation/uncertainty_png/rfca"
# out_path = "/home/diaoyueqin/hcy/Generation/RF_CA_mamto_generated_imgs/generation.png"
#
# rows = [
#     dict(
#         name="coffee_bean",
#         gen_dir=os.path.join(gen_root, "coffee_bean"),
#         gt_dir=os.path.join(gt_root, "00048_coffee_bean"),
#         unc=os.path.join(unc_root, "sample_047_uncertainty.png"),
#         ovl=os.path.join(unc_root, "sample_047_overlay.png"),
#     ),
#     dict(
#         name="humming_bird",
#         gen_dir=os.path.join(gen_root, "hummingbird"),
#         gt_dir=os.path.join(gt_root, "00097_hummingbird"),
#         unc=os.path.join(unc_root, "sample_096_uncertainty.png"),
#         ovl=os.path.join(unc_root, "sample_096_overlay.png"),
#     ),
#     dict(
#         name="television",
#         gen_dir=os.path.join(gen_root, "television"),
#         gt_dir=os.path.join(gt_root, "00181_television"),
#         unc=os.path.join(unc_root, "sample_180_uncertainty.png"),
#         ovl=os.path.join(unc_root, "sample_180_overlay.png"),
#     ),
# ]
#
# # target cell size for each image tile (change if you want bigger)
# CELL_W, CELL_H = 256, 256
#
# # layout spacing (px)
# MARGIN_X = 20
# MARGIN_Y = 20
# COL_GAP = 8              # gap between adjacent tiles inside a group
# GROUP_GAP = 28           # larger gap between GT | Generated | Uncertainty | Overlay
# ROW_GAP = 18
# LABEL_H = 60             # bottom label area height
#
# BG = (255, 255, 255)
#
# # -----------------------
# # Helpers
# # -----------------------
# IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
#
# def list_images(folder):
#     files = []
#     for ext in IMG_EXTS:
#         files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
#     return sorted(files)
#
# def open_rgb(path):
#     img = Image.open(path)
#     if img.mode not in ("RGB", "RGBA"):
#         img = img.convert("RGB")
#     if img.mode == "RGBA":
#         # white background for transparency
#         bg = Image.new("RGB", img.size, BG)
#         bg.paste(img, mask=img.split()[-1])
#         img = bg
#     return img
#
# def fit_to_cell(img, cell_w=CELL_W, cell_h=CELL_H, bg=BG):
#     # keep aspect ratio, pad to (cell_w, cell_h)
#     img = img.copy()
#     img.thumbnail((cell_w, cell_h), Image.LANCZOS)
#     canvas = Image.new("RGB", (cell_w, cell_h), bg)
#     x = (cell_w - img.size[0]) // 2
#     y = (cell_h - img.size[1]) // 2
#     canvas.paste(img, (x, y))
#     return canvas
#
# def make_blank():
#     return Image.new("RGB", (CELL_W, CELL_H), BG)
#
# # -----------------------
# # Collect tiles per row
# # Each row: [GT] + [10 Gen] + [Unc] + [Overlay]  => 13 tiles
# # -----------------------
# all_rows_tiles = []
# for r in rows:
#     # GT: pick first image in gt folder
#     gt_files = list_images(r["gt_dir"])
#     if len(gt_files) == 0:
#         raise FileNotFoundError(f"No GT image found in: {r['gt_dir']}")
#     gt_tile = fit_to_cell(open_rgb(gt_files[0]))
#
#     # Generated: take first 10 (sorted)
#     gen_files = list_images(r["gen_dir"])
#     if len(gen_files) < 10:
#         # pad with blanks if fewer than 10
#         gen_tiles = [fit_to_cell(open_rgb(p)) for p in gen_files] + [make_blank()] * (10 - len(gen_files))
#     else:
#         gen_tiles = [fit_to_cell(open_rgb(p)) for p in gen_files[:10]]
#
#     # Uncertainty + Overlay
#     if not os.path.exists(r["unc"]):
#         raise FileNotFoundError(f"Uncertainty not found: {r['unc']}")
#     if not os.path.exists(r["ovl"]):
#         raise FileNotFoundError(f"Overlay not found: {r['ovl']}")
#     unc_tile = fit_to_cell(open_rgb(r["unc"]))
#     ovl_tile = fit_to_cell(open_rgb(r["ovl"]))
#
#     tiles = [gt_tile] + gen_tiles + [unc_tile, ovl_tile]  # 13 tiles
#     all_rows_tiles.append(tiles)
#
# # -----------------------
# # Compute canvas size with group gaps
# # Columns indices:
# # 0=GT, 1..10=Gen(10), 11=Unc, 12=Overlay
# # group gaps after col 0, after col 10, after col 11
# # -----------------------
# def total_row_width():
#     w = 0
#     for c in range(13):
#         w += CELL_W
#         if c == 12:
#             break
#         # decide gap after this column
#         if c in (0, 10, 11):
#             w += GROUP_GAP
#         else:
#             w += COL_GAP
#     return w
#
# ROW_W = total_row_width()
# ROW_H = CELL_H
#
# canvas_w = MARGIN_X * 2 + ROW_W
# canvas_h = MARGIN_Y * 2 + 3 * ROW_H + 2 * ROW_GAP + LABEL_H
#
# canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
#
# # Paste tiles
# y = MARGIN_Y
# for row_tiles in all_rows_tiles:
#     x = MARGIN_X
#     for c, tile in enumerate(row_tiles):
#         canvas.paste(tile, (x, y))
#         # advance x
#         if c == 12:
#             break
#         if c in (0, 10, 11):
#             x += CELL_W + GROUP_GAP
#         else:
#             x += CELL_W + COL_GAP
#     y += ROW_H + ROW_GAP
#
# # -----------------------
# # Bottom labels: GT | Generated(10) | Uncertainty | Overlay
# # -----------------------
# draw = ImageDraw.Draw(canvas)
# try:
#     font = ImageFont.truetype("DejaVuSans.ttf", 26)
# except Exception:
#     font = ImageFont.load_default()
#
# label_y = MARGIN_Y + 3 * ROW_H + 2 * ROW_GAP + 10
#
# # compute x-centers for groups
# # GT center at col 0
# # Generated center spanning cols 1..10
# # Unc center at col 11
# # Overlay center at col 12
#
# # precompute left x for each column
# col_lefts = []
# x = MARGIN_X
# for c in range(13):
#     col_lefts.append(x)
#     if c == 12:
#         break
#     if c in (0, 10, 11):
#         x += CELL_W + GROUP_GAP
#     else:
#         x += CELL_W + COL_GAP
#
# def center_of_cols(c0, c1):
#     left = col_lefts[c0]
#     right = col_lefts[c1] + CELL_W
#     return (left + right) // 2
#
# labels = [
#     ("GT", center_of_cols(0, 0)),
#     ("Generated (10)", center_of_cols(1, 10)),
#     ("Uncertainty", center_of_cols(11, 11)),
#     ("Overlay", center_of_cols(12, 12)),
# ]
#
# for text, cx in labels:
#     tw, th = draw.textbbox((0, 0), text, font=font)[2:]
#     draw.text((cx - tw // 2, label_y), text, fill=(0, 0, 0), font=font)
#
# # Save
# os.makedirs(os.path.dirname(out_path), exist_ok=True)
# canvas.save(out_path)
# print(f"[OK] saved to: {out_path}")

import os
from PIL import Image, ImageDraw, ImageFont

# =========================
# Paths
# =========================
gt_root   = "/home/diaoyueqin/hcy/images_set/test_images"

rf_root   = "/home/diaoyueqin/hcy/Generation/RF_mamto_generated_imgs/sub-08"
rfca_root = "/home/diaoyueqin/hcy/Generation/RF_CA_mamto_generated_imgs/sub-08"

unc_rf_root   = "/home/diaoyueqin/hcy/Generation/uncertainty_png/rf"
unc_rfca_root = "/home/diaoyueqin/hcy/Generation/uncertainty_png/rfca"

out_path = "/home/diaoyueqin/hcy/Generation/RF_CA_mamto_generated_imgs/generation.png"

# =========================
# Selection (exactly as you specified)
# Each row will be:
# [GT] + [RF 4 imgs] + [RF unc] + [RF overlay] + [RFCA 4 imgs] + [RFCA unc] + [RFCA overlay]  => 13 tiles
# =========================
rows = [
    dict(
        name="coffee_bean",
        gt_dir=os.path.join(gt_root, "00048_coffee_bean"),
        rf_dir=os.path.join(rf_root, "coffee_bean"),
        rf_sel=["0.png", "4.png", "6.png", "9.png"],
        rf_unc=os.path.join(unc_rf_root, "sample_047_uncertainty.png"),
        rf_ovl=os.path.join(unc_rf_root, "sample_047_overlay.png"),
        rfca_dir=os.path.join(rfca_root, "coffee_bean"),
        rfca_sel=["0.png", "1.png", "4.png", "9.png"],
        rfca_unc=os.path.join(unc_rfca_root, "sample_047_uncertainty.png"),
        rfca_ovl=os.path.join(unc_rfca_root, "sample_047_overlay.png"),
    ),
    dict(
        name="hummingbird",
        gt_dir=os.path.join(gt_root, "00097_hummingbird"),
        rf_dir=os.path.join(rf_root, "hummingbird"),
        rf_sel=["0.png", "1.png", "5.png", "6.png"],
        rf_unc=os.path.join(unc_rf_root, "sample_096_uncertainty.png"),
        rf_ovl=os.path.join(unc_rf_root, "sample_096_overlay.png"),
        rfca_dir=os.path.join(rfca_root, "hummingbird"),
        rfca_sel=["2.png", "4.png", "8.png", "9.png"],
        rfca_unc=os.path.join(unc_rfca_root, "sample_096_uncertainty.png"),
        rfca_ovl=os.path.join(unc_rfca_root, "sample_096_overlay.png"),
    ),
    dict(
        name="television",
        gt_dir=os.path.join(gt_root, "00181_television"),
        rf_dir=os.path.join(rf_root, "television"),
        rf_sel=["0.png", "2.png", "4.png", "6.png"],
        rf_unc=os.path.join(unc_rf_root, "sample_180_uncertainty.png"),
        rf_ovl=os.path.join(unc_rf_root, "sample_180_overlay.png"),
        rfca_dir=os.path.join(rfca_root, "television"),
        rfca_sel=["0.png", "3.png", "4.png", "6.png"],
        rfca_unc=os.path.join(unc_rfca_root, "sample_180_uncertainty.png"),
        rfca_ovl=os.path.join(unc_rfca_root, "sample_180_overlay.png"),
    ),
]

# =========================
# Layout / Style
# =========================
CELL_W, CELL_H = 256, 256

MARGIN_X, MARGIN_Y = 20, 20
COL_GAP = 8
GROUP_GAP = 28     # after GT, after RF(4), after RF ovl, after RFCA(4), after RFCA ovl doesn't matter
ROW_GAP = 18
LABEL_H = 70

BG = (255, 255, 255)
TEXT = (0, 0, 0)

# column groups:
# 0: GT
# 1-4: RF gen (4)
# 5: RF unc
# 6: RF ovl
# 7-10: RFCA gen (4)
# 11: RFCA unc
# 12: RFCA ovl

# add big gaps after col 0, 4, 6, 10
BIG_GAP_AFTER = {0, 4, 6, 10}

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def list_images(folder):
    out = []
    for ext in IMG_EXTS:
        out.extend([p for p in (os.path.join(folder, f) for f in os.listdir(folder)) if p.lower().endswith(ext)])
    return sorted(out)


def open_rgb(path):
    img = Image.open(path)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, BG)
        bg.paste(img, mask=img.split()[-1])
        img = bg
    return img


def fit_to_cell(img, cell_w=CELL_W, cell_h=CELL_H, bg=BG):
    img = img.copy()
    img.thumbnail((cell_w, cell_h), Image.LANCZOS)
    canvas = Image.new("RGB", (cell_w, cell_h), bg)
    x = (cell_w - img.size[0]) // 2
    y = (cell_h - img.size[1]) // 2
    canvas.paste(img, (x, y))
    return canvas


def pick_gt_image(gt_dir):
    # pick the first image file (sorted) as GT
    files = []
    for ext in IMG_EXTS:
        files.extend(sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.lower().endswith(ext)]))
    if not files:
        raise FileNotFoundError(f"No GT image found in: {gt_dir}")
    return files[0]


def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


# =========================
# Build tiles for each row (13 tiles)
# =========================
all_rows_tiles = []
for r in rows:
    gt_path = require_file(pick_gt_image(r["gt_dir"]))
    gt_tile = fit_to_cell(open_rgb(gt_path))

    rf_tiles = []
    for fn in r["rf_sel"]:
        p = require_file(os.path.join(r["rf_dir"], fn))
        rf_tiles.append(fit_to_cell(open_rgb(p)))

    rf_unc_tile = fit_to_cell(open_rgb(require_file(r["rf_unc"])))
    rf_ovl_tile = fit_to_cell(open_rgb(require_file(r["rf_ovl"])))

    rfca_tiles = []
    for fn in r["rfca_sel"]:
        p = require_file(os.path.join(r["rfca_dir"], fn))
        rfca_tiles.append(fit_to_cell(open_rgb(p)))

    rfca_unc_tile = fit_to_cell(open_rgb(require_file(r["rfca_unc"])))
    rfca_ovl_tile = fit_to_cell(open_rgb(require_file(r["rfca_ovl"])))

    tiles = [gt_tile] + rf_tiles + [rf_unc_tile, rf_ovl_tile] + rfca_tiles + [rfca_unc_tile, rfca_ovl_tile]
    assert len(tiles) == 13, f"Row {r['name']} tiles != 13, got {len(tiles)}"
    all_rows_tiles.append(tiles)

# =========================
# Compute column left positions (with group gaps)
# =========================
col_lefts = []
x = MARGIN_X
for c in range(13):
    col_lefts.append(x)
    if c == 12:
        break
    gap = GROUP_GAP if c in BIG_GAP_AFTER else COL_GAP
    x += CELL_W + gap

row_w = (col_lefts[-1] + CELL_W) - MARGIN_X
canvas_w = MARGIN_X * 2 + row_w
canvas_h = MARGIN_Y * 2 + 3 * CELL_H + 2 * ROW_GAP + LABEL_H

canvas = Image.new("RGB", (canvas_w, canvas_h), BG)

# =========================
# Paste tiles
# =========================
y = MARGIN_Y
for row_tiles in all_rows_tiles:
    for c, tile in enumerate(row_tiles):
        canvas.paste(tile, (col_lefts[c], y))
    y += CELL_H + ROW_GAP

# =========================
# Bottom labels
# =========================
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 24)
except Exception:
    font = ImageFont.load_default()

def center_of_cols(c0, c1):
    left = col_lefts[c0]
    right = col_lefts[c1] + CELL_W
    return (left + right) // 2

label_y = MARGIN_Y + 3 * CELL_H + 2 * ROW_GAP + 16

labels = [
    ("GT", center_of_cols(0, 0)),
    ("no control", center_of_cols(1, 4)),
    ("uncertainty", center_of_cols(5, 5)),
    ("overlay", center_of_cols(6, 6)),
    ("control scale = 0.01", center_of_cols(7, 10)),
    ("uncertainty", center_of_cols(11, 11)),
    ("overlay", center_of_cols(12, 12)),
]

for text, cx in labels:
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((cx - tw // 2, label_y), text, fill=TEXT, font=font)

# Save
os.makedirs(os.path.dirname(out_path), exist_ok=True)
canvas.save(out_path)
print(f"[OK] saved to: {out_path}")

