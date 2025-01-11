import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from dataLoader2 import JSONImageDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
from scipy.stats import wasserstein_distance
class TransformerAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers,ff_dim):
        super(TransformerAttention, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)  # 将输入映射到嵌入维度
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,dim_feedforward=ff_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, embed_dim]
        x = self.transformer(x)  # [seq_len, batch_size, embed_dim]
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, embed_dim]
        return x

class TransformerVAE(nn.Module):
    def __init__(self, input_dim, embed_dim, latent_dim, num_heads, num_layers, seq_len,ff_dim,latent_space):
        super(TransformerVAE, self).__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # TransformerAttention for encoding
        self.transformer = TransformerAttention(input_dim, embed_dim, num_heads, num_layers,ff_dim)

        # VAE Encoder
        self.encoder_fc1 = nn.Linear(embed_dim * seq_len, latent_space)  # Flatten transformer output
        self.encoder_fc2_mean = nn.Linear(latent_space, latent_dim)
        self.encoder_fc2_logvar = nn.Linear(latent_space, latent_dim)

        # VAE Decoder
        self.decoder_fc1 = nn.Linear(latent_dim, latent_space)
        self.decoder_fc2 = nn.Linear(latent_space, embed_dim * seq_len)  # Output shape matches transformer output
        self.output_layer = nn.Linear(embed_dim, input_dim)  # Project back to input_dim

    def encode(self, x):
        # Pass through Transformer
        transformer_output = self.transformer(x)  # [batch_size, seq_len, embed_dim]

        # Flatten sequence and feed to VAE encoder
        print(transformer_output.shape)
        batch_size,seq_len,embed_dim = transformer_output.shape
        flattened = transformer_output.reshape(batch_size, -1)  # [batch_size, seq_len * embed_dim]
        h = F.relu(self.encoder_fc1(flattened))
        mean = self.encoder_fc2_mean(h)
        logvar = self.encoder_fc2_logvar(h)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        # Decode from latent space
        h = F.relu(self.decoder_fc1(z))
        decoded = F.relu(self.decoder_fc2(h))  # [batch_size, seq_len * embed_dim]

        # Reshape back to sequence format
        batch_size = z.size(0)
        decoded = decoded.view(batch_size, self.seq_len, self.embed_dim)  # [batch_size, seq_len, embed_dim]

        # Project back to input dimension
        output = self.output_layer(decoded)  # [batch_size, seq_len, input_dim]
        bbox=torch.sigmoid(output[...,:4])
        smooth_round = lambda x: x - (x - torch.round(x)).detach()
        smooth_bbox=bbox.clone()
        smooth_bbox[...,[0,2]]=smooth_round(smooth_bbox[...,[0,2]]*1440)/1440
        smooth_bbox[...,[1,3]]=smooth_round(smooth_bbox[...,[1,3]]*1920)/1920
        # Concatenate bbox and label probabilities
        #final_output = torch.cat([scaled_bbox, label_probs], dim=-1)
        label_logits=output[...,4:]
        label_probs=F.softmax(label_logits,dim=-1)
        final_output=torch.cat([smooth_bbox,label_probs],dim=-1)
        return final_output

    def forward(self, x):
        # Encode
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)

        # Decode
        recon_x = self.decode(z)
        return recon_x, mean, logvar
def ciou_loss(pred_bboxes, true_bboxes):
    """
    计算 CIoU 损失
    pred_bboxes 和 true_bboxes 的格式: [x1, y1, x2, y2]
    """
    # DIoU 损失的基础计算
    x1 = torch.max(pred_bboxes[..., 0], true_bboxes[..., 0])
    y1 = torch.max(pred_bboxes[..., 1], true_bboxes[..., 1])
    x2 = torch.min(pred_bboxes[..., 2], true_bboxes[..., 2])
    y2 = torch.min(pred_bboxes[..., 3], true_bboxes[..., 3])
    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    pred_area = torch.clamp((pred_bboxes[..., 2] - pred_bboxes[..., 0]), min=1e-6) * \
                torch.clamp((pred_bboxes[..., 3] - pred_bboxes[..., 1]), min=1e-6)
    true_area = torch.clamp((true_bboxes[..., 2] - true_bboxes[..., 0]), min=1e-6) * \
                torch.clamp((true_bboxes[..., 3] - true_bboxes[..., 1]), min=1e-6)
    union_area = pred_area + true_area - inter_area
    iou = inter_area / (union_area + 1e-6)

    pred_center_x = (pred_bboxes[..., 0] + pred_bboxes[..., 2]) / 2
    pred_center_y = (pred_bboxes[..., 1] + pred_bboxes[..., 3]) / 2
    true_center_x = (true_bboxes[..., 0] + true_bboxes[..., 2]) / 2
    true_center_y = (true_bboxes[..., 1] + true_bboxes[..., 3]) / 2
    center_distance = (pred_center_x - true_center_x) ** 2 + (pred_center_y - true_center_y) ** 2

    enclosing_x1 = torch.min(pred_bboxes[..., 0], true_bboxes[..., 0])
    enclosing_y1 = torch.min(pred_bboxes[..., 1], true_bboxes[..., 1])
    enclosing_x2 = torch.max(pred_bboxes[..., 2], true_bboxes[..., 2])
    enclosing_y2 = torch.max(pred_bboxes[..., 3], true_bboxes[..., 3])
    diagonal_distance = (enclosing_x2 - enclosing_x1) ** 2 + (enclosing_y2 - enclosing_y1) ** 2 + 1e-6

    # 宽高比一致性
    pred_w = torch.clamp(pred_bboxes[..., 2] - pred_bboxes[..., 0], min=1e-6)
    pred_h = torch.clamp(pred_bboxes[..., 3] - pred_bboxes[..., 1], min=1e-6)
    true_w = torch.clamp(true_bboxes[..., 2] - true_bboxes[..., 0], min=1e-6)
    true_h = torch.clamp(true_bboxes[..., 3] - true_bboxes[..., 1], min=1e-6)

    v = (4 / (3.14159265359 ** 2)) * (torch.atan(true_w / true_h) - torch.atan(pred_w / pred_h)) ** 2
    alpha = v / (1 - iou + v + 1e-6)

    ciou = iou - center_distance / diagonal_distance - alpha * v
    return 1 - ciou.mean()



# Define loss function
def loss_function(recon_x, x, mean, logvar):
    # Reconstruction loss for bbox (x, y, w, h)
    recon_loss_bbox = F.mse_loss(recon_x[..., :4], x[..., :4], reduction='sum')
    
    # Reconstruction loss for label (categorical cross-entropy)
    recon_loss_label = -torch.sum(x[..., 4:] * torch.log(recon_x[..., 4:] + 1e-9))
    iou_loss=ciou_loss(recon_x[..., :4], x[..., :4])
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    print(iou_loss)
    print(recon_loss_bbox)
    print(recon_loss_label)
    print(kl_loss)
    return iou_loss+recon_loss_bbox + 0.01*recon_loss_label + 0.001*kl_loss
def collate_fn(batch,seq_len=100,padding_value=0):
    """
    将序列调整到固定长度 seq_len。
    """
    adjusted_batch = []
    for sample in batch:
        sample_tensor = torch.tensor(sample)  # 确保样本是 Tensor
        if len(sample) > seq_len:
            # 裁剪
            adjusted_batch.append(sample_tensor[:seq_len])
        else:
            # 填充
            padding = torch.full((seq_len - len(sample), sample_tensor.size(1)), padding_value)
            adjusted_batch.append(torch.cat([sample_tensor, padding], dim=0))

    # 将所有调整后的序列堆叠成一个张量
    return torch.stack(adjusted_batch)
# 模型初始化
input_dim = 29  # x, y, w, h + label 独热向量维度 (4 + 4)
embed_dim = 32
latent_dim = 128
latent_space=256
num_heads = 8
num_layers = 4
ff_dim=1024
seq_len = 100
label_dim = 25  # 独热向量维度
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")  
# 创建数据集和数据加载器
json_folder="/mnt/d/data/RICO/semantic_annotations"
#dataset=JSONImageDataset(json_folder=json_folder)
def DatasetSpilt(json_folder):
    json_list=[os.path.join(json_folder, f) for f in os.listdir(json_folder) if f.endswith('.json')]
    train_data,test_data=train_test_split(json_list,test_size=0.1,random_state=42)
    return train_data,test_data
train_data,test_data=DatasetSpilt(json_folder)
train_dataset=JSONImageDataset(json_folder=train_data)
test_dataset=JSONImageDataset(json_folder=test_data)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_dataloader=DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
model=TransformerVAE(input_dim,embed_dim,latent_dim,num_heads,num_layers,seq_len,ff_dim,latent_space)
model=model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    # 使用 tqdm 包装 DataLoader，添加进度条
    with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
        for batch in pbar:
            # 将 batch 数据移到 GPU
            #print(batch)
            batch = batch.to(device)
            if torch.isnan(batch).any() or torch.isinf(batch).any():
                print("Found NaN or Inf in batch!")
            # 前向传播
            reconstructed, mu, logvar = model(batch)

            # 计算损失
            loss = loss_function(reconstructed, batch, mu, logvar)
            # print("loss")
            # print(loss)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 更新进度条
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_dataloader):.4f}")
def calculate_iou(pred_bboxes, true_bboxes):
    x1 = torch.max(pred_bboxes[..., 0], true_bboxes[..., 0])
    y1 = torch.max(pred_bboxes[..., 1], true_bboxes[..., 1])
    x2 = torch.min(pred_bboxes[..., 2], true_bboxes[..., 2])
    y2 = torch.min(pred_bboxes[..., 3], true_bboxes[..., 3])

    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    pred_area = (pred_bboxes[..., 2] - pred_bboxes[..., 0]) * (pred_bboxes[..., 3] - pred_bboxes[..., 1])
    true_area = (true_bboxes[..., 2] - true_bboxes[..., 0]) * (true_bboxes[..., 3] - true_bboxes[..., 1])

    union_area = pred_area + true_area - inter_area
    iou = inter_area / (union_area + 1e-6)
    return iou.mean()
def calculate_overlap(bboxes):
    num_boxes = bboxes.size(0)
    overlap_sum = 0.0
    for i in range(num_boxes):
        for j in range(i + 1, num_boxes):
            overlap_sum += calculate_iou(bboxes[i].unsqueeze(0), bboxes[j].unsqueeze(0))
    return overlap_sum / (num_boxes * (num_boxes - 1) / 2 + 1e-6)
def calculate_alignment(pred_bboxes, true_bboxes):
    pred_centers = (pred_bboxes[..., :2] + pred_bboxes[..., 2:]) / 2
    true_centers = (true_bboxes[..., :2] + true_bboxes[..., 2:]) / 2
    alignment_error = torch.mean(torch.abs(pred_centers - true_centers))
    return 1 / (alignment_error + 1e-6)  # Alignment 越高越好
def calculate_wasserstein_bbox(pred_bboxes, true_bboxes):
    pred_centers = (pred_bboxes[..., :2] + pred_bboxes[..., 2:]) / 2
    true_centers = (true_bboxes[..., :2] + true_bboxes[..., 2:]) / 2
    return wasserstein_distance(pred_centers.cpu().numpy().flatten(), true_centers.cpu().numpy().flatten())
def calculate_unique_matches(pred_bboxes, true_bboxes, threshold=0.5):
    matches = 0
    for pred_bbox in pred_bboxes:
        max_iou = 0
        for true_bbox in true_bboxes:
            max_iou = max(max_iou, calculate_iou(pred_bbox.unsqueeze(0), true_bbox.unsqueeze(0)))
        if max_iou >= threshold:
            matches += 1
    return matches

def test_model(model, test_loader, device):
    model.eval()  # 设置模型为评估模式
    total_reconstruction_loss = 0.0
    total_kl_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():  # 禁用梯度计算
        for batch_data in test_loader:
            batch_data = batch_data.to(device)  # 将数据移动到 GPU 或 CPU
            reconstructed, mu, logvar = model(batch_data)  # 前向传播

            # 计算重建损失
            size_true = batch_data[:, :, :4]
            size_recon = reconstructed[:, :, :4]
            size_loss = F.mse_loss(size_recon, size_true, reduction='sum')

            # 计算 KL 散度
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            total_reconstruction_loss += size_loss.item()
            total_kl_loss += kl_loss.item()
            total_batches += 1

    # 计算平均损失
    avg_reconstruction_loss = total_reconstruction_loss / len(test_loader.dataset)
    avg_kl_loss = total_kl_loss / len(test_loader.dataset)

    print(f"Test Reconstruction Loss: {avg_reconstruction_loss:.4f}")
    print(f"Test KL Divergence: {avg_kl_loss:.4f}")
    return avg_reconstruction_loss, avg_kl_loss
# 测试过程
test_model(model, test_dataloader, device)
def visualize_layouts_batch(original, reconstructed, num_classes=26, save_prefix="result", device="cuda:7"):
    """
    分批可视化真实布局和生成布局，每次绘制10个边界框，并保存为单独的图片。
    参数:
        original: 真实布局，形状 (seq_len, 4 + num_classes)
        reconstructed: 生成布局，形状 (seq_len, 4 + num_classes)
        num_classes: 类别数（默认为 26）
        save_prefix: 保存图片的前缀
        device: 使用的设备 ("cuda" 或 "cpu")
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    import math

    # 在 GPU 上处理数据
    original_boxes = original[:, :4].to(device)
    reconstructed_boxes = reconstructed[:, :4].to(device)

    # 处理坐标 (假设分辨率为 720x960)
    resolution_x = 720
    resolution_y = 960
    scale_tensor = torch.tensor([resolution_x, resolution_y, resolution_x, resolution_y], device=device)
    original_boxes = original_boxes * scale_tensor
    reconstructed_boxes = reconstructed_boxes * scale_tensor

    # 转换为 NumPy 数组以供绘图
    original_boxes = original_boxes.cpu().detach().numpy()
    reconstructed_boxes = reconstructed_boxes.cpu().detach().numpy()

     # 绘制前10个边界框
    max_boxes = 10
    for i in range(min(max_boxes, len(original_boxes))):
        plt.figure(figsize=(6, 7))
        ax = plt.gca()

        # 绘制真实边界框
        x, y, w, h = original_boxes[i]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="blue", facecolor='none', alpha=0.7)
        ax.add_patch(rect)
        ax.text(x, y, f"Original", color="blue", fontsize=10)  # 添加标签

        # 绘制生成边界框
        x, y, w, h = reconstructed_boxes[i]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="red", linestyle='--', facecolor='none', alpha=0.7)
        ax.add_patch(rect)
        ax.text(x + w, y, f"Reconstructed", color="red", fontsize=10)  # 添加标签

        # 设置标题和坐标轴范围
        #plt.title(f"Layout Visualization")
        plt.xlim(0, resolution_x)
        plt.ylim(0, resolution_y)
        plt.gca().invert_yaxis()
        plt.axis("off")  # 隐藏坐标轴

        # 保存结果
        save_path = f"{save_prefix}_box_{i + 1}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()
# 随机取一个测试样本
original = next(iter(test_dataloader))  # 从测试集中取一个 batch
original = original[0].to("cuda:7")  # 确保数据在 GPU 上
reconstructed, _, _ = model(original.unsqueeze(0))  # 单样本前向传播

# 可视化布局
visualize_layouts_batch(original, reconstructed[0], device="cuda:7")