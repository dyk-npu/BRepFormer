import os
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .encoder import BrepEncoder
from .layers.blocks import NonLinearClassifier
from .losses import WeightedCrossEntropyLoss

# ------------------------------------------------------------------------------
# 简单的注意力融合模块
# 作用：将两个特征向量（例如节点特征和图全局特征）通过注意力机制加权融合
# ------------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, in_channels):
        """
        初始化注意力模块
        Args:
            in_channels (int): 输入特征的维度 (dim_node)
        """
        super().__init__()
        # 用于计算注意力权重的线性层，将特征映射到 1 个标量权重
        self.dense_weight = nn.Linear(in_channels, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        """
        前向传播
        Args:
            inputs (list of Tensor): 一个包含两个张量的列表 [node_z, graph_z]
                                     其中每个张量的形状都是 [Total_Nodes, C]
        
        Returns:
            outputs (Tensor): 融合后的特征，形状为 [Total_Nodes, C]
        """
        # 1. 堆叠输入
        # 形状变化: list([N, C], [N, C]) -> [N, 2, C]
        stacked = torch.stack(inputs, dim=1)
        
        # 2. 计算注意力分数
        # [N, 2, C] -> [N, 2, 1]
        weights = self.dense_weight(stacked)
        
        # 3. 归一化权重 (在 dim=1，即那两个特征之间做 Softmax)
        # [N, 2, 1]
        weights = F.softmax(weights, dim=1)
        
        # 4. 加权求和
        # [N, 2, C] * [N, 2, 1] -> [N, 2, C] -> Sum(dim=1) -> [N, C]
        outputs = torch.sum(stacked * weights, dim=1)
        
        return outputs


# ------------------------------------------------------------------------------
# BrepFormer 主模型 (LightningModule)
# 作用：处理 B-rep 数据，进行节点级分类（例如面的语义分割）
# ------------------------------------------------------------------------------
class BrepFormer(pl.LightningModule):
    def __init__(self, args):
        """
        初始化模型结构
        Args:
            args: 包含所有超参数的命名空间对象 (d_model, n_heads, dropout 等)
        """
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = args.num_classes
        self.batch_size = args.batch_size
        self.args = args

        # --- 核心编码器 (Graph Transformer) ---
        self.brep_encoder = BrepEncoder(
            num_degree=128,
            num_distance=64,  # embedding table size for distance
            num_edge_dis=64,  # embedding table size for edge distance
            edge_type="multi_hop",
            multi_hop_max_dist=16,
            num_encoder_layers=args.n_layers_encode,
            embedding_dim=args.dim_node,         # 节点特征维度 (C)
            ffn_embedding_dim=args.d_model,      # FFN 内部维度
            num_attention_heads=args.n_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            layerdrop=0.1,
            encoder_normalize_before=True,
            pre_layernorm=True,
            apply_params_init=True,
            activation_fn="gelu",
        )

        # --- 特征融合模块 ---
        # 用于融合 局部节点特征 和 全局图特征
        self.attention = Attention(args.dim_node)

        # --- 分类头 (MLP) ---
        # 输入维度: dim_node, 输出维度: num_classes
        self.classifier = NonLinearClassifier(args.dim_node, args.num_classes, args.dropout)

        # 用于存储验证/测试阶段的预测结果
        self.pred = []
        self.label = []

    def forward(self, batch):
        """
        模型前向传播核心逻辑
        Args:
            batch (dict): 数据字典，包含 'node_data', 'padding_mask', 'graph' 等

        Returns:
            node_seg (Tensor): 每个有效节点的分类概率分布
                               Shape: [Total_Valid_Nodes, num_classes]
        """
        # 1. 通过 Encoder 提取特征
        # node_emb (原始): [Sequence_Len, Batch_Size, Dim] (Fairseq 风格输出)
        # graph_emb: [Batch_Size, Dim] (全局特征)
        node_emb, graph_emb = self.brep_encoder(batch, last_state_only=True)
        
        # 2. 调整 node_emb 的维度
        # 取出最后一层的输出 node_emb[0] 并转置
        # [Seq_Len+1, B, C] -> [B, Seq_Len+1, C] (Seq_Len+1 是因为它包含一个全局虚拟节点)
        node_emb = node_emb[0].permute(1, 0, 2) 
        
        # 去掉第一个 token (全局虚拟节点)，只保留实际的几何节点
        # [B, N, C]
        node_emb = node_emb[:, 1:, :]           

        # 3. 提取有效节点 (Masking)
        # padding_mask: [B, N], True 表示是 Padding, False 表示是真实节点
        padding_mask = batch["padding_mask"]
        
        # 获取所有真实节点的索引位置
        node_pos = torch.where(padding_mask == False)
        
        # Flatten 操作：将 Batch 中所有图的有效节点特征提取出来拼接在一起
        # node_z Shape: [Total_Valid_Nodes, C]  (Total_Valid_Nodes = sum of all nodes in batch)
        node_z = node_emb[node_pos]

        # 4. 对齐全局图特征
        # 计算每个图有多少个真实节点
        padding_mask_ = ~padding_mask
        num_nodes_per_graph = torch.sum(padding_mask_.long(), dim=-1) # [B]
        
        # 将图特征复制扩展，使其与每个对应的节点对齐
        # 例如图1有5个节点，图2有3个节点，graph_emb[0]重复5次，graph_emb[1]重复3次
        # graph_z Shape: [Total_Valid_Nodes, C]
        graph_z = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0).to(graph_emb.device)

        # 5. 特征融合 (Node + Graph)
        # z Shape: [Total_Valid_Nodes, C]
        z = self.attention([node_z, graph_z])
        
        # 6. 分类
        # node_seg Shape: [Total_Valid_Nodes, Num_Classes]
        node_seg = self.classifier(z)
        
        return node_seg

    def training_step(self, batch, batch_idx):
        """
        单步训练
        """
        # 前向传播，获取预测结果
        node_seg = self(batch)
        
        # 获取标签
        # labels Shape: [Total_Valid_Nodes]
        labels = batch["label_feature"].long()
        
        # 转 One-hot 编码
        labels_onehot = F.one_hot(labels, self.num_classes)
        
        # 计算自定义的加权交叉熵损失
        loss = WeightedCrossEntropyLoss(labels_onehot, node_seg)
        
        # 记录日志
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=self.batch_size)
        return loss

    def training_epoch_end(self, outputs):
        """
        训练 epoch 结束时的操作
        """
        # 记录当前学习率
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("current_lr", current_lr, on_step=False, on_epoch=True, batch_size=self.batch_size)

    def validation_step(self, batch, batch_idx):
        """
        单步验证
        """
        node_seg = self(batch)
        labels = batch["label_feature"].long()
        labels_onehot = F.one_hot(labels, self.num_classes)
        
        loss = WeightedCrossEntropyLoss(labels_onehot, node_seg)
        self.log("eval_loss", loss, on_step=False, on_epoch=True, batch_size=self.batch_size)

        # 保存预测结果以便计算 Epoch 级别的指标
        preds = torch.argmax(node_seg, dim=-1).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        self.pred.extend(preds)
        self.label.extend(labels_np)
        return loss

    def validation_epoch_end(self, outputs):
        """
        验证 epoch 结束，计算准确率
        """
        preds_np = np.array(self.pred)
        labels_np = np.array(self.label)
        
        # 清空缓存
        self.pred = []
        self.label = []
        
        # 计算 Face-level Accuracy (逐面准确率)
        per_face_comp = (preds_np == labels_np).astype(np.int32)
        self.log("per_face_accuracy", np.mean(per_face_comp), batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        """
        测试步骤
        """
        node_seg = self(batch)
        preds = torch.argmax(node_seg, dim=-1) # [Total_Valid_Nodes]
        labels = batch["label_feature"].long()
        
        # 过滤掉非法类别 (如果有)
        known_pos = torch.where(labels < self.num_classes)
        self.pred.extend(preds[known_pos].detach().cpu().numpy())
        self.label.extend(labels[known_pos].detach().cpu().numpy())
        
        # --- 将预测结果写入 txt 文件 (与原始逻辑一致) ---
        n_graph, max_n_node = batch["padding_mask"].size()[:2]
        
        # 初始化一个全是 -1 的矩阵来存放预测结果
        # [Batch, Max_Nodes]
        face_feature = -1 * torch.ones([n_graph, max_n_node], device=self.device, dtype=torch.long)
        
        # 将预测值填回对应位置 (Mask 为 False 的位置是有效节点)
        node_pos = torch.where(batch["padding_mask"] == False)
        face_feature[node_pos] = preds[:]
        
        out_face_feature = face_feature.detach().cpu().numpy()
        
        output_path = pathlib.Path("results/test")
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 逐图写入文件
        for i in range(n_graph):
            # 计算该图的实际节点数 (通过统计非 -1 的数量)
            end_index = max_n_node - np.sum((out_face_feature[i][:] == -1).astype(np.int32))
            pred_feature = out_face_feature[i][:end_index + 1]
            
            file_name = "feature_" + str(batch["id"][i].long().detach().cpu().numpy()) + ".txt"
            file_path = output_path / file_name
            
            with open(file_path, "a") as f:
                for j in range(end_index):
                    f.write(str(pred_feature[j]) + "\n")

    def test_epoch_end(self, outputs):
        """
        测试结束，计算详细指标 (Acc, Per-Class Acc, IoU)
        并打印到控制台
        """
        preds_np = np.array(self.pred)
        labels_np = np.array(self.label)
        self.pred = []
        self.label = []

        # 1. 总体准确率 (Per-Face Accuracy)
        per_face_comp = (preds_np == labels_np).astype(np.int32)
        acc = np.mean(per_face_comp)
        self.log("per_face_accuracy", acc, batch_size=self.batch_size)
        
        # 打印总体准确率
        print("-" * 30)
        print(f"Overall Per-Face Accuracy: {acc:.4f}")
        print("-" * 30)

        # 2. 逐类准确率 & IoU 计算
        per_class_acc = []
        per_class_iou = []
        
        print(f"{'Class ID':<10} | {'Accuracy':<10} | {'IoU':<10}")
        print("-" * 35)

        for i in range(self.num_classes):
            # --- 计算类 i 的 Accuracy ---
            class_pos = np.where(labels_np == i)
            if len(class_pos[0]) > 0:
                class_i_preds = preds_np[class_pos]
                class_i_label = labels_np[class_pos]
                acc_i = np.mean(class_i_preds == class_i_label)
                per_class_acc.append(acc_i)
            else:
                acc_i = 0.0 # 或者设置为 None，视情况而定
            
            # --- 计算类 i 的 IoU ---
            label_pos = np.where(labels_np == i)
            pred_pos = np.where(preds_np == i)
            
            iou_i = 0.0
            if len(pred_pos[0]) > 0 or len(label_pos[0]) > 0: # 只要预测或标签里有这个类就算
                Intersection = (preds_np[label_pos] == labels_np[label_pos]).astype(np.int32)
                Union = (preds_np[label_pos] != labels_np[label_pos]).astype(np.int32)
                Union_ = (preds_np[pred_pos] != labels_np[pred_pos]).astype(np.int32)
                
                denom = np.sum(Union) + np.sum(Intersection) + np.sum(Union_)
                if denom > 0:
                    iou_i = np.sum(Intersection) / denom
                per_class_iou.append(iou_i)
            
            # 打印每一类的详细信息 (如果该类存在于标签中)
            if len(class_pos[0]) > 0:
                print(f"{i:<10} | {acc_i:.4f}     | {iou_i:.4f}")

        # 计算平均值
        mean_class_acc = np.mean(per_class_acc)
        mean_iou = np.mean(per_class_iou)

        self.log("per_class_accuracy", mean_class_acc, batch_size=self.batch_size)
        self.log("IoU", mean_iou, batch_size=self.batch_size)

        # 打印平均值
        print("-" * 35)
        print(f"Mean Per-Class Accuracy:   {mean_class_acc:.4f}")
        print(f"Mean IoU:                  {mean_iou:.4f}")
        print("-" * 35)

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, betas=(0.9, 0.999))
        
        # 学习率衰减策略: 当 eval_loss 不再下降时减少 LR
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, 
            threshold=0.001, threshold_mode='rel', min_lr=0.000001, 
            cooldown=2, verbose=False
        )
        
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "epoch", 
                "frequency": 1, 
                "monitor": "eval_loss"
            }
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False, using_lbfgs=False, **kwargs):
        """
        自定义优化步，加入了 Warm-up 策略
        """
        # 更新参数
        optimizer.step(closure=optimizer_closure)
        
        # 如果在前 5000 步内，执行线性 Warm-up
        if self.trainer.global_step < 5000:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 5000.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * 0.001