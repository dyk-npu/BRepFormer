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
# Attention 模块 (保持不变)
# 作用：融合 [Node, Graph] 两个特征
# ------------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        # inputs: [node_z, graph_z]
        # node_z shape: [Total_Valid_Nodes, Dim]
        stacked = torch.stack(inputs, dim=1) # [N, 2, Dim]
        weights = self.dense_weight(stacked) # [N, 2, 1]
        weights = F.softmax(weights, dim=1)
        outputs = torch.sum(stacked * weights, dim=1) # [N, Dim]
        return outputs


# ------------------------------------------------------------------------------
# [新增] 图分类头
# 作用：接收 Attention 后的节点特征 z，进行池化 (Pooling)，然后分类
# ------------------------------------------------------------------------------
class GraphPoolingClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        """
        Args:
            input_dim: 输入特征维度 (dim_node)
            num_classes: 类别数 (例如 52)
        """
        super().__init__()
        # 这里复用 NonLinearClassifier 的结构，或者你可以写一个简单的 Linear
        self.mlp = NonLinearClassifier(input_dim, num_classes, dropout)

    def forward(self, z, batch_num_nodes):
        """
        Args:
            z (Tensor): Attention 融合后的特征 [Total_Valid_Nodes, Dim]
            batch_num_nodes (Tensor): 每个图包含的节点数量列表 [Batch_Size]
                                      例如 [10, 12, 8...] 表示第1张图10个节点，第2张12个...
        Returns:
            logits (Tensor): [Batch_Size, Num_Classes]
        """
        # 1. 将打平的 z 按照图的归属切分
        # split_z 是一个 tuple，包含 Batch_Size 个 tensor
        z_per_graph = torch.split(z, batch_num_nodes.tolist())

        # 2. Readout / Pooling (这里使用 Mean Pooling)
        # 对每个图的节点特征求平均，得到图级特征
        # [Node_i, Dim] -> [Dim]
        graph_feats = torch.stack([t.mean(dim=0) for t in z_per_graph]) # [Batch_Size, Dim]

        # 3. 分类
        logits = self.mlp(graph_feats) # [Batch_Size, Num_Classes]
        return logits


# ------------------------------------------------------------------------------
# BrepFormer 主模型
# ------------------------------------------------------------------------------
class BrepFormer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        
        # 获取任务类型：'segmentation' (默认) 或 'classification'
        self.task_type = getattr(args, 'task_type', 'segmentation')
        self.num_classes = args.num_classes
        self.batch_size = args.batch_size

        # --- 1. 通用编码器 ---
        self.brep_encoder = BrepEncoder(
            num_degree=128,
            num_distance=64,
            num_edge_dis=64,
            edge_type="multi_hop",
            multi_hop_max_dist=16,
            num_encoder_layers=args.n_layers_encode,
            embedding_dim=args.dim_node,
            ffn_embedding_dim=args.d_model,
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

        # --- 2. 通用 Attention 融合层 ---
        self.attention = Attention(args.dim_node)

        # --- 3. 根据任务类型初始化不同的分类头 ---
        if self.task_type == 'segmentation':
            # 分割：对每个节点分类
            self.classifier = NonLinearClassifier(args.dim_node, args.num_classes, args.dropout)
        
        elif self.task_type == 'classification':
            # 分类：先 Pooling 再分类
            self.classifier = GraphPoolingClassifier(args.dim_node, args.num_classes, args.dropout)
            # 分类任务的标准 Loss
            self.cls_loss_fn = nn.CrossEntropyLoss()

        # 记录指标用
        self.pred = []
        self.label = []

    def forward(self, batch):
        """
        Returns:
            logits: 
                - Segmentation: [Total_Valid_Nodes, Num_Classes]
                - Classification: [Batch_Size, Num_Classes]
        """
        # 1. Encoder 提取特征
        node_emb, graph_emb = self.brep_encoder(batch, last_state_only=True)
        
        # 调整维度 [Seq_Len, B, C] -> [B, Seq_Len, C] -> 去掉 token [B, N, C]
        node_emb = node_emb[0].permute(1, 0, 2)[:, 1:, :]

        # 2. 提取有效节点 (Masking)
        padding_mask = batch["padding_mask"]
        node_pos = torch.where(padding_mask == False)
        
        # node_z: [Total_Valid_Nodes, Dim]
        node_z = node_emb[node_pos]

        # 3. 对齐 Global Graph Feature
        padding_mask_ = ~padding_mask
        # 获取每个图的真实节点数量 [Batch_Size]
        num_nodes_per_graph = torch.sum(padding_mask_.long(), dim=-1)
        
        # 扩展 graph_z 以对齐 node_z
        graph_z = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0).to(graph_emb.device)

        # 4. Attention Feature Fusion (这是你要求的关键点)
        # z: [Total_Valid_Nodes, Dim] —— 包含融合了图信息的节点特征
        z = self.attention([node_z, graph_z])

        # 5. 根据任务分流
        if self.task_type == 'segmentation':
            # 直接对每个节点分类
            return self.classifier(z)
        
        else: # classification
            # 传入 batch_num_nodes 进行池化，然后分类
            # 注意：num_nodes_per_graph 就是每个图的有效节点数
            return self.classifier(z, num_nodes_per_graph)

    def _get_graph_labels(self, batch):
        """分类任务专用：从 batch 中提取图级别的标签"""
        # 原始 label_feature 是 [Total_Valid_Nodes]
        flat_labels = batch["label_feature"].long()
        
        # 我们只需要每个图取一个标签即可 (因为之前脚本给全图节点打了相同标签)
        batch_num_nodes = batch["graph"].batch_num_nodes() # DGL 提供的每个图节点数列表
        
        # 计算切分点，取每个图的第一个节点的标签
        cum_nodes = torch.cumsum(batch_num_nodes, dim=0)
        start_indices = torch.cat([torch.tensor([0], device=self.device), cum_nodes[:-1]])
        
        return flat_labels[start_indices]

    def training_step(self, batch, batch_idx):
        logits = self(batch)

        if self.task_type == 'segmentation':
            # 分割 Loss
            labels = batch["label_feature"].long()
            labels_onehot = F.one_hot(labels, self.num_classes)
            loss = WeightedCrossEntropyLoss(labels_onehot, logits)
        else:
            # 分类 Loss (Standard CrossEntropy)
            labels = self._get_graph_labels(batch)
            loss = self.cls_loss_fn(logits, labels)

        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=self.batch_size)
        return loss

    def training_epoch_end(self, outputs):
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("current_lr", current_lr, on_step=False, on_epoch=True, batch_size=self.batch_size)

    def validation_step(self, batch, batch_idx):
        logits = self(batch)

        if self.task_type == 'segmentation':
            labels = batch["label_feature"].long()
            labels_onehot = F.one_hot(labels, self.num_classes)
            loss = WeightedCrossEntropyLoss(labels_onehot, logits)
            
            preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
        else:
            labels = self._get_graph_labels(batch)
            loss = self.cls_loss_fn(logits, labels)
            
            preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()

        self.log("eval_loss", loss, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.pred.extend(preds)
        self.label.extend(labels_np)
        return loss

    def validation_epoch_end(self, outputs):
        preds_np = np.array(self.pred)
        labels_np = np.array(self.label)
        self.pred = []
        self.label = []
        
        acc = np.mean((preds_np == labels_np).astype(np.int32))
        
        # 区分 Log 名称
        metric_name = "graph_accuracy" if self.task_type == 'classification' else "per_face_accuracy"
        self.log(metric_name, acc, batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        logits = self(batch)
        preds = torch.argmax(logits, dim=-1)
        
        if self.task_type == 'segmentation':
            labels = batch["label_feature"].long()
            # 保留写 txt 的逻辑 (略，同原代码)
            self._write_segmentation_results(batch, preds) # 封装一下原代码的写文件逻辑
        else:
            labels = self._get_graph_labels(batch)

        self.pred.extend(preds.detach().cpu().numpy())
        self.label.extend(labels.detach().cpu().numpy())

    def _write_segmentation_results(self, batch, preds):
        # 将原来的写文件逻辑放在这里，保持代码整洁
        n_graph, max_n_node = batch["padding_mask"].size()[:2]
        node_pos = torch.where(batch["padding_mask"] == False)
        face_feature = -1 * torch.ones([n_graph, max_n_node], device=self.device, dtype=torch.long)
        face_feature[node_pos] = preds[:]
        out_face_feature = face_feature.detach().cpu().numpy()
        
        output_path = pathlib.Path("results/test")
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i in range(n_graph):
            end_index = max_n_node - np.sum((out_face_feature[i][:] == -1).astype(np.int32))
            pred_feature = out_face_feature[i][:end_index + 1]
            file_name = "feature_" + str(batch["id"][i].long().detach().cpu().numpy()) + ".txt"
            file_path = output_path / file_name
            with open(file_path, "a") as f:
                for j in range(end_index):
                    f.write(str(pred_feature[j]) + "\n")

    def test_epoch_end(self, outputs):
        preds_np = np.array(self.pred)
        labels_np = np.array(self.label)
        self.pred = []
        self.label = []

        acc = np.mean((preds_np == labels_np).astype(np.int32))
        metric_name = "graph_accuracy" if self.task_type == 'classification' else "per_face_accuracy"
        
        self.log(metric_name, acc, batch_size=self.batch_size)
        print(f"{metric_name}: {acc}")

        # Per-class accuracy
        per_class_acc = []
        for i in range(self.num_classes):
            class_pos = np.where(labels_np == i)
            if len(class_pos[0]) > 0:
                acc_i = np.mean(preds_np[class_pos] == labels_np[class_pos])
                per_class_acc.append(acc_i)
        
        self.log("per_class_accuracy", np.mean(per_class_acc), batch_size=self.batch_size)
        print(f"Mean Per-Class Accuracy: {np.mean(per_class_acc)}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, 
            threshold=0.001, threshold_mode='rel', min_lr=0.000001, 
            cooldown=2, verbose=False
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1, "monitor": "eval_loss"}}

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False, using_lbfgs=False, **kwargs):
        optimizer.step(closure=optimizer_closure)
        if self.trainer.global_step < 5000:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 5000.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * 0.001