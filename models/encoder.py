from typing import Optional, Tuple
import torch
import torch.nn as nn
from .layers.attention import MultiheadAttention, precompute_freqs_cis
from .layers.embedding import GraphNodeFeature, GraphAttnBias
from .layers.blocks import RMSNorm, SwiGLU, init_params_global

class GraphEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        activation_fn, # unused kept for compat
        export,        # unused kept for compat
        q_noise,
        qn_block_size,
        pre_layernorm,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pre_layernorm = pre_layernorm
        
        self.dropout_module = nn.Dropout(dropout)
        self.activation_dropout_module = nn.Dropout(activation_dropout)

        self.feed_forward = SwiGLU(embedding_dim, ffn_embedding_dim, multiple_of=256)
        
        self.self_attn = MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size
        )
        
        self.self_attn_layer_norm = RMSNorm(embedding_dim, eps=1e-5)
        self.final_layer_norm = RMSNorm(embedding_dim, eps=1e-5)

        rope_theta = 500000
        max_seq_len = 2048
        self.register_buffer("freqs_cis", precompute_freqs_cis(
            embedding_dim // num_attention_heads, max_seq_len * 2, rope_theta
        ))

    def forward(self, x, self_attn_bias=None, self_attn_mask=None, self_attn_padding_mask=None):
        residual = x
        if self.pre_layernorm:
            x = self.self_attn_layer_norm(x)

        batch_size, seq_len, _ = x.shape
        start_pos = 0
        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]

        x, attn = self.self_attn(
            x=x,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
            freqs_cis=freqs_cis,
        )

        x = self.dropout_module(x)
        x = residual + x

        if not self.pre_layernorm:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.pre_layernorm:
            x = self.final_layer_norm(x)

        x = self.feed_forward(x)
        x = self.activation_dropout_module(x)
        x = self.feed_forward(x) # 这里原代码确实调用了两次，必须保留

        x = self.dropout_module(x)
        x = residual + x

        if not self.pre_layernorm:
            x = self.final_layer_norm(x)
        return x, attn

class BrepEncoder(nn.Module):
    def __init__(
            self,
            num_degree: int,
            num_distance: int,
            num_edge_dis: int,
            edge_type: str,
            multi_hop_max_dist: int,
            num_encoder_layers: int = 8,
            embedding_dim: int = 256,
            ffn_embedding_dim: int = 128,
            num_attention_heads: int = 32,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            layerdrop: float = 0.0,
            encoder_normalize_before: bool = False,
            pre_layernorm: bool = False,
            apply_params_init: bool = False,
            activation_fn: str = "gelu",
            embed_scale: float = None,
            freeze_embeddings: bool = False,
            n_trans_layers_to_freeze: int = 0,
            export: bool = False,
            traceable: bool = False,
            q_noise: float = 0.0,
            qn_block_size: int = 8,
    ) -> None:
        super().__init__()
        self.dropout_module = nn.Dropout(dropout)
        self.layerdrop = layerdrop
        self.embedding_dim = embedding_dim
        self.apply_params_init = apply_params_init
        self.traceable = traceable
        self.embed_scale = embed_scale
        self.tanh = nn.Tanh()

        self.graph_node_feature = GraphNodeFeature(
            num_heads=num_attention_heads,
            num_degree=num_degree,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
        )

        self.graph_attn_bias = GraphAttnBias(
            dim_node=embedding_dim,
            num_heads=num_attention_heads,
            num_distance=num_distance,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            n_layers=num_encoder_layers,
        )

        if q_noise > 0:
            from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False), q_noise, qn_block_size
            )
        else:
            self.quant_noise = None

        if encoder_normalize_before:
            self.emb_layer_norm = nn.LayerNorm(self.embedding_dim)
        else:
            self.emb_layer_norm = None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            GraphEncoderLayer(
                embedding_dim=self.embedding_dim,
                ffn_embedding_dim=ffn_embedding_dim,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                activation_fn=activation_fn,
                export=export,
                q_noise=q_noise,
                qn_block_size=qn_block_size,
                pre_layernorm=pre_layernorm,
            ) for _ in range(num_encoder_layers)
        ])

        # 原代码在构建完所有层后，如果 apply_params_init 为 True，会执行全局初始化
        # 注意：这可能会覆盖 GraphNodeFeature 内部的 scaled initialization
        if self.apply_params_init:
            self.apply(init_params_global)

        # Freeze logic
        if freeze_embeddings:
            raise NotImplementedError("Freezing embeddings is not implemented yet.")
        for layer in range(n_trans_layers_to_freeze):
            for p in self.layers[layer].parameters():
                p.requires_grad = False

    def forward(
            self,
            batch_data,
            perturb=None,
            last_state_only: bool = True,
            token_embeddings: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        padding_mask = batch_data["padding_mask"]
        n_graph, n_node = padding_mask.size()[:2]

        if token_embeddings is not None:
            x = token_embeddings
        else:
            x, x_0 = self.graph_node_feature(
                batch_data["node_data"],
                batch_data["face_area"],
                batch_data["face_type"],
                batch_data["face_centroid"],
                batch_data["face_rational"],
                batch_data["padding_mask"]
            )

        if perturb is not None:
            x[:, 1:, :] += perturb

        attn_bias = self.graph_attn_bias(
            batch_data["attn_bias"],
            batch_data["shortest_distance"],
            batch_data["angle"],
            batch_data["centroid_distance"],
            batch_data["edge_path"],
            batch_data["edge_data"],
            batch_data["edge_type"],
            batch_data["edge_length"],
            batch_data["edge_convexity"],
            batch_data["edge_padding_mask"],
            batch_data["graph"],
            x_0
        )

        padding_mask_cls = torch.zeros(n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype)
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)

        if self.embed_scale is not None:
            x = x * self.embed_scale
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)
        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for layer in self.layers:
            x, _ = layer(
                x,
                self_attn_bias=attn_bias,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
            )
            if not last_state_only:
                inner_states.append(x)

        x = x.transpose(0, 1) # T x B x C

        graph_rep = x[0, :, :] # Global node

        if last_state_only:
            inner_states = [x]
        
        if self.traceable:
            return torch.stack(inner_states), graph_rep
        else:
            return inner_states, graph_rep