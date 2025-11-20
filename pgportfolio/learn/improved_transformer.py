import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FinancialPositionalEncoding(nn.Module):
    """
    专为金融时间序列设计的可学习位置编码
    """
    def __init__(self, d_model, max_len=5000):
        super(FinancialPositionalEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        nn.init.xavier_uniform_(self.pos_embedding.weight)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return x + self.pos_embedding(positions)

class AttentionPooling(nn.Module):
    """
    使用注意力机制进行时间维度聚合
    """
    def __init__(self, d_model):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        nn.init.xavier_uniform_(self.attention[0].weight)
        nn.init.xavier_uniform_(self.attention[2].weight)
        
    def forward(self, x):
        # x: [batch, assets, window, d_model]
        weights = self.attention(x)  # [batch, assets, window, 1]
        weights = F.softmax(weights, dim=2)
        return torch.sum(x * weights, dim=2)  # [batch, assets, d_model]

class AssetAttention(nn.Module):
    """
    资产间注意力机制，用于建模资产间的关系
    """
    def __init__(self, d_model, n_head):
        super(AssetAttention, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: [batch, assets, d_model]
        attn_output, _ = self.attention(x, x, x)
        return self.norm(x + attn_output)

class MarketContextModule(nn.Module):
    """
    市场上下文模块，生成多维度的市场表示
    """
    def __init__(self, d_model, context_dims=4):
        super(MarketContextModule, self).__init__()
        self.context_dims = context_dims
        input_dims = d_model * context_dims
        self.market_projection = nn.Sequential(
            nn.Linear(input_dims, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )
        
        # 初始化权重
        for module in self.market_projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
        
    def forward(self, asset_representations):
        # asset_representations: [batch, assets, d_model]
        
        # 1. 平均表示 (现有)
        mean_context = torch.mean(asset_representations, dim=1, keepdim=True)  # [batch, 1, d_model]
        
        # 2. 方差表示 (市场波动性)
        var_context = torch.var(asset_representations, dim=1, keepdim=True)  # [batch, 1, d_model]
        
        # 3. 最大最小表示 (极值信息)
        max_context, _ = torch.max(asset_representations, dim=1, keepdim=True)  # [batch, 1, d_model]
        min_context, _ = torch.min(asset_representations, dim=1, keepdim=True)  # [batch, 1, d_model]
        
        # 4. 拼接所有上下文信息
        market_context = torch.cat([mean_context, var_context, max_context, min_context], dim=-1)
        
        # 通过线性层投影回原始维度
        market_context = self.market_projection(market_context)
        
        return market_context

class StochasticDepth(nn.Module):
    """
    Stochastic Depth 正则化技术
    """
    def __init__(self, drop_prob=0.1):
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class ImprovedTransformerNet(nn.Module):
    """
    改进版本的Transformer网络，专门用于投资组合管理
    """
    def __init__(self, feature_number, rows, columns, layers, device, transformer_config):
        super(ImprovedTransformerNet, self).__init__()
        self.device = device
        self.rows = rows
        self.columns = columns
        self.feature_number = feature_number
        
        # 从配置中获取参数
        self.d_model = transformer_config.get("d_model", 128)
        self.nhead = transformer_config.get("nhead", 8)
        self.num_encoder_layers = transformer_config.get("num_encoder_layers", 6)
        self.dim_feedforward = transformer_config.get("dim_feedforward", 512)
        self.dropout = transformer_config.get("dropout", 0.1)
        self.use_asset_attention = transformer_config.get("use_asset_attention", True)
        self.use_market_context = transformer_config.get("use_market_context", True)
        self.pooling_method = transformer_config.get("pooling_method", "attention")
        self.pos_encoding_type = transformer_config.get("pos_encoding_type", "learnable")
        self.context_dimensions = transformer_config.get("context_dimensions", 4)
        self.use_residual = transformer_config.get("residual_connection", True)
        
        # 1. 输入投影层
        self.feature_projection = nn.Linear(feature_number, self.d_model).to(device)
        nn.init.xavier_uniform_(self.feature_projection.weight)
        nn.init.constant_(self.feature_projection.bias, 0.0)

        # 2. 资产特定嵌入
        self.asset_embedding = nn.Embedding(rows, self.d_model).to(device)
        nn.init.xavier_uniform_(self.asset_embedding.weight)

        # 3. 位置编码
        if self.pos_encoding_type == "learnable":
            self.pos_encoder = FinancialPositionalEncoding(self.d_model).to(device)
        else:
            # 默认使用标准位置编码
            self.pos_encoder = nn.Parameter(torch.zeros(1, columns, self.d_model)).to(device)
            nn.init.xavier_uniform_(self.pos_encoder)

        # 4. Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=transformer_config.get("activation", "gelu"),
            batch_first=True,
            norm_first=True
        ).to(device)
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_encoder_layers
        ).to(device)
        
        # 5. Stochastic Depth
        self.stochastic_depth = StochasticDepth(drop_prob=0.1).to(device)

        # 6. 时间聚合模块
        if self.pooling_method == "attention":
            self.time_aggregator = AttentionPooling(self.d_model).to(device)
        else:
            # 默认使用平均池化
            self.time_aggregator = nn.AdaptiveAvgPool2d((1, None)).to(device)

        # 7. 资产间注意力模块
        if self.use_asset_attention:
            self.asset_attention = AssetAttention(self.d_model, self.nhead).to(device)

        # 8. 市场上下文模块
        if self.use_market_context:
            self.market_context = MarketContextModule(self.d_model, self.context_dimensions).to(device)

        # 9. 层归一化
        self.layer_norm = nn.LayerNorm(self.d_model).to(device)

        # 10. 投资组合权重预测头
        self.portfolio_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model // 2, 1)
        ).to(device)

        # 初始化投资组合头
        for module in self.portfolio_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

        # 11. 最终组合层
        self.final_layer = nn.Linear(self.d_model + 1, 1).to(device)
        nn.init.xavier_uniform_(self.final_layer.weight)
        nn.init.constant_(self.final_layer.bias, 0.0)
        
        # 设置权重衰减
        if layers:
            self.final_layer._weight_decay = float(layers[-1].get("weight_decay", 0.0))
            self.feature_projection._weight_decay = float(layers[0].get("weight_decay", 0.0))
            self.transformer_encoder._weight_decay = float(layers[0].get("weight_decay", 0.0))
            self.portfolio_head._weight_decay = float(layers[0].get("weight_decay", 0.0))

        # 12. 现金偏差
        self.btc_bias = nn.Parameter(torch.zeros(1, 1))

    def forward(self, x, previous_w):
        batch_size = x.shape[0]
        
        # 输入标准化
        network = x.permute(0, 2, 3, 1)  # [batch, assets, window, features]
        divisor = network[:, :, -1, 0].unsqueeze(-1).unsqueeze(-1)
        network = network / (divisor + 1e-8)

        # 获取维度信息
        assets, window = network.shape[1], network.shape[2]

        # 1. 特征投影
        network = self.feature_projection(network)  # [batch, assets, window, d_model]

        # 2. 添加资产嵌入
        asset_ids = torch.arange(assets, device=self.device).unsqueeze(0).expand(batch_size, -1)
        asset_emb = self.asset_embedding(asset_ids).unsqueeze(2)  # [batch, assets, 1, d_model]
        asset_emb = asset_emb.expand(-1, -1, window, -1)  # [batch, assets, window, d_model]
        network = network + asset_emb

        # 3. 重塑为序列格式处理每个资产的时间序列
        # [batch, assets, window, d_model] -> [batch*assets, window, d_model]
        batch_size, assets, window, d_model = network.shape
        sequence_input = network.view(batch_size * assets, window, d_model)

        # 4. 添加位置编码
        if self.pos_encoding_type == "learnable":
            sequence_input = self.pos_encoder(sequence_input)
        else:
            sequence_input = sequence_input + self.pos_encoder[:, :window, :]

        # 5. 应用Transformer编码器
        encoded = self.transformer_encoder(sequence_input)
        
        # 6. 应用Stochastic Depth
        encoded = self.stochastic_depth(encoded)
        
        # 7. 重塑回资产维度
        encoded = encoded.view(batch_size, assets, window, d_model)

        # 8. 时间聚合
        if self.pooling_method == "attention":
            asset_representations = self.time_aggregator(encoded)  # [batch, assets, d_model]
        else:
            # 使用平均池化
            asset_representations = torch.mean(encoded, dim=2)  # [batch, assets, d_model]

        # 9. 资产间注意力
        if self.use_asset_attention:
            asset_representations = self.asset_attention(asset_representations)

        # 10. 市场上下文
        if self.use_market_context:
            market_context = self.market_context(asset_representations)  # [batch, 1, d_model]
            market_context = market_context.expand(-1, assets, -1)  # [batch, assets, d_model]
            
            if self.use_residual:
                combined = asset_representations + market_context
            else:
                combined = market_context
        else:
            combined = asset_representations

        # 11. 生成投资组合权重
        combined_flat = combined.view(-1, self.d_model)  # [batch*assets, d_model]
        portfolio_weights_flat = self.portfolio_head(combined_flat)  # [batch*assets, 1]
        portfolio_weights = portfolio_weights_flat.view(batch_size, assets)  # [batch, assets]

        # 12. 与之前的权重结合
        if previous_w.dim() == 1:
            previous_w = previous_w.unsqueeze(0).expand(batch_size, -1)

        # 获取市场级别的表示
        market_level = torch.mean(combined, dim=1, keepdim=True)  # [batch, 1, d_model]
        market_level = market_level.expand(-1, assets, -1)  # [batch, assets, d_model]

        # 为每个资产组合市场上下文和之前的权重
        final_input = torch.cat([market_level, previous_w.unsqueeze(-1)], dim=-1)
        final_input_flat = final_input.view(-1, self.d_model + 1)  # [batch*assets, d_model + 1]
        final_weights_flat = self.final_layer(final_input_flat)  # [batch*assets, 1]
        final_weights = final_weights_flat.view(batch_size, assets)  # [batch, assets]

        # 13. 组合投资组合权重和最终权重
        final_weights = 0.5 * final_weights + 0.5 * portfolio_weights

        # 14. 添加现金偏差并应用softmax
        cash_bias = self.btc_bias.repeat(batch_size, 1)
        final_weights = torch.cat([cash_bias, final_weights], dim=1)  # [batch, assets + 1]

        final_weights = F.softmax(final_weights, dim=1)

        # 15. 应用约束和裁剪
        final_weights = torch.clamp(final_weights, min=0.0, max=1.0)
        final_weights = final_weights / torch.sum(final_weights, dim=1, keepdim=True)

        return final_weights