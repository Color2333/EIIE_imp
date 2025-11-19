import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def allint(l):
    return [int(i) for i in l]

class SqueezeExcitationLayer(nn.Module):
    """
    Implements a Squeeze-and-Excitation block for channel-wise attention.
    It helps the network to learn the importance of different feature channels.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(SqueezeExcitationLayer, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """
    Implements a Residual Block. It adds a shortcut connection that allows
    the gradient to flow more easily and helps in training deeper networks.
    y = Activation(Conv(x) + shortcut(x))
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, device):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        ).to(device)
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)

        self.shortcut = nn.Sequential()
        is_stride_one = isinstance(stride, int) and stride == 1 or \
                        isinstance(stride, (list, tuple)) and all(s == 1 for s in stride)

        if not is_stride_one or in_channels != out_channels:
            # Use a 1x1 convolution to match dimensions if they differ
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            ).to(device)
            nn.init.xavier_uniform_(self.shortcut[0].weight)

    def forward(self, x):
        # Activation is applied *after* this forward pass in the parent CNN module's loop
        return self.conv(x) + self.shortcut(x)

class NeuralNetWork(nn.Module):
    def __init__(self, feature_number, rows, columns, layers, device):
        super(NeuralNetWork, self).__init__()
        self.device = device
        self._rows = rows
        self._columns = columns
        self.layers_conf = layers
        self.feature_number = feature_number
        
        # This will be built by the child class
        self.net = None

    def forward(self, x, previous_w):
        raise NotImplementedError

class CNN(NeuralNetWork):
    def __init__(self, feature_number, rows, columns, layers, device):
        super(CNN, self).__init__(feature_number, rows, columns, layers, device)
        
        self.layers_conf = layers
        if any(layer['type'] == 'EIIE_Output_WithW' for layer in layers):
            self.btc_bias = nn.Parameter(torch.zeros(1, 1))
        
        self._layer_modules = nn.ModuleList()
        self._output_layer_type = None # To store the type of the final output layer

        # Simulate forward pass to build layers and track shapes
        # Dummy input for shape tracking: [batch, features, assets, window]
        dummy_input = torch.randn(1, self.feature_number, self._rows, self._columns).to(self.device)
        # Initial input processing (normalization and transpose)
        network_shape_tracker = dummy_input.permute(0, 2, 3, 1) # [batch, assets, window, features]
        divisor = network_shape_tracker[:, :, -1, 0].unsqueeze(-1).unsqueeze(-1)
        network_shape_tracker = network_shape_tracker / (divisor + 1e-8)
        network_shape_tracker = network_shape_tracker.permute(0, 3, 1, 2) # [batch, features, assets, window]

        for i, layer_conf in enumerate(self.layers_conf):
            layer_type = layer_conf["type"]
            current_module = None

            if layer_type == "ConvLayer":
                in_channels = network_shape_tracker.shape[1]
                out_channels = int(layer_conf["filter_number"])
                kernel_size = tuple(allint(layer_conf["filter_shape"]))
                stride = tuple(allint(layer_conf.get("strides", [1, 1])))
                pad_cfg = layer_conf.get("padding", "valid").lower()
                padding = 0
                if pad_cfg == "same":
                    kh, kw = kernel_size
                    padding = ((kh - 1) // 2, (kw - 1) // 2) # For stride=1, SAME padding
                elif pad_cfg != "valid":
                    try:
                        padding = int(pad_cfg)
                    except ValueError:
                        pass # Default to 0 if invalid

                is_residual = layer_conf.get("residual", False)
                if is_residual:
                    current_module = ResidualBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        device=self.device
                    )
                else:
                    current_module = nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding
                    ).to(self.device)
                    nn.init.xavier_uniform_(current_module.weight)
                    if current_module.bias is not None:
                        nn.init.constant_(current_module.bias, 0.0)
                
                network_shape_tracker = current_module(network_shape_tracker)

            elif layer_type == "EIIE_Dense":
                in_channels = network_shape_tracker.shape[1]
                width = network_shape_tracker.shape[3]
                out_channels = int(layer_conf["filter_number"])
                kernel_size = (1, width)
                
                current_module = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=(1, 1),
                    padding=0
                ).to(self.device)
                nn.init.xavier_uniform_(current_module.weight)
                if current_module.bias is not None:
                    nn.init.constant_(current_module.bias, 0.0)
                
                network_shape_tracker = current_module(network_shape_tracker)

            elif layer_type == "DenseLayer":
                network_shape_tracker = torch.flatten(network_shape_tracker, 1)
                in_features = network_shape_tracker.shape[1]
                
                current_module = nn.Linear(in_features, int(layer_conf["neuron_number"])).to(self.device)
                nn.init.xavier_uniform_(current_module.weight)
                if current_module.bias is not None:
                    nn.init.constant_(current_module.bias, 0.0)
                
                network_shape_tracker = current_module(network_shape_tracker)
            
            elif layer_type == "AttentionLayer":
                in_channels = network_shape_tracker.shape[1]
                reduction_ratio = int(layer_conf.get("reduction_ratio", 16))
                current_module = SqueezeExcitationLayer(
                    in_channels=in_channels,
                    reduction_ratio=reduction_ratio
                ).to(self.device)
                network_shape_tracker = current_module(network_shape_tracker)

            elif layer_type == "DropOut":
                current_module = nn.Dropout(p=1.0 - float(layer_conf["keep_probability"]))
                network_shape_tracker = current_module(network_shape_tracker)

            elif layer_type == "MaxPooling":
                current_module = nn.MaxPool2d(kernel_size=allint(layer_conf["strides"]))
                network_shape_tracker = current_module(network_shape_tracker)

            elif layer_type == "AveragePooling":
                current_module = nn.AvgPool2d(kernel_size=allint(layer_conf["strides"]))
                network_shape_tracker = current_module(network_shape_tracker)

            elif layer_type == "LocalResponseNormalization":
                current_module = nn.LocalResponseNorm(size=5) # Default size, adjust if needed
                network_shape_tracker = current_module(network_shape_tracker)
            
            elif layer_type == "EIIE_Output":
                width = network_shape_tracker.shape[3]
                current_module = nn.Conv2d(network_shape_tracker.shape[1], 1, kernel_size=(1, width)).to(self.device)
                nn.init.xavier_uniform_(current_module.weight)
                if current_module.bias is not None:
                    nn.init.constant_(current_module.bias, 0.0)
                self._output_layer_type = layer_type # Mark as output layer
                # Don't update network_shape_tracker for terminal layers in this loop
            
            elif layer_type == "EIIE_Output_WithW":
                # This layer's input shape depends on previous_w, which is not available in dummy_input
                # We'll create the conv_out module here, but its forward pass logic is complex
                # and will remain in the forward method.
                # The in_channels for this conv_out will be (network_reshaped.shape[3] + w_reshaped.shape[3])
                # which is (width * features + 1)
                # For shape tracking, we need to estimate the concatenated dimension.
                # network_reshaped: [batch, height, 1, width * features]
                # w_reshaped: [batch, height, 1, 1] (after slicing previous_w and reshaping)
                # concatenated: [batch, height, 1, width * features + 1]
                # permuted: [batch, width * features + 1, height, 1]
                
                # Estimate in_channels for the final conv layer
                estimated_in_channels = network_shape_tracker.shape[1] * network_shape_tracker.shape[3] + 1
                current_module = nn.Conv2d(estimated_in_channels, 1, kernel_size=(1, 1), padding=0).to(self.device)
                nn.init.xavier_uniform_(current_module.weight)
                if current_module.bias is not None:
                    nn.init.constant_(current_module.bias, 0.0)
                self._output_layer_type = layer_type # Mark as output layer
                # Don't update network_shape_tracker for terminal layers in this loop

            if current_module is not None:
                try:
                    current_module._weight_decay = float(layer_conf.get("weight_decay", 0.0))
                except Exception:
                    current_module._weight_decay = 0.0
                self._layer_modules.append(current_module)
            
            # Apply activation function to shape tracker if not an output layer
            if "activation_function" in layer_conf and layer_type not in ["EIIE_Output", "EIIE_Output_WithW"]:
                network_shape_tracker = getattr(F, layer_conf["activation_function"])(network_shape_tracker)

        # Store the last layer's configuration for special handling in forward
        self._last_layer_conf = layers[-1] if layers else None


    def forward(self, x, previous_w):
        batch_size = x.shape[0]
        
        # Input processing
        network = x.permute(0, 2, 3, 1)
        divisor = network[:, :, -1, 0].unsqueeze(-1).unsqueeze(-1)
        network = network / (divisor + 1e-8)
        network = network.permute(0, 3, 1, 2)

        # Apply layers from _layer_modules
        for i, layer_module in enumerate(self._layer_modules):
            layer_conf = self.layers_conf[i]
            layer_type = layer_conf["type"]

            if layer_type == "EIIE_Output":
                network = layer_module(network) # This is the conv_out module
                network = network.squeeze(-1).squeeze(-1)
                btc_bias = torch.ones(batch_size, 1).to(self.device)
                network = torch.cat([btc_bias, network], dim=1)
                return F.softmax(network, dim=1)

            elif layer_type == "EIIE_Output_WithW":
                width = network.shape[3]
                height = network.shape[2]
                features = network.shape[1]

                network_reshaped = network.permute(0, 2, 3, 1).reshape(batch_size, height, 1, width * features)

                if previous_w.dim() == 1:
                    previous_w = previous_w.unsqueeze(0)

                expected_asset_cols = height
                actual_asset_cols = previous_w.shape[1]


                w_reshaped = previous_w.reshape(-1, expected_asset_cols, 1, 1)

                concatenated = torch.cat([network_reshaped, w_reshaped], dim=3)
                concatenated = concatenated.permute(0, 3, 1, 2)
                
                network = layer_module(concatenated) # This is the conv_out module
                network = network.squeeze()
                network = network.reshape(batch_size, -1)
                
                btc_bias_tiled = self.btc_bias.repeat(batch_size, 1)
                
                voting = torch.cat([btc_bias_tiled, network], dim=1)
                self.voting = voting
                
                return F.softmax(voting, dim=1)
            
            else: # Regular layer
                network = layer_module(network)
                if "activation_function" in layer_conf:
                    network = getattr(F, layer_conf["activation_function"])(network)

        # If we reach here, it means the last layer was not an EIIE_Output type.
        # This should ideally not happen if the config is well-formed.
        raise ValueError("No supported output layer found in network configuration.")

class CNNLSTM(NeuralNetWork):
    def __init__(self, feature_number, rows, columns, layers, device, lstm_config):
        super(CNNLSTM, self).__init__(feature_number, rows, columns, layers, device)

        self.conv_layers = nn.ModuleList()
        self.conv_activations = []

        # Simulate forward pass to build layers and track shapes correctly
        dummy_input = torch.randn(1, self.feature_number, self._rows, self._columns).to(self.device)
        
        # Correctly process dummy input to match the forward pass logic
        shape_tracker = dummy_input.permute(0, 2, 3, 1)
        divisor = shape_tracker[:, :, -1, 0].unsqueeze(-1).unsqueeze(-1)
        shape_tracker = shape_tracker / (divisor + 1e-8)
        shape_tracker = shape_tracker.permute(0, 3, 1, 2) # Back to [batch, features, assets, window]

        for layer_conf in layers:
            layer_type = layer_conf.get("type")
            module = None
            activation_fn = None

            if "activation_function" in layer_conf:
                 activation_fn = getattr(F, layer_conf["activation_function"], None)

            if layer_type == "ConvLayer":
                in_channels = shape_tracker.shape[1]
                out_channels = int(layer_conf["filter_number"])
                kernel_size = tuple(allint(layer_conf["filter_shape"]))
                stride = tuple(allint(layer_conf.get("strides", [1, 1])))
                padding_val = layer_conf.get("padding", "valid").lower()
                padding = 0
                if padding_val == "same":
                    # Basic 'same' padding for stride 1
                    padding = (kernel_size[0] // 2, kernel_size[1] // 2)
                
                module = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding).to(device)
                nn.init.xavier_uniform_(module.weight)
            
            elif layer_type == "MaxPooling":
                module = nn.MaxPool2d(kernel_size=allint(layer_conf["strides"])).to(device)

            if module:
                self.conv_layers.append(module)
                self.conv_activations.append(activation_fn)
                shape_tracker = module(shape_tracker)
                if activation_fn:
                    shape_tracker = activation_fn(shape_tracker)
        
        # Determine LSTM input size from the final shape of the CNN part
        # shape_tracker is [batch, channels, assets, window]
        # We flatten features for each asset: channels * window
        conv_output_dim = shape_tracker.shape[1] * shape_tracker.shape[3]

        # Build LSTM part
        self.lstm_hidden_size = lstm_config.get("hidden_size", 64)
        self.lstm_num_layers = lstm_config.get("num_layers", 1)
        self.lstm = nn.LSTM(input_size=conv_output_dim,
                              hidden_size=self.lstm_hidden_size,
                              num_layers=self.lstm_num_layers,
                              batch_first=True).to(device)

        # Build Output part
        self.output_layer = nn.Linear(self.lstm_hidden_size, 1).to(device)
        self.btc_bias = nn.Parameter(torch.zeros(1, 1))

    def forward(self, x, previous_w):
        # Input normalization from CNN class
        network = x.permute(0, 2, 3, 1)
        divisor = network[:, :, -1, 0].unsqueeze(-1).unsqueeze(-1)
        network = network / (divisor + 1e-8)
        network = network.permute(0, 3, 1, 2)

        # CNN part
        for layer, act_fn in zip(self.conv_layers, self.conv_activations):
            network = layer(network)
            if act_fn:
                network = act_fn(network)
        
        # Reshape for LSTM
        # network shape: [batch, channels, assets, window_dim]
        # We want [batch, assets, features] for LSTM (seq_len = assets)
        batch_size = network.shape[0]
        # -> [batch, assets, channels, window_dim]
        network = network.permute(0, 2, 1, 3) 
        # -> [batch, assets, channels * window_dim]
        lstm_input = torch.flatten(network, start_dim=2) 
        
        # LSTM part
        # input shape: [batch_size, seq_len, input_size]
        lstm_out, _ = self.lstm(lstm_input) # output shape: [batch, seq_len, hidden_size]
        
        # Output part
        # -> [batch, assets, 1]
        asset_weights = self.output_layer(lstm_out)
        
        # -> [batch, assets]
        asset_weights = asset_weights.squeeze(-1) 

        # Add cash bias and apply softmax
        cash_bias = self.btc_bias.repeat(batch_size, 1)
        # -> [batch, assets + 1]
        final_weights = torch.cat([cash_bias, asset_weights], dim=1)
        
        return F.softmax(final_weights, dim=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PureTransformerNet(NeuralNetWork):
    def __init__(self, feature_number, rows, columns, layers, device, transformer_config):
        super(PureTransformerNet, self).__init__(feature_number, rows, columns, layers, device)

        # 1. Input projection: directly project features to d_model
        # Input shape: [batch, features, assets, window] -> [batch, assets*window, features]
        self.d_model = transformer_config.get("d_model", 128)
        self.feature_projection = nn.Linear(feature_number, self.d_model).to(device)
        nn.init.xavier_uniform_(self.feature_projection.weight)
        nn.init.constant_(self.feature_projection.bias, 0.0)

        # 2. Asset-specific embeddings to distinguish between different assets
        self.asset_embedding = nn.Embedding(rows, self.d_model).to(device)
        nn.init.xavier_uniform_(self.asset_embedding.weight)

        # 3. Positional encoding for time dimension
        self.pos_encoder = PositionalEncoding(self.d_model, transformer_config.get("dropout", 0.1)).to(device)

        # 4. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=transformer_config.get("nhead", 8),
            dim_feedforward=transformer_config.get("dim_feedforward", 512),
            dropout=transformer_config.get("dropout", 0.1),
            batch_first=True,  # Use batch_first for better usability
            norm_first=True
        ).to(device)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=transformer_config.get("num_encoder_layers", 6)
        ).to(device)

        # 5. Layer normalization
        self.layer_norm = nn.LayerNorm(self.d_model).to(device)

        # 6. Output heads
        # 6a. Portfolio weights prediction head
        self.portfolio_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model // 2, 1)
        ).to(device)

        # Initialize portfolio head
        for module in self.portfolio_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

        # 7. Final combination with previous weights - process each asset individually
        self.final_layer = nn.Linear(self.d_model + 1, 1).to(device)
        nn.init.xavier_uniform_(self.final_layer.weight)
        nn.init.constant_(self.final_layer.bias, 0.0)
        self.final_layer._weight_decay = float(layers[-1].get("weight_decay", 0.0)) if layers else 0.0

        # 8. Cash bias
        self.btc_bias = nn.Parameter(torch.zeros(1, 1))

        # Set weight decay for transformer components
        self.feature_projection._weight_decay = float(layers[0].get("weight_decay", 0.0)) if layers else 0.0
        self.transformer_encoder._weight_decay = float(layers[0].get("weight_decay", 0.0)) if layers else 0.0
        self.portfolio_head._weight_decay = float(layers[0].get("weight_decay", 0.0)) if layers else 0.0
  
    def forward(self, x, previous_w):
        batch_size = x.shape[0]

        # Input normalization
        network = x.permute(0, 2, 3, 1)  # [batch, assets, window, features]
        divisor = network[:, :, -1, 0].unsqueeze(-1).unsqueeze(-1)
        network = network / (divisor + 1e-8)

        # Reshape to sequence: [batch, assets, window, features]
        assets, window = network.shape[1], network.shape[2]

        # 1. Feature projection: project features to d_model
        # [batch, assets, window, features] -> [batch, assets, window, d_model]
        network = self.feature_projection(network)

        # 2. Add asset embeddings to distinguish different assets
        asset_ids = torch.arange(assets, device=self.device).unsqueeze(0).expand(batch_size, -1)
        asset_emb = self.asset_embedding(asset_ids).unsqueeze(2)  # [batch, assets, 1, d_model]
        asset_emb = asset_emb.expand(-1, -1, window, -1)  # [batch, assets, window, d_model]
        network = network + asset_emb

        # 3. Reshape to treat each time step as a token in the sequence
        # [batch, assets, window, d_model] -> [batch, assets*window, d_model]
        batch_size, assets, window, d_model = network.shape
        sequence_input = network.view(batch_size, assets * window, d_model)

        # 4. Apply transformer on the full time sequence for each asset
        # This preserves temporal patterns across 31 days
        # [batch, assets*window, d_model] -> [batch, assets*window, d_model]
        encoded = self.transformer_encoder(sequence_input)
        encoded = self.layer_norm(encoded)

        # 5. Reshape back to separate assets and time
        # [batch, assets*window, d_model] -> [batch, assets, window, d_model]
        encoded = encoded.view(batch_size, assets, window, d_model)

        # 6. Aggregate time information with attention-weighted average
        # This preserves important temporal patterns while reducing dimension
        # Simple yet effective: use learned weights for different time steps
        time_weights = torch.softmax(torch.mean(encoded, dim=-1), dim=-1)  # [batch, assets, window]
        time_weights = time_weights.unsqueeze(-1)  # [batch, assets, window, 1]

        # Weighted aggregation across time dimension
        asset_representations = torch.sum(encoded * time_weights, dim=2)  # [batch, assets, d_model]

        # 7. Simple market context (no additional attention computation)
        market_context = torch.mean(asset_representations, dim=1, keepdim=True)  # [batch, 1, d_model]
        market_context = market_context.expand(-1, assets, -1)  # [batch, assets, d_model]

        # 8. Combine with original asset representations
        combined = asset_representations + market_context  # [batch, assets, d_model]

        # 9. Generate portfolio weights
        # Reshape combined to [batch*assets, d_model] for processing
        combined_flat = combined.view(-1, self.d_model)  # [batch*assets, d_model]
        portfolio_weights_flat = self.portfolio_head(combined_flat)  # [batch*assets, 1]
        portfolio_weights = portfolio_weights_flat.view(batch_size, assets)  # [batch, assets]

        # 10. Combine with previous weights for final decision
        # previous_w: [batch, assets] -> [batch, assets]
        if previous_w.dim() == 1:
            previous_w = previous_w.unsqueeze(0).expand(batch_size, -1)

        # Get market-level representation (average of all assets)
        market_level = torch.mean(combined, dim=1, keepdim=True)  # [batch, 1, d_model]
        market_level = market_level.expand(-1, assets, -1)  # [batch, assets, d_model]

        # For each asset, combine market context with its previous weight
        final_input = torch.cat([market_level, previous_w.unsqueeze(-1)], dim=-1)
        # final_input shape: [batch, assets, d_model + 1]

        # Reshape for processing all assets at once
        final_input_flat = final_input.view(-1, self.d_model + 1)  # [batch*assets, d_model + 1]
        final_weights_flat = self.final_layer(final_input_flat)    # [batch*assets, 1]
        final_weights = final_weights_flat.view(batch_size, assets)  # [batch, assets]

        # 9. Combine with portfolio weights and add cash bias
        final_weights = 0.5 * final_weights + 0.5 * portfolio_weights

        # 10. Add cash bias and apply softmax
        cash_bias = self.btc_bias.repeat(batch_size, 1)
        final_weights = torch.cat([cash_bias, final_weights], dim=1)  # [batch, assets + 1]

        final_weights = F.softmax(final_weights, dim=1)

        # Apply cash flow constraints and clipping
        # Ensure no negative weights (no short selling)
        final_weights = torch.clamp(final_weights, min=0.0, max=1.0)

        # Renormalize to ensure sum = 1
        final_weights = final_weights / torch.sum(final_weights, dim=1, keepdim=True)

        return final_weights