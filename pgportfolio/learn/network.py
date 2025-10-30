import torch
import torch.nn as nn
import torch.nn.functional as F

def allint(l):
    return [int(i) for i in l]

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