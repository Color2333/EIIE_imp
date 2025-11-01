import torch
import torch.optim as optim
import numpy as np
from pgportfolio.constants import LAMBDA
import pgportfolio.learn.network as network

class NNAgent:
    def __init__(self, config, restore_path=None, device="cpu"):
        self.__config = config
        self.__coin_number = config["input"]["coin_number"]
        self.device = torch.device(device)
        
        # Pass rows = coin_number + 1 to include the cash asset as the first row
        # Keep rows equal to coin_number to match the TF implementation
        # (TF's network internally handles the cash/btc bias column). This
        # prevents a mismatch between previous_w shapes (assets-only) used
        # by the training loop and the network expectation.
        self.__net = network.CNN(
            config["input"]["feature_number"],
            self.__coin_number,
            config["input"]["window_size"],
            config["layers"],
            device=device
        ).to(self.device)

        self.__commission_ratio = self.__config["trading"]["trading_consumption"]
        self.regularization_strength = config["training"].get("regularization_strength", 0.0)

        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler(self.optimizer)

        if restore_path:
            self.__net.load_state_dict(torch.load(restore_path, map_location=self.device))
            
        self.__net.train() # Set the model to training mode by default

    def _init_optimizer(self):
        training_method = self.__config["training"]["training_method"]
        learning_rate = self.__config["training"]["learning_rate"]
        # L2 regularization is passed as weight_decay in PyTorch optimizers
        weight_decay = self.__config["training"].get("weight_decay", 0.0)

        if training_method == 'GradientDescent':
            return optim.SGD(self.__net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif training_method == 'Adam':
            # Explicitly set common Adam parameters (betas and eps) to match TF defaults
            # and reduce implicit cross-framework differences.
            return optim.Adam(self.__net.parameters(), lr=learning_rate,
                              weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
        elif training_method == 'RMSProp':
            return optim.RMSprop(self.__net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported training_method: {training_method}")

    def _init_scheduler(self, optimizer):
        decay_steps = self.__config["training"]["decay_steps"]
        decay_rate = self.__config["training"]["decay_rate"]
        # PyTorch's scheduler step is typically called per epoch, not per global step.
        # We will adapt this in the training loop.
        # The gamma for ExponentialLR is decay_rate^(1/decay_steps)
        gamma = decay_rate ** (1 / decay_steps)
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    def _calculate_loss(self, weights, future_price_changes, pv_vector):
        loss_func_name = self.__config["training"]["loss_function"]
        
        log_portfolio_returns = torch.log(torch.sum(weights * future_price_changes, dim=1))

        if loss_func_name == "loss_function4":
            base_loss = -torch.mean(log_portfolio_returns)
        elif loss_func_name == "loss_function5":
            # This loss includes a penalty for low-entropy (non-diverse) portfolios
            entropy_penalty = LAMBDA * torch.mean(torch.sum(-torch.log(1 + 1e-6 - weights), dim=1))
            base_loss = -torch.mean(log_portfolio_returns) + entropy_penalty
        elif loss_func_name == "loss_function6":
            log_pv_vector = torch.log(pv_vector)
            base_loss = -torch.mean(log_pv_vector)
        else:
            # Defaulting to a common portfolio loss
            base_loss = -torch.mean(log_portfolio_returns)

        # Add per-layer L2 regularization to match TF/tflearn behavior where
        # each layer can define its own weight_decay in the config.
        reg_loss = torch.tensor(0.0, device=base_loss.device, dtype=base_loss.dtype)
        try:
            for module in self.__net.modules():
                wd = getattr(module, "_weight_decay", 0.0)
                if wd and wd > 0:
                    for p in module.parameters(recurse=False):
                        reg_loss = reg_loss + wd * torch.sum(p ** 2)
        except Exception:
            # If anything goes wrong, fall back to no per-layer regularization
            reg_loss = reg_loss

        return base_loss + reg_loss

    def train(self, x, y, last_w, setw_func):
        self.__net.train()
        
        x = np.asarray(x, dtype=np.float32)
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        last_w_tensor = torch.tensor(last_w, dtype=torch.float32).to(self.device)
        


        weights = self.__net(x_tensor, last_w_tensor) # Pass with cash component
        
        # Replicate TF logic for calculating metrics
        future_price_changes = torch.cat([torch.ones(y_tensor.shape[0], 1).to(self.device), y_tensor[:, 0, :]], dim=1)
        
        # Portfolio value vector (pv_vector)
        pv_vector = torch.sum(weights * future_price_changes, dim=1)
        
        # Handling commission costs
        future_omega = (weights * future_price_changes) / torch.sum(weights * future_price_changes, dim=1, keepdim=True)
        w_t = future_omega[:-1]
        w_t1 = weights[1:]
        mu = 1 - torch.sum(torch.abs(w_t1[:, 1:] - w_t[:, 1:]), dim=1) * self.__commission_ratio
        pv_vector_with_cost = pv_vector * torch.cat([torch.ones(1).to(self.device), mu], dim=0)

        loss = self._calculate_loss(weights, future_price_changes, pv_vector_with_cost)

        # Ensure gradients are zeroed before backward to avoid accumulation
        # across batches (PyTorch accumulates gradients by default).
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    # Scheduler stepping moved to Trainer to reproduce TF's staircase behavior

        setw_func(weights[:, 1:].detach().cpu().numpy())
        return loss.item()

    def step_scheduler(self):
        """Expose scheduler stepping so Trainer can control when to decay the LR.

        TF uses exponential_decay with staircase=True (decay every decay_steps).
        We mimic that by stepping the PyTorch scheduler from the trainer when
        a global step hits a decay interval.
        """
        if self.scheduler is not None:
            self.scheduler.step()

    def evaluate_tensors(self, x, y, last_w, setw_func, tensors_to_eval):
        self.__net.eval()
        with torch.no_grad():
            x = np.asarray(x, dtype=np.float32)
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
            last_w_tensor = torch.tensor(last_w, dtype=torch.float32).to(self.device)



            weights = self.__net(x_tensor, last_w_tensor)
            setw_func(weights[:, 1:].detach().cpu().numpy())

            future_price_changes = torch.cat([torch.ones(y_tensor.shape[0], 1).to(self.device), y_tensor[:, 0, :]], dim=1)
            pv_vector = torch.sum(weights * future_price_changes, dim=1)
            
            # This is the portfolio value without transaction costs
            portfolio_value = torch.prod(pv_vector).item()
            log_mean_free = torch.mean(torch.log(pv_vector)).item()

            # Calculate metrics with transaction costs
            future_omega = (weights * future_price_changes) / torch.sum(weights * future_price_changes, dim=1, keepdim=True)
            w_t = future_omega[:-1]
            w_t1 = weights[1:]
            mu = 1 - torch.sum(torch.abs(w_t1[:, 1:] - w_t[:, 1:]), dim=1) * self.__commission_ratio
            pv_vector_with_cost = pv_vector * torch.cat([torch.ones(1).to(self.device), mu], dim=0)

            loss = self._calculate_loss(weights, future_price_changes, pv_vector_with_cost).item()
            final_portfolio_value = torch.prod(pv_vector_with_cost).item()
            log_mean = torch.mean(torch.log(pv_vector_with_cost)).item()

            results = []
            for t_name in tensors_to_eval:
                if t_name == "portfolio_value": results.append(final_portfolio_value)
                elif t_name == "log_mean": results.append(log_mean)
                elif t_name == "loss": results.append(loss)
                elif t_name == "log_mean_free": results.append(log_mean_free)
                elif t_name == "portfolio_weights": results.append(weights.cpu().numpy())
                elif t_name == "pv_vector": results.append(pv_vector_with_cost.cpu().numpy())
                else: results.append(None) # Placeholder for other tensors
            return results

    def decide_by_history(self, history, last_w):
        self.__net.eval()
        with torch.no_grad():
            history_tensor = torch.tensor(history, dtype=torch.float32).unsqueeze(0).to(self.device)
            last_w_tensor = torch.tensor(last_w[1:], dtype=torch.float32).unsqueeze(0).to(self.device)
            
            output = self.__net(history_tensor, last_w_tensor)
            return output.squeeze(0).cpu().numpy()

    def save_model(self, path):
        torch.save(self.__net.state_dict(), path)
