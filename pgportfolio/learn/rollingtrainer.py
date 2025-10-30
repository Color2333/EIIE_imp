from pgportfolio.learn.tradertrainer import TraderTrainer
import logging

class RollingTrainer(TraderTrainer):
    def __init__(self, config, restore_dir=None, agent=None):
        super().__init__(config, restore_dir=restore_dir,
                         fake_data=True, agent=agent)

    @property
    def agent(self):
        return self._agent

    @property
    def coin_list(self):
        return self._matrix.coin_list

    @property
    def data_matrices(self):
        return self._matrix

    @property
    def rolling_training_steps(self):
        return self.config["trading"]["rolling_training_steps"]

    def __rolling_logging(self):
        fast_train = self.train_config["fast_train"]
        if not fast_train:
            self._agent._net.eval() # Set model to evaluation mode

            v_pv, v_log_mean = self._evaluate("test", "portfolio_value", "log_mean")
            loss_value, = self._evaluate("training", "loss")

            logging.info('rolling training loss is %s\n' % loss_value)
            logging.info('the portfolio value on test asset is %s\nlog_mean is %s\n' % (v_pv, v_log_mean))

    def decide_by_history(self, history, last_w):
        # This now uses the PyTorch agent's method
        return self._agent.decide_by_history(history, last_w)

    def rolling_train(self, online_w=None):
        steps = self.rolling_training_steps
        if steps > 0:
            self._matrix.append_experience(online_w)
            for i in range(steps):
                batch_data = self._matrix.next_batch()
                x = batch_data["X"]
                y = batch_data["y"]
                last_w = batch_data["last_w"]
                setw_func = batch_data["setw"]
                self._agent.train(x, y, last_w, setw_func)
            self.__rolling_logging()
