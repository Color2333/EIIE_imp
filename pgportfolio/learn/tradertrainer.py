import json
import os
import time
import collections
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from pgportfolio.learn.nnagent import NNAgent
from pgportfolio.marketdata.datamatrices import DataMatrices
import logging

Result = collections.namedtuple("Result",
                                [
                                 "test_pv",
                                 "test_log_mean",
                                 "test_log_mean_free",
                                 "test_history",
                                 "config",
                                 "net_dir",
                                 "backtest_test_pv",
                                 "backtest_test_history",
                                 "backtest_test_log_mean",
                                 "training_time"])

class TraderTrainer:
    def __init__(self, config, fake_data=False, restore_dir=None, save_path=None, device="cpu", agent=None):
        self.config = config
        self.train_config = config["training"]
        self.input_config = config["input"]
        self.save_path = save_path
        self.best_metric = 0
        self.writer = None

        np.random.seed(config["random_seed"])
        torch.manual_seed(config["random_seed"])

        config["input"]["fake_data"] = fake_data
        self._matrix = DataMatrices.create_from_config(config)

        self.test_set = self._matrix.get_test_set()
        if not config["training"]["fast_train"]:
            self.training_set = self._matrix.get_training_set()
        
        self.device = torch.device(device)
        
        if agent:
            self._agent = agent
        else:
            self._agent = NNAgent(config, restore_dir, device)

    def _evaluate(self, set_name, *tensors_to_eval):
        if set_name == "test":
            feed = self.test_set
        elif set_name == "training":
            feed = self.training_set
        else:
            raise ValueError()
        
        return self._agent.evaluate_tensors(feed["X"], feed["y"], last_w=feed["last_w"],
                                              setw_func=feed["setw"], tensors_to_eval=tensors_to_eval)

    @staticmethod
    def calculate_upperbound(y):
        array = np.maximum.reduce(y[:, 0, :], 1)
        return np.prod(array)

    def log_between_steps(self, step):
        self._agent._NNAgent__net.eval()

        eval_tensors = ["portfolio_value", "log_mean", "loss", "log_mean_free", "portfolio_weights"]
        v_pv, v_log_mean, v_loss, log_mean_free, weights = self._evaluate("test", *eval_tensors)
        
        if self.writer:
            self.writer.add_scalar('benefit/test', v_pv, step)
            self.writer.add_scalar('log_mean/test', v_log_mean, step)
            self.writer.add_scalar('loss/test', v_loss, step)

            if not self.train_config["fast_train"]:
                train_loss, = self._evaluate("training", "loss")
                self.writer.add_scalar('loss/train', train_loss, step)
                logging.info('training loss is %s\n' % train_loss)

        logging.info('='*30)
        logging.info('step %d' % step)
        logging.info('-'*30)
        logging.info(f'test portfolio value: {v_pv}\nlog_mean: {v_log_mean}\nloss: {v_loss}\nlog_mean_free: {log_mean_free}')
        logging.info('='*30+"\n")

        if not self.train_config["snap_shot"]:
            self._agent.save_model(self.save_path)
        elif v_pv > self.best_metric:
            self.best_metric = v_pv
            logging.info(f"New best model at step {step} with test portfolio value: {v_pv}")
            if self.save_path:
                self._agent.save_model(self.save_path)

    def __init_tensor_board(self, log_file_dir):
        if log_file_dir:
            self.writer = SummaryWriter(log_dir=log_file_dir)

    def __print_upperbound(self):
        upperbound_test = self.calculate_upperbound(self.test_set["y"])
        logging.info("upper bound in test is %s" % upperbound_test)

    def train_net(self, log_file_dir="./tensorboard", index="0"):
        self.__print_upperbound()
        self.__init_tensor_board(log_file_dir)
        
        starttime = time.time()

        for i in range(self.train_config["steps"]):
            batch = self._matrix.next_batch()
            self._agent.train(batch["X"], batch["y"], batch["last_w"], batch["setw"])
            
            if i % 1000 == 0:
                self.log_between_steps(i)

            # Mimic TF's staircase exponential decay: step scheduler every decay_steps
            try:
                decay_steps = int(self.train_config.get("decay_steps", 0))
            except Exception:
                decay_steps = 0
            if decay_steps > 0 and i > 0 and i % decay_steps == 0:
                # delegate scheduler stepping to the agent
                try:
                    self._agent.step_scheduler()
                except Exception:
                    pass

        if self.save_path:
            self._agent = NNAgent(self.config, restore_path=self.save_path, device=self.device)

        pv, log_mean = self._evaluate("test", "portfolio_value", "log_mean")
        logging.warning(f'Training No.{index} finished. Portfolio value: {pv}, log_mean: {log_mean}, training time: {time.time() - starttime:.2f}s')

        return self.__log_result_csv(index, time.time() - starttime)

    def __log_result_csv(self, index, training_time):
        from pgportfolio.trade import backtest
        self._agent._NNAgent__net.eval()
        
        eval_tensors = ["portfolio_value", "log_mean", "pv_vector", "log_mean_free"]
        v_pv, v_log_mean, benefit_array, v_log_mean_free = self._evaluate("test", *eval_tensors)

        # Run backtest with the trained PyTorch agent
        backtester = backtest.BackTest(self.config.copy(), net_dir=None, agent=self._agent)
        backtester.start_trading()

        result = Result(test_pv=[v_pv],
                        test_log_mean=[v_log_mean],
                        test_log_mean_free=[v_log_mean_free],
                        test_history=[",".join(map(str, benefit_array))],
                        config=[json.dumps(self.config)],
                        net_dir=[index],
                        backtest_test_pv=[backtester.test_pv],
                        backtest_test_history=[",".join(map(str, backtester.test_pc_vector))],
                        backtest_test_log_mean=[np.mean(np.log(backtester.test_pc_vector))],
                        training_time=int(training_time))
        
        new_data_frame = pd.DataFrame(result._asdict()).set_index("net_dir")
        csv_dir = './train_package/train_summary.csv'
        if os.path.isfile(csv_dir):
            dataframe = pd.read_csv(csv_dir).set_index("net_dir")
            dataframe = dataframe.append(new_data_frame)
        else:
            dataframe = new_data_frame
        
        if int(index) > 0:
            dataframe.to_csv(csv_dir)
        return result
