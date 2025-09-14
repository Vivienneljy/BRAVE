import datetime
import logging
import os

import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter

from util.metrics import class_accuracy


class Memorandum():
    def __init__(self, args):
        self.args = args
        if not os.path.exists('./' + args.log + '/figs/'):
            os.makedirs('./' + args.log + '/figs/')
        self.writer = SummaryWriter('./' + args.log + '/figs/')
        file_name = './' + args.log + '/infs_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
        logging.basicConfig(filename=file_name, level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.idxs_users = None
        self.iter = 0

    def print_param_detail(self):
        self.logger.info('Experimental details:')
        self.logger.info(f'    Data   : {self.args.dataset}')
        self.logger.info(f'    Data Distribution   : {self.args.data_distribution}')
        self.logger.info(f'    Aggregation     : {self.args.aggregation}')
        self.logger.info(f'    Iteration : {self.args.iteration}')
        self.logger.info(f'    Local Batch size   : {self.args.local_bs}')
        self.logger.info(f'    Local Epochs       : {self.args.local_ep}')
        self.logger.info(f'    Users  : {self.args.num_users}')
        self.logger.info(f'    Participants  : {self.args.num_users * self.args.frac}\n')

        self.logger.info('\nAttack details:')
        self.logger.info(f'    Attackers  : {self.args.num_atk}')
        self.logger.info(f'    Source Label  : {self.args.source_label}')
        self.logger.info(f'    Target Label  : {self.args.target_label}')

        self.logger.info('\nDefense details:')
        self.logger.info(f'    Defense  : {self.args.defense}')
        self.logger.info(f'    Idea1  : {self.args.idea1}')
        self.logger.info(f'    Auxiliary Data  : {self.args.auxiliary_data}')
        self.logger.info(f'    Auxiliary Data Size  : {self.args.auxiliary_data_size}')

    def print_checkpoint(self):
        if self.args.save:
            self.logger.info(f'Loading last saved checkpoint: {self.args.checkpoint}\n')

    def print_iteration(self, iter, idxs_users):
        self.iter = iter
        self.idxs_users = idxs_users
        if self.args.save:
            self.logger.info(f'| Global Training Round : {iter + 1} |')

    def print_local_performance(self, acc, loss, asr, index):
        if self.args.save:
            if index < self.args.num_atk:
                self.logger.info(
                    'malicious client {}, mal loss {}, mal acc {}, mal asr {}'.format(index, loss, acc * 100,
                                                                                      asr * 100))
            else:
                self.logger.info('benign client {}, ben loss {}, ben acc {}'.format(index, loss, acc * 100))

    def print_defense_performance(self, classify_score, tpr, fpr, tnr, fnr):
        self.writer.add_scalar('Defense/Classify_Accuracy', classify_score * 100, self.iter + 1)
        self.writer.add_scalar('Defense/Cluster_TPR', tpr * 100, self.iter + 1)
        self.writer.add_scalar('Defense/Cluster_FPR', fpr * 100, self.iter + 1)
        self.writer.add_scalar('Defense/Cluster_TNR', tnr * 100, self.iter + 1)
        self.writer.add_scalar('Defense/Cluster_FNR', fnr * 100, self.iter + 1)

    def print_global_performance_benign(self, acc, loss, actuals, predictions, asr):
        self.writer.add_scalar("Benign/Loss", loss, self.iter + 1)
        self.writer.add_scalar("Benign/Main_Task_Accuracy", acc, self.iter + 1)
        self.writer.add_scalars("Benign/Class_Test_Accuracy", class_accuracy(actuals, predictions), self.iter + 1)
        self.writer.add_scalar("Malicious/Attack_Successfully_Rate", asr * 100, self.iter + 1)

        if self.args.save:
            self.logger.info(f'Aggregate Training Stats after {self.iter + 1} global rounds:')
            self.logger.info(f'Training Loss : {loss}')
            self.logger.info('Global model Benign Test Accuracy: {:.2f}%'.format(100 * acc))
            self.logger.info("Global model Attack Successfully Rate: {:.2f}%".format(asr * 100))

    def print_global_performance_malicious(self, acc, loss, actuals, predictions):
        self.writer.add_scalar("Malicious/Loss", loss, self.iter + 1)
        self.writer.add_scalar("Malicious/Backdoor_Task_Accuracy", 100 * acc, self.iter + 1)
        self.writer.add_scalars("Malicious/Class_Test_Accuracy", class_accuracy(actuals, predictions), self.iter + 1)

        if self.args.save:
            self.logger.info(
                'Global model Malicious Accuracy: {:.2f}%, Malicious Loss: {:.2f}\n'.format(100 * acc, loss))

    def print_brave_anormaly_score(self,vars):
        scores = {
            "Benign": np.mean(vars[self.idxs_users >= self.args.num_atk]),
            "Malicious": np.mean(vars[self.idxs_users < self.args.num_atk])
        }
        self.writer.add_scalars('Defense/Anormaly_Score', scores, self.iter + 1)

    def save_model(self, model):
        if self.args.save:
            if not os.path.exists('./' + self.args.log + '/checkpoints/'):
                os.makedirs('./' + self.args.log + '/checkpoints/')
            state = {'iter': self.iter, 'state_dict': model}
            file_name = './{}/checkpoints/{}.pkl'.format(self.args.log, self.iter)
            torch.save(state, file_name)
