

r"""
SUBCDR.trainer.trainer
################################
"""

import numpy as np
from recbole.trainer import Trainer
from SUBCDR.utils import train_mode2state
import wandb
from tqdm import tqdm
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.cuda.amp as amp
from recbole.utils import (
    ensure_dir,
    get_local_time,
    early_stopping,
    calculate_valid_score,
    dict2str,
    EvaluatorType,
    KGDataLoaderState,
    get_tensorboard,
    set_color,
    get_gpu_usage,
    WandbLogger,
)
from time import time
from recbole.data.interaction import Interaction
from recbole.evaluator import Evaluator, Collector
import math
class CrossDomainTrainer(Trainer):
    r"""Trainer for training cross-domain models. It contains four training mode: SOURCE, TARGET, BOTH, OVERLAP
    which can be set by the parameter of `train_epochs`
    """

    def __init__(self, config, model):
        super(CrossDomainTrainer, self).__init__(config, model)
        self.train_modes = config['train_modes']
        self.train_epochs = config['epoch_num']
        self.split_valid_flag = config['source_split']

    def _reinit(self, phase):
        """Reset the parameters when start a new training phase.
        """
        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.item_tensor = None
        self.tot_item_num = None
        self.train_loss_dict = dict()
        self.epochs = int(self.train_epochs[phase])
        self.eval_step = min(self.config['eval_step'], self.epochs)

    def fit(self, train_data, test_data=None, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None,src_test_neg=None,src_valid_neg=None, tgt_test_neg=None,tgt_valid_neg=None):
        r"""Train the model based on the train data and the valid data.

            Args:
                train_data (DataLoader): the train data
                valid_data (DataLoader, optional): the valid data, default: None.
                                                    If it's None, the early_stopping is invalid.
                verbose (bool, optional): whether to write training and evaluation information to logger, default: True
                saved (bool, optional): whether to save the model parameters, default: True
                show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
                callback_fn (callable): Optional callback function executed at end of epoch.
                                        Includes (epoch_idx, valid_score) input arguments.

            Returns:
                    (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        for phase in range(len(self.train_modes)):
            self._reinit(phase)
            scheme = self.train_modes[phase]
            self.logger.info("Start training with {} mode".format(scheme))
            state = train_mode2state[scheme]
            train_data.set_mode(state)
            self.model.set_phase(scheme)
            if self.model.skip_target:
                if scheme == 'TARGET':
                    continue
            self.optimizer = self._build_optimizer(params=filter(lambda p: p.requires_grad, self.model.parameters()))
            if self.split_valid_flag and valid_data is not None:
                source_valid_data, target_valid_data = valid_data
                if scheme == 'SOURCE':
                    super().fit(train_data, test_data, source_valid_data, verbose, saved, show_progress, callback_fn, src_test_neg=src_test_neg,
            src_valid_neg=src_valid_neg,
            tgt_test_neg=tgt_test_neg,
            tgt_valid_neg=tgt_valid_neg)
                else:
                    if saved and self.start_epoch >= self.epochs:
                        self._save_checkpoint(-1, verbose=verbose)
                    # self._valid_epoch(
                    #     valid_data, show_progress=show_progress, src_test_neg=src_test_neg,
                    #     src_valid_neg=src_valid_neg, tgt_test_neg=tgt_test_neg, tgt_valid_neg=tgt_valid_neg
                    # )
                    self.eval_collector.data_collect(train_data)
                    if self.config["train_neg_sample_args"].get("dynamic", False):
                        train_data.get_model(self.model)
                    valid_step = 0
                    #gai
                    func = self.model.calculate_loss_subcdr
                    for epoch_idx in range(self.start_epoch, self.epochs):
                        # train

                        training_start_time = time()
                        train_loss = self._train_epoch(
                            train_data, epoch_idx, show_progress=show_progress, loss_func=func
                        )
                        mywandb = 0
                        if mywandb == 1:
                            wandb.log({"loss": train_loss})
                        self.train_loss_dict[epoch_idx] = (
                            sum(train_loss) if isinstance(train_loss, tuple) else train_loss
                        )
                        training_end_time = time()
                        train_loss_output = self._generate_train_loss_output(
                            epoch_idx, training_start_time, training_end_time, train_loss
                        )
                        if verbose:
                            self.logger.info(train_loss_output)
                        self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
                        self.wandblogger.log_metrics(
                            {"epoch": epoch_idx, "train_loss": train_loss, "train_step": epoch_idx},
                            head="train",
                        )

                        # eval
                        if self.eval_step <= 0 or not valid_data:
                            if saved:
                                self._save_checkpoint(epoch_idx, verbose=verbose)
                            continue
                        if (epoch_idx + 1) % self.eval_step == 0:
                            valid_start_time = time()
                            valid_score, valid_result = self._valid_epoch(
                                valid_data, show_progress=show_progress, src_test_neg=src_test_neg,
                                src_valid_neg=src_valid_neg, tgt_test_neg=tgt_test_neg, tgt_valid_neg=tgt_valid_neg
                            )

                            (
                                self.best_valid_score,
                                self.cur_step,
                                stop_flag,
                                update_flag,
                            ) = early_stopping(
                                valid_score,
                                self.best_valid_score,
                                self.cur_step,
                                max_step=self.stopping_step,
                                bigger=self.valid_metric_bigger,
                            )
                            valid_end_time = time()
                            valid_score_output = (
                                                         set_color("epoch %d evaluating", "green")
                                                         + " ["
                                                         + set_color("time", "blue")
                                                         + ": %.2fs, "
                                                         + set_color("valid_score", "blue")
                                                         + ": %f]"
                                                 ) % (
                                                 epoch_idx, valid_end_time - valid_start_time, valid_result['hit@10'])
                            valid_result_output = (
                                    set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
                            )
                            if verbose:
                                self.logger.info(valid_score_output)
                                self.logger.info(valid_result_output)
                            # self.tensorboard.add_scalar("Vaild_score", valid_score, epoch_idx)
                            self.wandblogger.log_metrics(
                                {**valid_result, "valid_step": valid_step}, head="valid"
                            )

                            if update_flag:
                                if saved:
                                    self._save_checkpoint(epoch_idx, verbose=verbose)
                                self.best_valid_result = valid_result

                            if callback_fn:
                                callback_fn(epoch_idx, valid_score)

                            if stop_flag:
                                stop_output = "Finished training, best eval result in epoch %d" % (
                                        epoch_idx - self.cur_step * self.eval_step
                                )
                                if verbose:
                                    self.logger.info(stop_output)
                                break

                            valid_step += 1

                    self._add_hparam_to_tensorboard(self.best_valid_score)
                    return self.best_valid_score, self.best_valid_result


            else:
                super().fit(train_data, test_data, valid_data, verbose, saved, show_progress, callback_fn, src_test_neg=src_test_neg,
            src_valid_neg=src_valid_neg,
            tgt_test_neg=tgt_test_neg,
            tgt_valid_neg=tgt_valid_neg)

        self.model.set_phase('OVERLAP')
        return self.best_valid_score, self.best_valid_result

    def _valid_epoch(self, valid_data, show_progress=False,src_test_neg=None,src_valid_neg=None, tgt_test_neg=None,tgt_valid_neg=None):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(
            valid_data, load_best_model=False, show_progress=show_progress,src_test_neg=src_test_neg,src_valid_neg=src_valid_neg, tgt_test_neg=tgt_test_neg,tgt_valid_neg=tgt_valid_neg
        )

        baseline=1
        if baseline==1:
            valid_score = calculate_valid_score(valid_result, "hit_avg@10")
            return valid_score, valid_result
        else:
            valid_score = calculate_valid_score(valid_result, self.valid_metric)
            return valid_score, valid_result
    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        loss_func = loss_func
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", "pink"),
            )
            if show_progress
            else train_data
        )

        if not self.config["single_spec"] and train_data.shuffle:
            train_data.sampler.set_epoch(epoch_idx)
        scaler = amp.GradScaler(enabled=self.enable_scaler)
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            sync_loss = 0
            if not self.config["single_spec"]:
                self.set_reduce_hook()
                sync_loss = self.sync_grad_loss()
            with torch.autocast(device_type=self.device.type, enabled=self.enable_amp):
                losses = loss_func(interaction)

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = (
                    loss_tuple
                    if total_loss is None
                    else tuple(map(sum, zip(total_loss, loss_tuple)))
                )
            else:
                loss = losses
                total_loss = (
                    losses.item() if total_loss is None else total_loss + losses.item()
                )
            self._check_nan(loss)
            scaler.scale(loss + sync_loss).backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            scaler.step(self.optimizer)
            scaler.update()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )
        return total_loss

    @torch.no_grad()
    def evaluate(
            self, eval_data, load_best_model=True, model_file=None, show_progress=False, src_test_neg=None,
            src_valid_neg=None, tgt_test_neg=None, tgt_valid_neg=None,mode=None
    ):
        r"""Evaluate the model based on the eval data.
            
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            collections.OrderedDict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.load_other_parameter(checkpoint.get("other_parameter"))
            message_output = "Loading model structure and parameters from {}".format(
                checkpoint_file
            )
            self.logger.info(message_output)

        self.model.eval()
        src, tgt = eval_data
        eval_func = self._full_sort_batch_eval
        if self.config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = tgt._dataset.item_num + src._dataset.item_num

        iter_data_tgt = (
            tqdm(
                tgt,
                total=len(tgt),
                ncols=100,
                desc=set_color(f"Evaluate tgt   ", "pink"),
            )
            if show_progress
            else tgt
        )
        iter_data_src = (
            tqdm(
                src,
                total=len(src),
                ncols=100,
                desc=set_color(f"Evaluate src  ", "pink"),
            )
            if show_progress
            else src
        )
        baseline = 1
        if baseline == 1:
            num_sample_tgt = 0
            num_sample_src = 0
            all_src_test_ranks = []
            all_src_valid_ranks = []
            all_tgt_test_ranks = []
            all_tgt_valid_ranks = []
            for batch_idx, batched_data in enumerate(iter_data_src):
                num_sample_src += len(batched_data[0])
                interaction, positive_u, positive_i, src_test, src_valid = eval_func(
                    batched_data, mode='neg', test_neg=src_test_neg, valid_neg=src_valid_neg)
                src_test_ranks = torch.sum(src_test > src_test[:, :1], dim=-1) + 1
                src_valid_ranks = torch.sum(src_valid > src_valid[:, :1], dim=-1) + 1
                all_src_test_ranks.append(src_test_ranks)
                all_src_valid_ranks.append(src_valid_ranks)
            all_src_test_ranks = torch.cat(all_src_test_ranks, dim=0)
            all_src_valid_ranks = torch.cat(all_src_valid_ranks, dim=0)
            src_test_hit_1, src_test_hit_10, src_test_hit_20, src_test_hit_50, src_test_ndcg_1, src_test_ndcg_10, src_test_ndcg_20, src_test_ndcg_50 = self.cal_rank_hit_ndcg(
                all_src_test_ranks)
            src_valid_hit_1, src_valid_hit_10, src_valid_hit_20, src_valid_hit_50, src_valid_ndcg_1, src_valid_ndcg_10, src_valid_ndcg_20, src_valid_ndcg_50 = self.cal_rank_hit_ndcg(
                all_src_valid_ranks)

            for batch_idx, batched_data in enumerate(iter_data_tgt):
                num_sample_tgt += len(batched_data[0])
                interaction, positive_u, positive_i, tgt_test, tgt_valid = eval_func(batched_data, mode='neg',
                                                                                     test_neg=tgt_test_neg,
                                                                                     valid_neg=tgt_valid_neg)
                tgt_test_ranks = torch.sum(tgt_test > tgt_test[:, :1], dim=-1) + 1
                tgt_valid_ranks = torch.sum(tgt_valid > tgt_valid[:, :1], dim=-1) + 1
                all_tgt_test_ranks.append(tgt_test_ranks)
                all_tgt_valid_ranks.append(tgt_valid_ranks)
            all_tgt_test_ranks = torch.cat(all_tgt_test_ranks, dim=0)
            all_tgt_valid_ranks = torch.cat(all_tgt_valid_ranks, dim=0)
            tgt_test_hit_1, tgt_test_hit_10, tgt_test_hit_20, tgt_test_hit_50, tgt_test_ndcg_1, tgt_test_ndcg_10, tgt_test_ndcg_20, tgt_test_ndcg_50 = self.cal_rank_hit_ndcg(
                all_tgt_test_ranks)
            tgt_valid_hit_1, tgt_valid_hit_10, tgt_valid_hit_20, tgt_valid_hit_50, tgt_valid_ndcg_1, tgt_valid_ndcg_10, tgt_valid_ndcg_20, tgt_valid_ndcg_50 = self.cal_rank_hit_ndcg(
                all_tgt_valid_ranks)

            src_test_hits_1, src_test_hits_10, src_test_hits_20, src_test_hits_50, src_test_ndcgs_1, src_test_ndcgs_10, src_test_ndcgs_20, src_test_ndcgs_50 = src_test_hit_1 / num_sample_src, src_test_hit_10 / num_sample_src, src_test_hit_20 / num_sample_src, src_test_hit_50 / num_sample_src, src_test_ndcg_1 / num_sample_src, src_test_ndcg_10 / num_sample_src, src_test_ndcg_20 / num_sample_src, src_test_ndcg_50 / num_sample_src
            src_valid_hits_1, src_valid_hits_10, src_valid_hits_20, src_valid_hits_50, src_valid_ndcgs_1, src_valid_ndcgs_10, src_valid_ndcgs_20, src_valid_ndcgs_50 = src_valid_hit_1 / num_sample_src, src_valid_hit_10 / num_sample_src, src_valid_hit_20 / num_sample_src, src_valid_hit_50 / num_sample_src, src_valid_ndcg_1 / num_sample_src, src_valid_ndcg_10 / num_sample_src, src_valid_ndcg_20 / num_sample_src, src_valid_ndcg_50 / num_sample_src
            tgt_test_hits_1, tgt_test_hits_10, tgt_test_hits_20, tgt_test_hits_50, tgt_test_ndcgs_1, tgt_test_ndcgs_10, tgt_test_ndcgs_20, tgt_test_ndcgs_50 = tgt_test_hit_1 / num_sample_tgt, tgt_test_hit_10 / num_sample_tgt, tgt_test_hit_20 / num_sample_tgt, tgt_test_hit_50 / num_sample_tgt, tgt_test_ndcg_1 / num_sample_tgt, tgt_test_ndcg_10 / num_sample_tgt, tgt_test_ndcg_20 / num_sample_tgt, tgt_test_ndcg_50 / num_sample_tgt
            tgt_valid_hits_1, tgt_valid_hits_10, tgt_valid_hits_20, tgt_valid_hits_50, tgt_valid_ndcgs_1, tgt_valid_ndcgs_10, tgt_valid_ndcgs_20, tgt_valid_ndcgs_50 = tgt_valid_hit_1 / num_sample_tgt, tgt_valid_hit_10 / num_sample_tgt, tgt_valid_hit_20 / num_sample_tgt, tgt_valid_hit_50 / num_sample_tgt, tgt_valid_ndcg_1 / num_sample_tgt, tgt_valid_ndcg_10 / num_sample_tgt, tgt_valid_ndcg_20 / num_sample_tgt, tgt_valid_ndcg_50 / num_sample_tgt
            if mode == "test":
                result = {
                    "hit_avg@10": (tgt_test_hits_10 + src_test_hits_10) / 2,
                    "hit@10": tgt_test_hits_10,
                    "hit@20": tgt_test_hits_20,
                    "ndcg@10": tgt_test_ndcgs_10,
                    "ndcg@20": tgt_test_ndcgs_20,
                    "src_hit@10": src_test_hits_10,
                    "src_hit@20": src_test_hits_20,
                    "src_ndcg@10": src_test_ndcgs_10,
                    "src_ndcg@20": src_test_ndcgs_20,
                }
            else:
                result = {
                    "hit_avg@10": (tgt_valid_hits_10+src_valid_hits_10)/2,
                    "hit@10": tgt_valid_hits_10,
                    "hit@20": tgt_valid_hits_20,
                    "ndcg@10": tgt_valid_ndcgs_10,
                    "ndcg@20": tgt_valid_ndcgs_20,
                    "src_hit@10": src_valid_hits_10,
                    "src_hit@20": src_valid_hits_20,
                    "src_ndcg@10": src_valid_ndcgs_10,
                    "src_ndcg@20": src_valid_ndcgs_20,
                }
            mywandb = 0
            if mywandb == 1:
                wandb.log({f"D1-valid": {
                    "Hit@10": src_valid_hits_10, "NDCG@10": src_valid_ndcgs_10,
                    "Hit@20": src_valid_hits_20, "NDCG@20": src_valid_ndcgs_20
                }})
                wandb.log({f"D1-test": {
                    "Hit@10": src_test_hits_10, "NDCG@10": src_test_ndcgs_10,
                    "Hit@20": src_test_hits_20, "NDCG@20": src_test_ndcgs_20
                }})
                wandb.log({f"D2-valid": {
                    "Hit@10": tgt_valid_hits_10, "NDCG@10": tgt_valid_ndcgs_10,
                    "Hit@20": tgt_valid_hits_20, "NDCG@20": tgt_valid_ndcgs_20
                }})
                wandb.log({f"D2-test": {
                    "Hit@10": tgt_test_hits_10, "NDCG@10": tgt_test_ndcgs_10,
                    "Hit@20": tgt_test_hits_20, "NDCG@20": tgt_test_ndcgs_20
                }})

            return result
    def cal_rank_hit_ndcg(self, ranks):
        hit_1, hit_10, hit_20, hit_50, ndcg_1, ndcg_10, ndcg_20, ndcg_50 = 0, 0, 0, 0, 0, 0, 0, 0
        for i in range(len(ranks)):
            rank = ranks[i].item()
            if rank == 1:
                hit_1 += 1
                ndcg_1 += 1 / math.log2(rank + 1)
            if rank <= 10:
                hit_10 += 1
                ndcg_10 += 1 / math.log2(rank + 1)
            if rank <= 20:
                hit_20 += 1
                ndcg_20 += 1 / math.log2(rank + 1)
            if rank <= 50:
                hit_50 += 1
                ndcg_50 += 1 / math.log2(rank + 1)
        return hit_1, hit_10, hit_20, hit_50, ndcg_1, ndcg_10, ndcg_20, ndcg_50
    def _full_sort_batch_eval(self, batched_data,mode='neg',test_neg=None,valid_neg=None):
        interaction, history_index, positive_u, positive_i = batched_data
        try:
            if mode=='full':
                # Note: interaction without item ids
                scores = self.model.full_sort_predict(interaction.to(self.device))
                scores = scores.view(-1, self.tot_item_num)
                scores[:, 0] = -np.inf
                if history_index is not None:
                    scores[history_index] = -np.inf
                return interaction, scores, positive_u, positive_i
            else:
                test, valid = self.model.neg_sort_predict(interaction.to(self.device),test_neg=test_neg,valid_neg=valid_neg)
                return interaction, positive_u, positive_i, test, valid
        except NotImplementedError:
            inter_len = len(interaction)
            new_inter = interaction.to(self.device).repeat_interleave(self.tot_item_num)
            batch_size = len(new_inter)
            new_inter.update(self.item_tensor.repeat(inter_len))
            if batch_size <= self.test_batch_size:
                scores = self.model.predict(new_inter)
            else:
                scores = self._spilt_predict(new_inter, batch_size)

            scores = scores.view(-1, self.tot_item_num)
            scores[:, 0] = -np.inf
            if history_index is not None:
                scores[history_index] = -np.inf
            return interaction, scores, positive_u, positive_i


class DCDCSRTrainer(Trainer):
    r"""Trainer for training DCDCSR models."""

    def __init__(self, config, model):
        super(DCDCSRTrainer, self).__init__(config, model)
        self.train_modes = config['train_modes']
        self.train_epochs = config['epoch_num']
        self.split_valid_flag = config['source_split']

    def _reinit(self, phase):
        """Reset the parameters when start a new training phase.
        """
        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.item_tensor = None
        self.tot_item_num = None
        self.train_loss_dict = dict()
        self.epochs = int(self.train_epochs[phase])
        self.eval_step = min(self.config['eval_step'], self.epochs)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

            Args:
                train_data (DataLoader): the train data
                valid_data (DataLoader, optional): the valid data, default: None.
                                                    If it's None, the early_stopping is invalid.
                verbose (bool, optional): whether to write training and evaluation information to logger, default: True
                saved (bool, optional): whether to save the model parameters, default: True
                show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
                callback_fn (callable): Optional callback function executed at end of epoch.
                                        Includes (epoch_idx, valid_score) input arguments.

            Returns:
                    (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        for phase in range(len(self.train_modes)):
            self._reinit(phase)
            scheme = self.train_modes[phase]
            self.logger.info("Start training with {} mode".format(scheme))
            state = train_mode2state[scheme]
            train_data.set_mode(state)
            self.model.set_phase(scheme)
            if scheme == 'BOTH':
                super().fit(train_data, None, verbose, saved, show_progress, callback_fn)
            else:
                if self.split_valid_flag and valid_data is not None:
                    source_valid_data, target_valid_data = valid_data
                    if scheme == 'SOURCE':
                        super().fit(train_data, source_valid_data, verbose, saved, show_progress, callback_fn)
                    else:
                        super().fit(train_data, target_valid_data, verbose, saved, show_progress, callback_fn)
                else:
                    super().fit(train_data, valid_data, verbose, saved, show_progress, callback_fn)

        self.model.set_phase('OVERLAP')
        return self.best_valid_score, self.best_valid_result
