import os
"""
SUBCDR.quick_start
########################
"""
import logging
from logging import getLogger
import torch

from recbole.utils import init_logger, init_seed, set_color

from SUBCDR.config import CDRConfig
from SUBCDR.data import create_dataset, data_preparation
from SUBCDR.utils import get_model, get_trainer
from collections import defaultdict

def run_ours_cdr(model=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    #args = init()
    # configurations initialization
    config = CDRConfig(model=model, config_file_list=config_file_list, config_dict=config_dict)


    metric_dir = f"metric_results/{config['src_dataset']}_{config['tgt_dataset']}/{config['graph_layer']}/{config['user_similary_type']}_{config['emb_regloss_ratio']}_{config['conloss_ratio']}/{config['train_batch_size']}_{config['learning_rate']}/{config['seed']}"

    # if os.path.exists(f"{metric_dir}/test_result.json"):
    #     exit()

    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)
    
    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    tgt_test_neg = process_test_file(f'dataset/{config["dataset_name"]}/{config["dataset_name"]}_{config["tgt_dataset"]}/test.txt', dataset.target_user_ID_remap_dict,dataset.target_item_ID_remap_dict)
    tgt_valid_neg = process_test_file(f'dataset/{config["dataset_name"]}/{config["dataset_name"]}_{config["tgt_dataset"]}/valid.txt', dataset.target_user_ID_remap_dict, dataset.target_item_ID_remap_dict)
    src_test_neg = process_test_file(f'dataset/{config["dataset_name"]}/{config["dataset_name"]}_{config["src_dataset"]}/test.txt', dataset.source_user_ID_remap_dict, dataset.source_item_ID_remap_dict)
    src_valid_neg = process_test_file(f'dataset/{config["dataset_name"]}/{config["dataset_name"]}_{config["src_dataset"]}/valid.txt', dataset. source_user_ID_remap_dict, dataset.source_item_ID_remap_dict)
    # model loading and initialization
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)
    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    if config['load_checkpoint']:
        trainer.resume_checkpoint(config['check_dir'])

    if not config['load_checkpoint']:
        trainer.fit(
            train_data, test_data = test_data, valid_data = valid_data, saved=saved, show_progress=config['show_progress'],src_test_neg=src_test_neg,src_valid_neg=src_valid_neg, tgt_test_neg=tgt_test_neg,tgt_valid_neg=tgt_valid_neg
        )
        
    test_result = trainer.evaluate(valid_data, load_best_model=saved, show_progress=config['show_progress'],
                                    src_test_neg=src_test_neg,
                                    src_valid_neg=src_valid_neg, tgt_test_neg=tgt_test_neg,
                                    tgt_valid_neg=tgt_valid_neg, mode="test")
    
    logger.info(test_result)

    os.makedirs(metric_dir, exist_ok=True)
    import json
    json.dump(test_result, open(f"{metric_dir}/test_result.json", 'w'))
    return test_result

def process_test_file(file_path, user_remap_dict, item_remap_dict):
    test_dict = defaultdict(dict)

    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split()
            original_user_id = data[0]
            original_positive_item_id = data[1]
            original_negative_item_ids = data[2:]

            
            mapped_user_id = user_remap_dict[original_user_id]
            mapped_positive_item_id = item_remap_dict[original_positive_item_id]
            mapped_negative_item_ids = [item_remap_dict[item_id] for item_id in original_negative_item_ids]

            
            test_dict[mapped_user_id] = {
                "positive": mapped_positive_item_id,
                "negative": mapped_negative_item_ids
            }

    return test_dict

def set_device(gpu_id):
    if gpu_id == -1:
        return torch.device('cpu')
    else:
        return torch.device(
            'cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')

def objective_function(config_dict=None, config_file_list=None, saved=True):
    r""" The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = CDRConfig(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config['seed'], config['reproducibility'])
    model_name = config["model"]
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    return {
        "model": model_name,
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    checkpoint = torch.load(model_file)
    config = checkpoint['config']
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))

    return config, model, dataset, train_data, valid_data, test_data
