import os

import argparse

from CUT.quick_start import run_ours_cdr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='OURS', help='name of models')
    parser.add_argument('--config_files', type=str, default=None, help='config files')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--emb_regloss_ratio', type=float, default=0.0)
    parser.add_argument('--conloss_ratio', type=float, default=0.0)
    parser.add_argument('--reg_weight', type=float, default=0.001)
    parser.add_argument('--global_lambda_gate', type=float, default=0.5)
    parser.add_argument('--train_batch_size', type=int, default=4096)
    parser.add_argument('--eval_step', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--graph_layer', type=str, default='han-v3')
    parser.add_argument('--user_similary_type', type=str, default='profile')
    parser.add_argument('--cross_dataset', type=str, default='Amazon-office-arts')
    parser.add_argument('--overlap', type=str, default='yes')
    parser.add_argument('--load_checkpoint', type=str, default='no')
    parser.add_argument('--check_dir', type=str, default='saved/OURS-Jan-31-2025_15-57-28.pth')

    args, _ = parser.parse_known_args()

    args.dataset_name, args.src_dataset, args.tgt_dataset = args.cross_dataset.split('-')
    
    args.user_user_file = f'dataset/{args.dataset_name}/{args.dataset_name}_{args.tgt_dataset}/similar_tags/{args.tgt_dataset}_similar_{args.user_similary_type}.txt'
    args.src_user_user_file = f'dataset/{args.dataset_name}/{args.dataset_name}_{args.tgt_dataset}/similar_tags/{args.tgt_dataset}_similar_{args.user_similary_type}.txt'

    parameter_dict = {
        'learning_rate': args.learning_rate,
        'train_batch_size': args.train_batch_size,
        'eval_step': args.eval_step,
        'graph_layer': args.graph_layer,
        'load_checkpoint': args.load_checkpoint == 'yes',
        'check_dir': args.check_dir,
        'seed': args.seed,
        'emb_regloss_ratio': args.emb_regloss_ratio,
        'conloss_ratio': args.conloss_ratio,
        'reg_weight': args.reg_weight,
        'global_lambda_gate': args.global_lambda_gate,
        'user_user_file': args.user_user_file,
        'user_similary_type': args.user_similary_type,
        'overlap': args.overlap,
        'src_dataset': args.src_dataset,
        'tgt_dataset': args.tgt_dataset,
        'dataset_name': args.dataset_name,
        'source_domain': {
            'dataset': f"{args.dataset_name}_{args.src_dataset}",
            'data_path': f"dataset/{args.dataset_name}",
        },
        'target_domain': {
            'dataset': f"{args.dataset_name}_{args.tgt_dataset}",
            'data_path': f"dataset/{args.dataset_name}",
        }
    }
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None

    run_ours_cdr(model=args.model, config_file_list=config_file_list, config_dict=parameter_dict)
