export CUDA_VISIBLE_DEVICES='1'
load_checkpoint='no'
cross_dataset='Amazon-Movie-Music'

for seed in 2020 2021 2022 2023 2024
do
for learning_rate in 0.001
do
for conloss_ratio in 0.0
do
for emb_regloss_ratio in 0.0
do
for user_similary_type in v1-56336
do
for graph_layer in 'han-v3'
do
for overlap in 'no'
do
python run_cut.py --graph_layer $graph_layer --load_checkpoint $load_checkpoint --learning_rate $learning_rate --seed $seed --emb_regloss_ratio $emb_regloss_ratio --conloss_ratio $conloss_ratio --user_similary_type $user_similary_type --cross_dataset $cross_dataset --overlap $overlap 
done
done
done
done
done
done
done