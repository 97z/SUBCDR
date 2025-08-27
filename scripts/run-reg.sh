export CUDA_VISIBLE_DEVICES='3'
load_checkpoint='no'
cross_dataset='Amazon-Movie-Music'

for seed in 2024
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
for reg_weight in 0.005 0.01 0.0001 0.0005
do
python run_cut.py --graph_layer $graph_layer --load_checkpoint $load_checkpoint --learning_rate $learning_rate --seed $seed --emb_regloss_ratio $emb_regloss_ratio --conloss_ratio $conloss_ratio --user_similary_type $user_similary_type --cross_dataset $cross_dataset --overlap $overlap --reg_weight $reg_weight
done
done
done
done
done
done
done
done