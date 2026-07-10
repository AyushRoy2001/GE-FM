export CUDA_VISIBLE_DEVICES=2
cd 4mode

python code.py --loss_config M1 --save_figs --sc_ratio 0.0 --num_steps 100 --fig_dir ./figures_100_our_2 --christoffel

python code.py --loss_config M1 --save_figs --sc_ratio 0.0 --num_steps 50 --fig_dir ./figures_50_our_2 --christoffel

python code.py --loss_config M1 --save_figs --sc_ratio 0.0 --num_steps 20 --fig_dir ./figures_20_our_2 --christoffel

python code.py --loss_config M1 --save_figs --sc_ratio 0.0 --num_steps 10 --fig_dir ./figures_10_our_2 --christoffel

python code.py --loss_config M1 --save_figs --sc_ratio 0.0 --num_steps 1 --fig_dir ./figures_1_our_2 --christoffel