export CUDA_VISIBLE_DEVICES=1
cd 8mode

python code.py --loss_config SC --save_figs --sc_ratio 1.0 --num_steps 100 --fig_dir ./figures_100 --epochs 100
python code.py --loss_config M1+SC --save_figs --sc_ratio 0.5 --num_steps 100 --fig_dir ./figures_100
python code.py --loss_config M1 --save_figs --sc_ratio 0.0 --num_steps 100 --fig_dir ./figures_100
python code.py --loss_config M2 --save_figs --sc_ratio 0.0 --num_steps 100 --fig_dir ./figures_100
python code.py --loss_config M2+SC --save_figs --sc_ratio 1.0 --num_steps 100 --fig_dir ./figures_100
python code.py --loss_config M1+M2 --save_figs --sc_ratio 0.0 --num_steps 100 --fig_dir ./figures_100
python code.py --loss_config M1+M2+SC --save_figs --sc_ratio 0.5 --num_steps 100 --fig_dir ./figures_100

python code.py --loss_config SC --save_figs --sc_ratio 1.0 --num_steps 50 --fig_dir ./figures_50 --epochs 100
python code.py --loss_config M1+SC --save_figs --sc_ratio 0.5 --num_steps 50 --fig_dir ./figures_50
python code.py --loss_config M1 --save_figs --sc_ratio 0.0 --num_steps 50 --fig_dir ./figures_50
python code.py --loss_config M2 --save_figs --sc_ratio 0.0 --num_steps 50 --fig_dir ./figures_50
python code.py --loss_config M2+SC --save_figs --sc_ratio 1.0 --num_steps 50 --fig_dir ./figures_50
python code.py --loss_config M1+M2 --save_figs --sc_ratio 0.0 --num_steps 50 --fig_dir ./figures_50
python code.py --loss_config M1+M2+SC --save_figs --sc_ratio 0.5 --num_steps 50 --fig_dir ./figures_50

python code.py --loss_config SC --save_figs --sc_ratio 1.0 --num_steps 20 --fig_dir ./figures_20 --epochs 100
python code.py --loss_config M1+SC --save_figs --sc_ratio 0.5 --num_steps 20 --fig_dir ./figures_20
python code.py --loss_config M1 --save_figs --sc_ratio 0.0 --num_steps 20 --fig_dir ./figures_20
python code.py --loss_config M2 --save_figs --sc_ratio 0.0 --num_steps 20 --fig_dir ./figures_20
python code.py --loss_config M2+SC --save_figs --sc_ratio 1.0 --num_steps 20 --fig_dir ./figures_20
python code.py --loss_config M1+M2 --save_figs --sc_ratio 0.0 --num_steps 20 --fig_dir ./figures_20
python code.py --loss_config M1+M2+SC --save_figs --sc_ratio 0.5 --num_steps 20 --fig_dir ./figures_20

python code.py --loss_config SC --save_figs --sc_ratio 1.0 --num_steps 10 --fig_dir ./figures_10 --epochs 100
python code.py --loss_config M1+SC --save_figs --sc_ratio 0.5 --num_steps 10 --fig_dir ./figures_10
python code.py --loss_config M1 --save_figs --sc_ratio 0.0 --num_steps 10 --fig_dir ./figures_10
python code.py --loss_config M2 --save_figs --sc_ratio 0.0 --num_steps 10 --fig_dir ./figures_10
python code.py --loss_config M2+SC --save_figs --sc_ratio 1.0 --num_steps 10 --fig_dir ./figures_10
python code.py --loss_config M1+M2 --save_figs --sc_ratio 0.0 --num_steps 10 --fig_dir ./figures_10
python code.py --loss_config M1+M2+SC --save_figs --sc_ratio 0.5 --num_steps 10 --fig_dir ./figures_10

python code.py --loss_config SC --save_figs --sc_ratio 1.0 --num_steps 1 --fig_dir ./figures_1 --epochs 100
python code.py --loss_config M1+SC --save_figs --sc_ratio 0.5 --num_steps 1 --fig_dir ./figures_1
python code.py --loss_config M1 --save_figs --sc_ratio 0.0 --num_steps 1 --fig_dir ./figures_1
python code.py --loss_config M2 --save_figs --sc_ratio 0.0 --num_steps 1 --fig_dir ./figures_1
python code.py --loss_config M2+SC --save_figs --sc_ratio 1.0 --num_steps 1 --fig_dir ./figures_1
python code.py --loss_config M1+M2 --save_figs --sc_ratio 0.0 --num_steps 1 --fig_dir ./figures_1
python code.py --loss_config M1+M2+SC --save_figs --sc_ratio 0.5 --num_steps 1 --fig_dir ./figures_1