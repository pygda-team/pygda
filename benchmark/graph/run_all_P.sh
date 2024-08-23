python grade.py --source 'PROTEINS_P1' --target 'PROTEINS_P2' --nhid 128 --num_layers 5 --lr 0.001 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.5 --weight 0.005 --device 'cuda:2' --filename 'results-P.txt'
python adagcn.py --source 'PROTEINS_P1' --target 'PROTEINS_P2' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --domain_weight 0.1 --device 'cuda:2' --filename 'results-P.txt'
python udagcn.py --source 'PROTEINS_P1' --target 'PROTEINS_P2' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 400 --dropout_ratio 0.4 --device 'cuda:2' --filename 'results-P.txt'
python a2gnn.py --source 'PROTEINS_P1' --target 'PROTEINS_P2' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.005 --epochs 200 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 10 --weight 10 --device 'cuda:2' --filename 'results-P.txt'
python cwgcn.py --source 'PROTEINS_P1' --target 'PROTEINS_P2' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.0001 --epochs 200 --device 'cuda:2' --filename 'results-P.txt'
python dane.py --source 'PROTEINS_P1' --target 'PROTEINS_P2' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.001 --epochs 200 --device 'cuda:2' --filename 'results-P.txt'
python sagda.py --source 'PROTEINS_P1' --target 'PROTEINS_P2' --nhid 128 --num_layers 1 --lr 0.001 --weight_decay 0.001 --epochs 200 --adv_dim 40 --device 'cuda:2' --filename 'results-P.txt'

python grade.py --source 'PROTEINS_P2' --target 'PROTEINS_P1' --nhid 128 --num_layers 5 --lr 0.001 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.5 --weight 0.005 --device 'cuda:2' --filename 'results-P.txt'
python adagcn.py --source 'PROTEINS_P2' --target 'PROTEINS_P1' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --domain_weight 0.1 --device 'cuda:2' --filename 'results-P.txt'
python udagcn.py --source 'PROTEINS_P2' --target 'PROTEINS_P1' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 400 --dropout_ratio 0.4 --device 'cuda:2' --filename 'results-P.txt'
python a2gnn.py --source 'PROTEINS_P2' --target 'PROTEINS_P1' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.005 --epochs 200 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 10 --weight 10 --device 'cuda:2' --filename 'results-P.txt'
python cwgcn.py --source 'PROTEINS_P2' --target 'PROTEINS_P1' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.0001 --epochs 200 --device 'cuda:2' --filename 'results-P.txt'
python dane.py --source 'PROTEINS_P2' --target 'PROTEINS_P1' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.001 --epochs 200 --device 'cuda:2' --filename 'results-P.txt'
python sagda.py --source 'PROTEINS_P2' --target 'PROTEINS_P1' --nhid 128 --num_layers 1 --lr 0.001 --weight_decay 0.001 --epochs 200 --adv_dim 40 --device 'cuda:2' --filename 'results-P.txt'