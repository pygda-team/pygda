#!/bin/sh

echo "Task BRAZIL->USA"
echo "=========="
python grade.py --source 'BRAZIL' --target 'USA' --nhid 128 --num_layers 5 --lr 0.003 --weight_decay 0.01 --epochs 300 --dropout_ratio 0.2 --weight 0.01 --filename 'results-airport.txt'
python strurw.py --source 'BRAZIL' --target 'USA' --nhid 128 --num_layers 3 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --lamb 0.6 --filename 'results-airport.txt'
python asn.py --source 'BRAZIL' --target 'USA' --nhid 128 --hid_dim_vae 128 --lr 0.001 --weight_decay 0.001 --epochs 100 --dropout_ratio 0.5 --lambda_r 0.1 --lambda_d 0.5 --lambda_f 0.0001 --filename 'results-airport.txt'
python acdne.py --source 'BRAZIL' --target 'USA' --nhid 128 --lr 0.001 --weight_decay 0.01 --epochs 200 --dropout_ratio 0.2 --pair_weight 0.0001 --filename 'results-airport.txt'
python adagcn.py --source 'BRAZIL' --target 'USA' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --domain_weight 0.1 --filename 'results-airport.txt'
python udagcn.py --source 'BRAZIL' --target 'USA' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 400 --dropout_ratio 0.4 --filename 'results-airport.txt'
python specreg.py --source 'BRAZIL' --target 'USA' --nhid 128 --num_layers 5 --lr 0.003 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.1 --gamma_adv 0.1 --gamma_smooth 0.001 --gamma_mfr 0.001 --filename 'results-airport.txt'
python a2gnn.py --source 'BRAZIL' --target 'USA' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.005 --epochs 200 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 10 --weight 10 --filename 'results-airport.txt'
python pairalign.py --source 'BRAZIL' --target 'USA' --nhid 128 --num_layers 2 --lr 0.003 --weight_decay 0.003 --epochs 200 --dropout_ratio 0.0 --rw_lmda 1 --ls_lambda 3.0 --lw_lambda 0.01 --filename 'results-airport.txt'

python kbl.py --source 'BRAZIL' --target 'USA' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.01 --epochs 500 --k_cross 20 --k_within 10 --filename 'results-airport.txt'
python cwgcn.py --source 'BRAZIL' --target 'USA' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.0001 --epochs 200 --filename 'results-airport.txt'
python dane.py --source 'BRAZIL' --target 'USA' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.001 --epochs 200 --filename 'results-airport.txt'
python dgda.py --source 'BRAZIL' --target 'USA' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --m_w 0.5 --beta 0.5 --filename 'results-airport.txt'
python dmgnn.py --source 'BRAZIL' --target 'USA' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pair_weight 0.1 --filename 'results-airport.txt'
python jhgda.py --source 'BRAZIL' --target 'USA' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pool_ratio 0.2 --filename 'results-airport.txt'
python sagda.py --source 'BRAZIL' --target 'USA' --nhid 128 --num_layers 1 --lr 0.001 --weight_decay 0.001 --epochs 200 --adv_dim 40 --filename 'results-airport.txt'

echo "Task BRAZIL->EUROPE"
echo "=========="
python grade.py --source 'BRAZIL' --target 'EUROPE' --nhid 128 --num_layers 4 --lr 0.001 --weight_decay 0.01 --epochs 200 --dropout_ratio 0.2 --weight 0.01 --filename 'results-airport.txt'
python strurw.py --source 'BRAZIL' --target 'EUROPE' --nhid 128 --num_layers 3 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --lamb 0.6 --filename 'results-airport.txt'
python asn.py --source 'BRAZIL' --target 'EUROPE' --nhid 128 --hid_dim_vae 128 --lr 0.001 --weight_decay 0.001 --epochs 100 --dropout_ratio 0.5 --lambda_r 0.1 --lambda_d 0.5 --lambda_f 0.0001 --filename 'results-airport.txt'
python acdne.py --source 'BRAZIL' --target 'EUROPE' --nhid 128 --lr 0.001 --weight_decay 0.01 --epochs 300 --dropout_ratio 0.2 --pair_weight 0.0001 --filename 'results-airport.txt'
python adagcn.py --source 'BRAZIL' --target 'EUROPE' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --domain_weight 0.1 --filename 'results-airport.txt'
python udagcn.py --source 'BRAZIL' --target 'EUROPE' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 400 --dropout_ratio 0.4 --filename 'results-airport.txt'
python specreg.py --source 'BRAZIL' --target 'EUROPE' --nhid 128 --num_layers 5 --lr 0.003 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.1 --gamma_adv 0.1 --gamma_smooth 0.001 --gamma_mfr 0.001 --filename 'results-airport.txt'
python a2gnn.py --source 'BRAZIL' --target 'EUROPE' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.005 --epochs 200 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 10 --weight 10 --filename 'results-airport.txt'
python pairalign.py --source 'BRAZIL' --target 'EUROPE' --nhid 128 --num_layers 2 --lr 0.003 --weight_decay 0.003 --epochs 200 --dropout_ratio 0.0 --rw_lmda 1 --ls_lambda 3.0 --lw_lambda 0.01 --filename 'results-airport.txt'

python kbl.py --source 'BRAZIL' --target 'EUROPE' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.01 --epochs 500 --k_cross 20 --k_within 10 --filename 'results-airport.txt'
python cwgcn.py --source 'BRAZIL' --target 'EUROPE' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.0001 --epochs 200 --filename 'results-airport.txt'
python dane.py --source 'BRAZIL' --target 'EUROPE' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.001 --epochs 200 --filename 'results-airport.txt'
python dgda.py --source 'BRAZIL' --target 'EUROPE' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --m_w 0.5 --beta 0.5 --filename 'results-airport.txt'
python dmgnn.py --source 'BRAZIL' --target 'EUROPE' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pair_weight 0.1 --filename 'results-airport.txt'
python jhgda.py --source 'BRAZIL' --target 'EUROPE' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pool_ratio 0.2 --filename 'results-airport.txt'
python sagda.py --source 'BRAZIL' --target 'EUROPE' --nhid 128 --num_layers 1 --lr 0.001 --weight_decay 0.001 --epochs 200 --adv_dim 40 --filename 'results-airport.txt'

echo "Task USA->BRAZIL"
echo "=========="
python grade.py --source 'USA' --target 'BRAZIL' --nhid 128 --num_layers 4 --lr 0.001 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.2 --weight 0.02 --filename 'results-airport.txt'
python strurw.py --source 'USA' --target 'BRAZIL' --nhid 128 --num_layers 3 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --lamb 0.6 --filename 'results-airport.txt'
python asn.py --source 'USA' --target 'BRAZIL' --nhid 128 --hid_dim_vae 128 --lr 0.001 --weight_decay 0.001 --epochs 100 --dropout_ratio 0.5 --lambda_r 0.1 --lambda_d 0.5 --lambda_f 0.0001 --filename 'results-airport.txt'
python acdne.py --source 'USA' --target 'BRAZIL' --nhid 128 --lr 0.001 --weight_decay 0.01 --epochs 300 --dropout_ratio 0.2 --pair_weight 0.0001 --filename 'results-airport.txt'
python adagcn.py --source 'USA' --target 'BRAZIL' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --domain_weight 0.1 --filename 'results-airport.txt'
python udagcn.py --source 'USA' --target 'BRAZIL' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 400 --dropout_ratio 0.4 --filename 'results-airport.txt' 
python specreg.py --source 'USA' --target 'BRAZIL' --nhid 128 --num_layers 5 --lr 0.003 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.1 --gamma_adv 0.1 --gamma_smooth 0.001 --gamma_mfr 0.001 --filename 'results-airport.txt'
python a2gnn.py --source 'USA' --target 'BRAZIL' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.005 --epochs 200 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 10 --weight 10 --filename 'results-airport.txt'
python pairalign.py --source 'USA' --target 'BRAZIL' --nhid 128 --num_layers 2 --lr 0.003 --weight_decay 0.003 --epochs 200 --dropout_ratio 0.0 --rw_lmda 1 --ls_lambda 3.0 --lw_lambda 0.01 --filename 'results-airport.txt'

python kbl.py --source 'USA' --target 'BRAZIL' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.01 --epochs 500 --k_cross 20 --k_within 10 --filename 'results-airport.txt'
python cwgcn.py --source 'USA' --target 'BRAZIL' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.0001 --epochs 200 --filename 'results-airport.txt'
python dane.py --source 'USA' --target 'BRAZIL' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.001 --epochs 200 --filename 'results-airport.txt'
python dgda.py --source 'USA' --target 'BRAZIL' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --m_w 0.5 --beta 0.5 --filename 'results-airport.txt'
python dmgnn.py --source 'USA' --target 'BRAZIL' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pair_weight 0.1 --filename 'results-airport.txt'
python jhgda.py --source 'USA' --target 'BRAZIL' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pool_ratio 0.2 --filename 'results-airport.txt'
python sagda.py --source 'USA' --target 'BRAZIL' --nhid 128 --num_layers 1 --lr 0.001 --weight_decay 0.001 --epochs 200 --adv_dim 40 --filename 'results-airport.txt'

echo "Task USA->EUROPE"
echo "=========="
python grade.py --source 'USA' --target 'EUROPE' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.2 --weight 0.002 --filename 'results-airport.txt'
python strurw.py --source 'USA' --target 'EUROPE' --nhid 128 --num_layers 3 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --lamb 0.6 --filename 'results-airport.txt'
python asn.py --source 'USA' --target 'EUROPE' --nhid 128 --hid_dim_vae 128 --lr 0.001 --weight_decay 0.001 --epochs 100 --dropout_ratio 0.5 --lambda_r 0.1 --lambda_d 0.5 --lambda_f 0.0001 --filename 'results-airport.txt'
python acdne.py --source 'USA' --target 'EUROPE' --nhid 128 --lr 0.001 --weight_decay 0.01 --epochs 200 --dropout_ratio 0.2 --pair_weight 0.0001 --filename 'results-airport.txt'
python adagcn.py --source 'USA' --target 'EUROPE' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --domain_weight 0.1 --filename 'results-airport.txt'
python udagcn.py --source 'USA' --target 'EUROPE' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 400 --dropout_ratio 0.4 --filename 'results-airport.txt'
python specreg.py --source 'USA' --target 'EUROPE' --nhid 128 --num_layers 5 --lr 0.003 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.1 --gamma_adv 0.1 --gamma_smooth 0.001 --gamma_mfr 0.001 --filename 'results-airport.txt'
python a2gnn.py --source 'USA' --target 'EUROPE' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.005 --epochs 200 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 10 --weight 10 --filename 'results-airport.txt'
python pairalign.py --source 'USA' --target 'EUROPE' --nhid 128 --num_layers 2 --lr 0.003 --weight_decay 0.003 --epochs 200 --dropout_ratio 0.0 --rw_lmda 1 --ls_lambda 3.0 --lw_lambda 0.01 --filename 'results-airport.txt'

python kbl.py --source 'USA' --target 'EUROPE' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.01 --epochs 500 --k_cross 20 --k_within 10 --filename 'results-airport.txt'
python cwgcn.py --source 'USA' --target 'EUROPE' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.0001 --epochs 200 --filename 'results-airport.txt'
python dane.py --source 'USA' --target 'EUROPE' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.001 --epochs 200 --filename 'results-airport.txt'
python dgda.py --source 'USA' --target 'EUROPE' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --m_w 0.5 --beta 0.5 --filename 'results-airport.txt'
python dmgnn.py --source 'USA' --target 'EUROPE' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pair_weight 0.1 --filename 'results-airport.txt'
python jhgda.py --source 'USA' --target 'EUROPE' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pool_ratio 0.2 --filename 'results-airport.txt'
python sagda.py --source 'USA' --target 'EUROPE' --nhid 128 --num_layers 1 --lr 0.001 --weight_decay 0.001 --epochs 200 --adv_dim 40 --filename 'results-airport.txt'

echo "Task EUROPE->BRAZIL"
echo "=========="
python grade.py --source 'EUROPE' --target 'BRAZIL' --nhid 128 --num_layers 4 --lr 0.001 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.2 --weight 0.02 --filename 'results-airport.txt'
python strurw.py --source 'EUROPE' --target 'BRAZIL' --nhid 128 --num_layers 3 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --lamb 0.6 --filename 'results-airport.txt'
python asn.py --source 'EUROPE' --target 'BRAZIL' --nhid 128 --hid_dim_vae 128 --lr 0.001 --weight_decay 0.001 --epochs 100 --dropout_ratio 0.5 --lambda_r 0.1 --lambda_d 0.5 --lambda_f 0.0001 --filename 'results-airport.txt'
python acdne.py --source 'EUROPE' --target 'BRAZIL' --nhid 128 --lr 0.001 --weight_decay 0.01 --epochs 300 --dropout_ratio 0.2 --pair_weight 0.0001 --filename 'results-airport.txt'
python adagcn.py --source 'EUROPE' --target 'BRAZIL' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --domain_weight 0.1 --filename 'results-airport.txt'
python udagcn.py --source 'EUROPE' --target 'BRAZIL' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 400 --dropout_ratio 0.4 --filename 'results-airport.txt'
python specreg.py --source 'EUROPE' --target 'BRAZIL' --nhid 128 --num_layers 5 --lr 0.003 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.1 --gamma_adv 0.1 --gamma_smooth 0.001 --gamma_mfr 0.001 --filename 'results-airport.txt'
python a2gnn.py --source 'EUROPE' --target 'BRAZIL' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.005 --epochs 200 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 10 --weight 10 --filename 'results-airport.txt'
python pairalign.py --source 'EUROPE' --target 'BRAZIL' --nhid 128 --num_layers 2 --lr 0.003 --weight_decay 0.003 --epochs 200 --dropout_ratio 0.0 --rw_lmda 1 --ls_lambda 3.0 --lw_lambda 0.01 --filename 'results-airport.txt'

python kbl.py --source 'EUROPE' --target 'BRAZIL' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.01 --epochs 500 --k_cross 20 --k_within 10 --filename 'results-airport.txt'
python cwgcn.py --source 'EUROPE' --target 'BRAZIL' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.0001 --epochs 200 --filename 'results-airport.txt'
python dane.py --source 'EUROPE' --target 'BRAZIL' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.001 --epochs 200 --filename 'results-airport.txt'
python dgda.py --source 'EUROPE' --target 'BRAZIL' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --m_w 0.5 --beta 0.5 --filename 'results-airport.txt'
python dmgnn.py --source 'EUROPE' --target 'BRAZIL' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pair_weight 0.1 --filename 'results-airport.txt'
python jhgda.py --source 'EUROPE' --target 'BRAZIL' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pool_ratio 0.2 --filename 'results-airport.txt'
python sagda.py --source 'EUROPE' --target 'BRAZIL' --nhid 128 --num_layers 1 --lr 0.001 --weight_decay 0.001 --epochs 200 --adv_dim 40 --filename 'results-airport.txt'

echo "Task EUROPE->USA"
echo "=========="
python grade.py --source 'EUROPE' --target 'USA' --nhid 128 --num_layers 4 --lr 0.001 --weight_decay 0.01 --epochs 200 --dropout_ratio 0.2 --weight 0.005 --filename 'results-airport.txt'
python strurw.py --source 'EUROPE' --target 'USA' --nhid 128 --num_layers 3 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --lamb 0.6 --filename 'results-airport.txt'
python asn.py --source 'EUROPE' --target 'USA' --nhid 128 --hid_dim_vae 128 --lr 0.001 --weight_decay 0.001 --epochs 100 --dropout_ratio 0.5 --lambda_r 0.1 --lambda_d 0.5 --lambda_f 0.0001 --filename 'results-airport.txt'
python acdne.py --source 'EUROPE' --target 'USA' --nhid 128 --lr 0.001 --weight_decay 0.01 --epochs 300 --dropout_ratio 0.2 --pair_weight 0.0001 --filename 'results-airport.txt'
python adagcn.py --source 'EUROPE' --target 'USA' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --domain_weight 0.1 --filename 'results-airport.txt'
python udagcn.py --source 'EUROPE' --target 'USA' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 400 --dropout_ratio 0.4 --filename 'results-airport.txt'
python specreg.py --source 'EUROPE' --target 'USA' --nhid 128 --num_layers 5 --lr 0.003 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.1 --gamma_adv 0.1 --gamma_smooth 0.001 --gamma_mfr 0.001 --filename 'results-airport.txt'
python a2gnn.py --source 'EUROPE' --target 'USA' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.005 --epochs 200 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 10 --weight 10 --filename 'results-airport.txt'
python pairalign.py --source 'EUROPE' --target 'USA' --nhid 128 --num_layers 2 --lr 0.003 --weight_decay 0.003 --epochs 200 --dropout_ratio 0.0 --rw_lmda 1 --ls_lambda 3.0 --lw_lambda 0.01 --filename 'results-airport.txt'

python kbl.py --source 'EUROPE' --target 'USA' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.01 --epochs 500 --k_cross 20 --k_within 10 --filename 'results-airport.txt'
python cwgcn.py --source 'EUROPE' --target 'USA' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.0001 --epochs 200 --filename 'results-airport.txt'
python dane.py --source 'EUROPE' --target 'USA' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.001 --epochs 200 --filename 'results-airport.txt'
python dgda.py --source 'EUROPE' --target 'USA' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --m_w 0.5 --beta 0.5 --filename 'results-airport.txt'
python dmgnn.py --source 'EUROPE' --target 'USA' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pair_weight 0.1 --filename 'results-airport.txt'
python jhgda.py --source 'EUROPE' --target 'USA' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pool_ratio 0.2 --filename 'results-airport.txt'
python sagda.py --source 'EUROPE' --target 'USA' --nhid 128 --num_layers 1 --lr 0.001 --weight_decay 0.001 --epochs 200 --adv_dim 40 --filename 'results-airport.txt'
