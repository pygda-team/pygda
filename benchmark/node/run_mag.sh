#!/bin/sh

echo "Task MAG_FR->MAG_RU"
echo "=========="

python grade.py --source 'MAG_FR' --target 'MAG_RU' --nhid 300 --num_layers 5 --lr 0.001 --weight_decay 0.001 --epochs 400 --dropout_ratio 0.5 --weight 0.1 --filename 'results-mag.txt' --device 'cuda:1'
python strurw.py --source 'MAG_FR' --target 'MAG_RU' --nhid 300 --num_layers 2 --lr 0.003 --weight_decay 0.0001 --epochs 800 --dropout_ratio 0.2 --lamb 0.8 --filename 'results-mag.txt' --device 'cuda:1'
python asn.py --source 'MAG_FR' --target 'MAG_RU' --nhid 300 --hid_dim_vae 300 --lr 0.001 --weight_decay 0.0001 --epochs 200 --dropout_ratio 0.3 --lambda_r 0.1 --lambda_d 1.0 --lambda_f 0.001 --filename 'results-mag.txt' --device 'cuda:1'
python acdne.py --source 'MAG_FR' --target 'MAG_RU' --nhid 300 --lr 0.001 --weight_decay 0.0001 --epochs 200 --dropout_ratio 0.5 --pair_weight 0.1 --filename 'results-mag.txt' --device 'cuda:1'
python adagcn.py --source 'MAG_FR' --target 'MAG_RU' --nhid 300 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.4 --domain_weight 0.1 --filename 'results-mag.txt' --device 'cuda:1'
python udagcn.py --source 'MAG_FR' --target 'MAG_RU' --nhid 300 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.4 --filename 'results-mag.txt' --device 'cuda:1'
python specreg.py --source 'MAG_FR' --target 'MAG_RU' --nhid 300 --num_layers 2 --lr 0.003 --weight_decay 0.0001 --epochs 800 --dropout_ratio 0.1 --gamma_adv 0.1 --gamma_smooth 0.001 --gamma_mfr 0.001 --filename 'results-mag.txt' --device 'cuda:1'
python a2gnn.py --source 'MAG_FR' --target 'MAG_RU' --nhid 300 --num_layers 3 --lr 0.001 --weight_decay 0.0001 --epochs 800 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 2 --weight 0.0001 --filename 'results-mag.txt' --device 'cuda:1'
python pairalign.py --source 'MAG_FR' --target 'MAG_RU' --nhid 300 --num_layers 2 --lr 0.003 --weight_decay 0.003 --epochs 800 --dropout_ratio 0.0 --rw_lmda 1 --ls_lambda 3.0 --lw_lambda 0.01 --filename 'results-mag.txt' --device 'cuda:1'

python kbl.py --source 'MAG_FR' --target 'MAG_RU' --nhid 300 --num_layers 3 --lr 0.001 --weight_decay 0.0001 --epochs 800 --k_cross 20 --k_within 10 --filename 'results-mag.txt' --device 'cuda:1'
python cwgcn.py --source 'MAG_FR' --target 'MAG_RU' --nhid 300 --num_layers 2 --lr 0.0001 --weight_decay 0.0 --epochs 600 --filename 'results-mag.txt' --device 'cuda:1'
python dane.py --source 'MAG_FR' --target 'MAG_RU' --nhid 300 --num_layers 3 --lr 0.001 --weight_decay 0.001 --epochs 800 --filename 'results-mag.txt' --device 'cuda:1'
python dgda.py --source 'MAG_FR' --target 'MAG_RU' --nhid 300 --num_layers 2 --lr 0.001 --weight_decay 0.0001 --epochs 800 --m_w 0.5 --beta 0.5 --filename 'results-mag.txt' --device 'cuda:1'
python dmgnn.py --source 'MAG_FR' --target 'MAG_RU' --nhid 300 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pair_weight 0.1 --filename 'results-mag.txt' --device 'cuda:1'
python jhgda.py --source 'MAG_FR' --target 'MAG_RU' --nhid 300 --num_layers 2 --lr 0.001 --weight_decay 0.0001 --epochs 800 --pool_ratio 0.02 --filename 'results-mag.txt' --device 'cuda:1'
python sagda.py --source 'MAG_FR' --target 'MAG_RU' --nhid 300 --num_layers 1 --lr 0.001 --weight_decay 0.001 --epochs 800 --adv_dim 40 --filename 'results-mag.txt' --device 'cuda:1'

echo "Task MAG_FR->MAG_JP"
echo "=========="

python grade.py --source 'MAG_FR' --target 'MAG_JP' --nhid 300 --num_layers 5 --lr 0.001 --weight_decay 0.001 --epochs 400 --dropout_ratio 0.5 --weight 0.1 --filename 'results-mag.txt' --device 'cuda:1'
python strurw.py --source 'MAG_FR' --target 'MAG_JP' --nhid 300 --num_layers 2 --lr 0.003 --weight_decay 0.0001 --epochs 800 --dropout_ratio 0.2 --lamb 0.8 --filename 'results-mag.txt' --device 'cuda:1'
python asn.py --source 'MAG_FR' --target 'MAG_JP' --nhid 300 --hid_dim_vae 300 --lr 0.001 --weight_decay 0.0001 --epochs 200 --dropout_ratio 0.3 --lambda_r 0.1 --lambda_d 1.0 --lambda_f 0.001 --filename 'results-mag.txt' --device 'cuda:1'
python acdne.py --source 'MAG_FR' --target 'MAG_JP' --nhid 300 --lr 0.001 --weight_decay 0.0001 --epochs 200 --dropout_ratio 0.5 --pair_weight 0.1 --filename 'results-mag.txt' --device 'cuda:1'
python adagcn.py --source 'MAG_FR' --target 'MAG_JP' --nhid 300 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.4 --domain_weight 0.1 --filename 'results-mag.txt' --device 'cuda:1'
python udagcn.py --source 'MAG_FR' --target 'MAG_JP' --nhid 300 --num_layers 2 --lr 0.0001 --weight_decay 0.0001 --epochs 800 --dropout_ratio 0.4 --filename 'results-mag.txt' --device 'cuda:1'
python specreg.py --source 'MAG_FR' --target 'MAG_JP' --nhid 300 --num_layers 2 --lr 0.003 --weight_decay 0.0001 --epochs 800 --dropout_ratio 0.1 --gamma_adv 0.1 --gamma_smooth 0.001 --gamma_mfr 0.001 --filename 'results-mag.txt' --device 'cuda:1'
python a2gnn.py --source 'MAG_FR' --target 'MAG_JP' --nhid 300 --num_layers 3 --lr 0.001 --weight_decay 0.0001 --epochs 800 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 2 --weight 0.0001 --filename 'results-mag.txt' --device 'cuda:1'
python pairalign.py --source 'MAG_FR' --target 'MAG_JP' --nhid 300 --num_layers 2 --lr 0.003 --weight_decay 0.003 --epochs 800 --dropout_ratio 0.0 --rw_lmda 1 --ls_lambda 3.0 --lw_lambda 0.01 --filename 'results-mag.txt' --device 'cuda:1'

python kbl.py --source 'MAG_FR' --target 'MAG_JP' --nhid 300 --num_layers 3 --lr 0.001 --weight_decay 0.0001 --epochs 800 --k_cross 20 --k_within 10 --filename 'results-mag.txt' --device 'cuda:1'
python cwgcn.py --source 'MAG_FR' --target 'MAG_JP' --nhid 300 --num_layers 2 --lr 0.0001 --weight_decay 0.0 --epochs 600 --filename 'results-mag.txt' --device 'cuda:1'
python dane.py --source 'MAG_FR' --target 'MAG_JP' --nhid 300 --num_layers 3 --lr 0.001 --weight_decay 0.001 --epochs 800 --filename 'results-mag.txt' --device 'cuda:1'
python dgda.py --source 'MAG_FR' --target 'MAG_JP' --nhid 300 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 800 --m_w 0.5 --beta 0.5 --filename 'results-mag.txt' --device 'cuda:1'
python dmgnn.py --source 'MAG_FR' --target 'MAG_JP' --nhid 300 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pair_weight 0.1 --filename 'results-mag.txt' --device 'cuda:1'
python jhgda.py --source 'MAG_FR' --target 'MAG_JP' --nhid 300 --num_layers 2 --lr 0.001 --weight_decay 0.0001 --epochs 800 --pool_ratio 0.02 --filename 'results-mag.txt' --device 'cuda:1'
python sagda.py --source 'MAG_FR' --target 'MAG_JP' --nhid 300 --num_layers 1 --lr 0.001 --weight_decay 0.0001 --epochs 800 --adv_dim 40 --filename 'results-mag.txt' --device 'cuda:1'

echo "Task MAG_JP->MAG_RU"
echo "=========="

python grade.py --source 'MAG_JP' --target 'MAG_RU' --nhid 300 --num_layers 5 --lr 0.001 --weight_decay 0.001 --epochs 400 --dropout_ratio 0.5 --weight 0.1 --filename 'results-mag.txt' --device 'cuda:1'
python strurw.py --source 'MAG_JP' --target 'MAG_RU' --nhid 300 --num_layers 2 --lr 0.003 --weight_decay 0.0001 --epochs 800 --dropout_ratio 0.2 --lamb 0.8 --filename 'results-mag.txt' --device 'cuda:1'
python asn.py --source 'MAG_JP' --target 'MAG_RU' --nhid 300 --hid_dim_vae 300 --lr 0.001 --weight_decay 0.0001 --epochs 200 --dropout_ratio 0.3 --lambda_r 0.1 --lambda_d 1.0 --lambda_f 0.001 --filename 'results-mag.txt' --device 'cuda:1'
python acdne.py --source 'MAG_JP' --target 'MAG_RU' --nhid 300 --lr 0.001 --weight_decay 0.0001 --epochs 200 --dropout_ratio 0.5 --pair_weight 0.1 --filename 'results-mag.txt' --device 'cuda:1'
python adagcn.py --source 'MAG_JP' --target 'MAG_RU' --nhid 300 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.4 --domain_weight 0.1 --filename 'results-mag.txt' --device 'cuda:1'
python udagcn.py --source 'MAG_JP' --target 'MAG_RU' --nhid 300 --num_layers 2 --lr 0.0001 --weight_decay 0.0001 --epochs 800 --dropout_ratio 0.4 --filename 'results-mag.txt' --device 'cuda:1'
python specreg.py --source 'MAG_JP' --target 'MAG_RU' --nhid 300 --num_layers 2 --lr 0.003 --weight_decay 0.0001 --epochs 800 --dropout_ratio 0.1 --gamma_adv 0.1 --gamma_smooth 0.001 --gamma_mfr 0.001 --filename 'results-mag.txt' --device 'cuda:1'
python a2gnn.py --source 'MAG_JP' --target 'MAG_RU' --nhid 300 --num_layers 3 --lr 0.001 --weight_decay 0.0001 --epochs 800 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 2 --weight 0.0001 --filename 'results-mag.txt' --device 'cuda:1'
python pairalign.py --source 'MAG_JP' --target 'MAG_RU' --nhid 300 --num_layers 2 --lr 0.003 --weight_decay 0.003 --epochs 800 --dropout_ratio 0.0 --rw_lmda 1 --ls_lambda 3.0 --lw_lambda 0.01 --filename 'results-mag.txt' --device 'cuda:1'

python kbl.py --source 'MAG_JP' --target 'MAG_RU' --nhid 300 --num_layers 3 --lr 0.001 --weight_decay 0.0001 --epochs 800 --k_cross 20 --k_within 10 --filename 'results-mag.txt' --device 'cuda:1'
python cwgcn.py --source 'MAG_JP' --target 'MAG_RU' --nhid 300 --num_layers 2 --lr 0.0001 --weight_decay 0.0 --epochs 800 --filename 'results-mag.txt' --device 'cuda:1'
python dane.py --source 'MAG_JP' --target 'MAG_RU' --nhid 300 --num_layers 3 --lr 0.001 --weight_decay 0.001 --epochs 800 --filename 'results-mag.txt' --device 'cuda:1'
python dgda.py --source 'MAG_JP' --target 'MAG_RU' --nhid 300 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 800 --m_w 0.5 --beta 0.5 --filename 'results-mag.txt' --device 'cuda:1'
python dmgnn.py --source 'MAG_JP' --target 'MAG_RU' --nhid 300 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pair_weight 0.1 --filename 'results-mag.txt' --device 'cuda:1'
python jhgda.py --source 'MAG_JP' --target 'MAG_RU' --nhid 300 --num_layers 2 --lr 0.001 --weight_decay 0.0001 --epochs 800 --pool_ratio 0.02 --filename 'results-mag.txt' --device 'cuda:1'
python sagda.py --source 'MAG_JP' --target 'MAG_RU' --nhid 300 --num_layers 1 --lr 0.001 --weight_decay 0.0001 --epochs 300 --adv_dim 40 --filename 'results-mag.txt' --device 'cuda:1'

echo "Task MAG_JP->MAG_FR"
echo "=========="

python grade.py --source 'MAG_JP' --target 'MAG_FR' --nhid 300 --num_layers 5 --lr 0.001 --weight_decay 0.001 --epochs 400 --dropout_ratio 0.5 --weight 0.1 --filename 'results-mag.txt' --device 'cuda:1'
python strurw.py --source 'MAG_JP' --target 'MAG_FR' --nhid 300 --num_layers 2 --lr 0.003 --weight_decay 0.0001 --epochs 800 --dropout_ratio 0.2 --lamb 0.8 --filename 'results-mag.txt' --device 'cuda:1'
python asn.py --source 'MAG_JP' --target 'MAG_FR' --nhid 300 --hid_dim_vae 300 --lr 0.001 --weight_decay 0.0001 --epochs 200 --dropout_ratio 0.3 --lambda_r 0.1 --lambda_d 1.0 --lambda_f 0.001 --filename 'results-mag.txt' --device 'cuda:1'
python acdne.py --source 'MAG_JP' --target 'MAG_FR' --nhid 300 --lr 0.001 --weight_decay 0.0001 --epochs 200 --dropout_ratio 0.5 --pair_weight 0.1 --filename 'results-mag.txt' --device 'cuda:1'
python adagcn.py --source 'MAG_JP' --target 'MAG_FR' --nhid 300 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.4 --domain_weight 0.1 --filename 'results-mag.txt' --device 'cuda:1'
python udagcn.py --source 'MAG_JP' --target 'MAG_FR' --nhid 300 --num_layers 2 --lr 0.0001 --weight_decay 0.0001 --epochs 800 --dropout_ratio 0.4 --filename 'results-mag.txt' --device 'cuda:1'
python specreg.py --source 'MAG_JP' --target 'MAG_FR' --nhid 300 --num_layers 2 --lr 0.003 --weight_decay 0.0001 --epochs 800 --dropout_ratio 0.1 --gamma_adv 0.1 --gamma_smooth 0.001 --gamma_mfr 0.001 --filename 'results-mag.txt' --device 'cuda:1'
python a2gnn.py --source 'MAG_JP' --target 'MAG_FR' --nhid 300 --num_layers 3 --lr 0.001 --weight_decay 0.0001 --epochs 800 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 2 --weight 0.0001 --filename 'results-mag.txt' --device 'cuda:1'
python pairalign.py --source 'MAG_JP' --target 'MAG_FR' --nhid 300 --num_layers 2 --lr 0.003 --weight_decay 0.003 --epochs 800 --dropout_ratio 0.0 --rw_lmda 1 --ls_lambda 3.0 --lw_lambda 0.01 --filename 'results-mag.txt' --device 'cuda:1'

python kbl.py --source 'MAG_JP' --target 'MAG_FR' --nhid 300 --num_layers 3 --lr 0.001 --weight_decay 0.01 --epochs 800 --k_cross 20 --k_within 10 --filename 'results-mag.txt' --device 'cuda:1'
python cwgcn.py --source 'MAG_JP' --target 'MAG_FR' --nhid 300 --num_layers 2 --lr 0.0001 --weight_decay 0.0 --epochs 800 --filename 'results-mag.txt' --device 'cuda:1'
python dane.py --source 'MAG_JP' --target 'MAG_FR' --nhid 300 --num_layers 3 --lr 0.001 --weight_decay 0.001 --epochs 800 --filename 'results-mag.txt' --device 'cuda:1'
python dgda.py --source 'MAG_JP' --target 'MAG_FR' --nhid 300 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 800 --m_w 0.5 --beta 0.5 --filename 'results-mag.txt' --device 'cuda:1'
python dmgnn.py --source 'MAG_JP' --target 'MAG_FR' --nhid 300 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pair_weight 0.1 --filename 'results-mag.txt' --device 'cuda:1'
python jhgda.py --source 'MAG_JP' --target 'MAG_FR' --nhid 300 --num_layers 2 --lr 0.001 --weight_decay 0.0001 --epochs 800 --pool_ratio 0.02 --filename 'results-mag.txt' --device 'cuda:1'
python sagda.py --source 'MAG_JP' --target 'MAG_FR' --nhid 300 --num_layers 1 --lr 0.001 --weight_decay 0.0001 --epochs 300 --adv_dim 40 --filename 'results-mag.txt' --device 'cuda:1'
