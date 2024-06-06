#!/bin/sh

echo "Task DE->RU"
echo "=========="

python grade.py --source 'DE' --target 'RU' --nhid 64 --num_layers 1 --lr 0.0003 --weight_decay 0.001 --epochs 1000 --dropout_ratio 0.2 --weight 0.02 --filename 'results-twitch.txt'
python strurw.py --source 'DE' --target 'RU' --nhid 64 --num_layers 3 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --lamb 0.6 --filename 'results-twitch.txt'
python asn.py --source 'DE' --target 'RU' --nhid 64 --hid_dim_vae 64 --lr 0.001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.5 --lambda_r 0.1 --lambda_d 0.5 --lambda_f 0.0001 --filename 'results-twitch.txt'
python acdne.py --source 'DE' --target 'RU' --nhid 64 --lr 0.001 --weight_decay 0.01 --epochs 200 --dropout_ratio 0.2 --pair_weight 0.0001 --filename 'results-twitch.txt'
python adagcn.py --source 'DE' --target 'RU' --nhid 64 --num_layers 2 --lr 0.01 --weight_decay 0.01 --epochs 800 --dropout_ratio 0.4 --domain_weight 0.1 --filename 'results-twitch.txt'
python udagcn.py --source 'DE' --target 'RU' --nhid 64 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.4 --filename 'results-twitch.txt'
python specreg.py --source 'DE' --target 'RU' --nhid 64 --num_layers 5 --lr 0.003 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.1 --gamma_adv 0.1 --gamma_smooth 0.001 --gamma_mfr 0.001 --filename 'results-twitch.txt'
python a2gnn.py --source 'DE' --target 'RU' --nhid 64 --num_layers 2 --lr 0.01 --weight_decay 0.005 --epochs 800 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 10 --weight 10 --filename 'results-twitch.txt'
python pairalign.py --source 'DE' --target 'RU' --nhid 64 --num_layers 2 --lr 0.003 --weight_decay 0.003 --epochs 800 --dropout_ratio 0.0 --rw_lmda 1 --ls_lambda 3.0 --lw_lambda 0.01 --filename 'results-twitch.txt'

python kbl.py --source 'DE' --target 'RU' --nhid 64 --num_layers 3 --lr 0.001 --weight_decay 0.01 --epochs 500 --k_cross 20 --k_within 10 --filename 'results-twitch.txt'
python cwgcn.py --source 'DE' --target 'RU' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.0001 --epochs 800 --filename 'results-twitch.txt'
python dane.py --source 'DE' --target 'RU' --nhid 64 --num_layers 3 --lr 0.001 --weight_decay 0.001 --epochs 800 --filename 'results-twitch.txt'
python dgda.py --source 'DE' --target 'RU' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 800 --m_w 0.5 --beta 0.5 --filename 'results-twitch.txt'
python dmgnn.py --source 'DE' --target 'RU' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pair_weight 0.1 --filename 'results-twitch.txt'
python jhgda.py --source 'DE' --target 'RU' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 800 --pool_ratio 0.2 --filename 'results-twitch.txt'
python sagda.py --source 'DE' --target 'RU' --nhid 64 --num_layers 1 --lr 0.001 --weight_decay 0.001 --epochs 800 --adv_dim 40 --filename 'results-twitch.txt'

echo "Task DE->PT"
echo "=========="

python grade.py --source 'DE' --target 'PT' --nhid 64 --num_layers 1 --lr 0.0003 --weight_decay 0.001 --epochs 1000 --dropout_ratio 0.2 --weight 0.02 --filename 'results-twitch.txt'
python strurw.py --source 'DE' --target 'PT' --nhid 64 --num_layers 3 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --lamb 0.6 --filename 'results-twitch.txt'
python asn.py --source 'DE' --target 'PT' --nhid 64 --hid_dim_vae 64 --lr 0.001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.5 --lambda_r 0.1 --lambda_d 0.5 --lambda_f 0.0001 --filename 'results-twitch.txt'
python acdne.py --source 'DE' --target 'PT' --nhid 64 --lr 0.001 --weight_decay 0.01 --epochs 200 --dropout_ratio 0.2 --pair_weight 0.0001 --filename 'results-twitch.txt'
python adagcn.py --source 'DE' --target 'PT' --nhid 64 --num_layers 2 --lr 0.01 --weight_decay 0.01 --epochs 800 --dropout_ratio 0.4 --domain_weight 0.1 --filename 'results-twitch.txt'
python udagcn.py --source 'DE' --target 'PT' --nhid 64 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.4 --filename 'results-twitch.txt'
python specreg.py --source 'DE' --target 'PT' --nhid 64 --num_layers 5 --lr 0.003 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.1 --gamma_adv 0.1 --gamma_smooth 0.001 --gamma_mfr 0.001 --filename 'results-twitch.txt'
python a2gnn.py --source 'DE' --target 'PT' --nhid 64 --num_layers 2 --lr 0.01 --weight_decay 0.005 --epochs 800 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 10 --weight 10 --filename 'results-twitch.txt'
python pairalign.py --source 'DE' --target 'PT' --nhid 64 --num_layers 2 --lr 0.003 --weight_decay 0.003 --epochs 800 --dropout_ratio 0.0 --rw_lmda 1 --ls_lambda 3.0 --lw_lambda 0.01 --filename 'results-twitch.txt'

python kbl.py --source 'DE' --target 'PT' --nhid 64 --num_layers 3 --lr 0.001 --weight_decay 0.01 --epochs 500 --k_cross 20 --k_within 10 --filename 'results-twitch.txt'
python cwgcn.py --source 'DE' --target 'PT' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.0001 --epochs 800 --filename 'results-twitch.txt'
python dane.py --source 'DE' --target 'PT' --nhid 64 --num_layers 3 --lr 0.001 --weight_decay 0.001 --epochs 800 --filename 'results-twitch.txt'
python dgda.py --source 'DE' --target 'PT' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 800 --m_w 0.5 --beta 0.5 --filename 'results-twitch.txt'
python dmgnn.py --source 'DE' --target 'PT' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pair_weight 0.1 --filename 'results-twitch.txt'
python jhgda.py --source 'DE' --target 'PT' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 800 --pool_ratio 0.2 --filename 'results-twitch.txt'
python sagda.py --source 'DE' --target 'PT' --nhid 64 --num_layers 1 --lr 0.001 --weight_decay 0.001 --epochs 800 --adv_dim 40 --filename 'results-twitch.txt'

echo "Task DE->FR"
echo "=========="

python grade.py --source 'DE' --target 'FR' --nhid 64 --num_layers 1 --lr 0.0003 --weight_decay 0.001 --epochs 1000 --dropout_ratio 0.2 --weight 0.02 --filename 'results-twitch.txt'
python strurw.py --source 'DE' --target 'FR' --nhid 64 --num_layers 3 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --lamb 0.6 --filename 'results-twitch.txt'
python asn.py --source 'DE' --target 'FR' --nhid 64 --hid_dim_vae 64 --lr 0.001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.5 --lambda_r 0.1 --lambda_d 0.5 --lambda_f 0.0001 --filename 'results-twitch.txt'
python acdne.py --source 'DE' --target 'FR' --nhid 64 --lr 0.001 --weight_decay 0.01 --epochs 200 --dropout_ratio 0.2 --pair_weight 0.0001 --filename 'results-twitch.txt'
python adagcn.py --source 'DE' --target 'FR' --nhid 64 --num_layers 2 --lr 0.01 --weight_decay 0.01 --epochs 800 --dropout_ratio 0.4 --domain_weight 0.1 --filename 'results-twitch.txt'
python udagcn.py --source 'DE' --target 'FR' --nhid 64 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.4 --filename 'results-twitch.txt'
python specreg.py --source 'DE' --target 'FR' --nhid 64 --num_layers 5 --lr 0.003 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.1 --gamma_adv 0.1 --gamma_smooth 0.001 --gamma_mfr 0.001 --filename 'results-twitch.txt'
python a2gnn.py --source 'DE' --target 'FR' --nhid 64 --num_layers 2 --lr 0.01 --weight_decay 0.005 --epochs 800 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 10 --weight 10 --filename 'results-twitch.txt'
python pairalign.py --source 'DE' --target 'FR' --nhid 64 --num_layers 2 --lr 0.003 --weight_decay 0.003 --epochs 800 --dropout_ratio 0.0 --rw_lmda 1 --ls_lambda 3.0 --lw_lambda 0.01 --filename 'results-twitch.txt'

python kbl.py --source 'DE' --target 'FR' --nhid 64 --num_layers 3 --lr 0.001 --weight_decay 0.01 --epochs 500 --k_cross 20 --k_within 10 --filename 'results-twitch.txt'
python cwgcn.py --source 'DE' --target 'FR' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.0001 --epochs 800 --filename 'results-twitch.txt'
python dane.py --source 'DE' --target 'FR' --nhid 64 --num_layers 3 --lr 0.001 --weight_decay 0.001 --epochs 800 --filename 'results-twitch.txt'
python dgda.py --source 'DE' --target 'FR' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 800 --m_w 0.5 --beta 0.5 --filename 'results-twitch.txt'
python dmgnn.py --source 'DE' --target 'FR' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pair_weight 0.1 --filename 'results-twitch.txt'
python jhgda.py --source 'DE' --target 'FR' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 800 --pool_ratio 0.2 --filename 'results-twitch.txt'
python sagda.py --source 'DE' --target 'FR' --nhid 64 --num_layers 1 --lr 0.001 --weight_decay 0.001 --epochs 800 --adv_dim 40 --filename 'results-twitch.txt'

echo "Task DE->ES"
echo "=========="

python grade.py --source 'DE' --target 'ES' --nhid 64 --num_layers 1 --lr 0.0003 --weight_decay 0.001 --epochs 1000 --dropout_ratio 0.2 --weight 0.02 --filename 'results-twitch.txt'
python strurw.py --source 'DE' --target 'ES' --nhid 64 --num_layers 3 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --lamb 0.6 --filename 'results-twitch.txt'
python asn.py --source 'DE' --target 'ES' --nhid 64 --hid_dim_vae 64 --lr 0.001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.5 --lambda_r 0.1 --lambda_d 0.5 --lambda_f 0.0001 --filename 'results-twitch.txt'
python acdne.py --source 'DE' --target 'ES' --nhid 64 --lr 0.001 --weight_decay 0.01 --epochs 200 --dropout_ratio 0.2 --pair_weight 0.0001 --filename 'results-twitch.txt'
python adagcn.py --source 'DE' --target 'ES' --nhid 64 --num_layers 2 --lr 0.01 --weight_decay 0.01 --epochs 800 --dropout_ratio 0.4 --domain_weight 0.1 --filename 'results-twitch.txt'
python udagcn.py --source 'DE' --target 'ES' --nhid 64 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.4 --filename 'results-twitch.txt'
python specreg.py --source 'DE' --target 'ES' --nhid 64 --num_layers 5 --lr 0.003 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.1 --gamma_adv 0.1 --gamma_smooth 0.001 --gamma_mfr 0.001 --filename 'results-twitch.txt'
python a2gnn.py --source 'DE' --target 'ES' --nhid 64 --num_layers 2 --lr 0.01 --weight_decay 0.005 --epochs 800 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 10 --weight 10 --filename 'results-twitch.txt'
python pairalign.py --source 'DE' --target 'ES' --nhid 64 --num_layers 2 --lr 0.003 --weight_decay 0.003 --epochs 800 --dropout_ratio 0.0 --rw_lmda 1 --ls_lambda 3.0 --lw_lambda 0.01 --filename 'results-twitch.txt'

python kbl.py --source 'DE' --target 'ES' --nhid 64 --num_layers 3 --lr 0.001 --weight_decay 0.01 --epochs 500 --k_cross 20 --k_within 10 --filename 'results-twitch.txt'
python cwgcn.py --source 'DE' --target 'ES' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.0001 --epochs 800 --filename 'results-twitch.txt'
python dane.py --source 'DE' --target 'ES' --nhid 64 --num_layers 3 --lr 0.001 --weight_decay 0.001 --epochs 800 --filename 'results-twitch.txt'
python dgda.py --source 'DE' --target 'ES' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 800 --m_w 0.5 --beta 0.5 --filename 'results-twitch.txt'
python dmgnn.py --source 'DE' --target 'ES' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pair_weight 0.1 --filename 'results-twitch.txt'
python jhgda.py --source 'DE' --target 'ES' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 800 --pool_ratio 0.2 --filename 'results-twitch.txt'
python sagda.py --source 'DE' --target 'ES' --nhid 64 --num_layers 1 --lr 0.001 --weight_decay 0.001 --epochs 800 --adv_dim 40 --filename 'results-twitch.txt'

echo "Task DE->EN"
echo "=========="

python grade.py --source 'DE' --target 'EN' --nhid 64 --num_layers 1 --lr 0.0003 --weight_decay 0.001 --epochs 1000 --dropout_ratio 0.2 --weight 0.02 --filename 'results-twitch.txt'
python strurw.py --source 'DE' --target 'EN' --nhid 64 --num_layers 3 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --lamb 0.6 --filename 'results-twitch.txt'
python asn.py --source 'DE' --target 'EN' --nhid 64 --hid_dim_vae 64 --lr 0.001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.5 --lambda_r 0.1 --lambda_d 0.5 --lambda_f 0.0001 --filename 'results-twitch.txt'
python acdne.py --source 'DE' --target 'EN' --nhid 64 --lr 0.001 --weight_decay 0.01 --epochs 200 --dropout_ratio 0.2 --pair_weight 0.0001 --filename 'results-twitch.txt'
python adagcn.py --source 'DE' --target 'EN' --nhid 64 --num_layers 2 --lr 0.01 --weight_decay 0.01 --epochs 800 --dropout_ratio 0.4 --domain_weight 0.1 --filename 'results-twitch.txt'
python udagcn.py --source 'DE' --target 'EN' --nhid 64 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.4 --filename 'results-twitch.txt'
python specreg.py --source 'DE' --target 'EN' --nhid 64 --num_layers 5 --lr 0.003 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.1 --gamma_adv 0.1 --gamma_smooth 0.001 --gamma_mfr 0.001 --filename 'results-twitch.txt'
python a2gnn.py --source 'DE' --target 'EN' --nhid 64 --num_layers 2 --lr 0.01 --weight_decay 0.005 --epochs 800 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 10 --weight 10 --filename 'results-twitch.txt'
python pairalign.py --source 'DE' --target 'EN' --nhid 64 --num_layers 2 --lr 0.003 --weight_decay 0.003 --epochs 800 --dropout_ratio 0.0 --rw_lmda 1 --ls_lambda 3.0 --lw_lambda 0.01 --filename 'results-twitch.txt'

python kbl.py --source 'DE' --target 'EN' --nhid 64 --num_layers 3 --lr 0.001 --weight_decay 0.01 --epochs 500 --k_cross 20 --k_within 10 --filename 'results-twitch.txt'
python cwgcn.py --source 'DE' --target 'EN' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.0001 --epochs 800 --filename 'results-twitch.txt'
python dane.py --source 'DE' --target 'EN' --nhid 64 --num_layers 3 --lr 0.001 --weight_decay 0.001 --epochs 800 --filename 'results-twitch.txt'
python dgda.py --source 'DE' --target 'EN' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 800 --m_w 0.5 --beta 0.5 --filename 'results-twitch.txt'
python dmgnn.py --source 'DE' --target 'EN' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pair_weight 0.1 --filename 'results-twitch.txt'
python jhgda.py --source 'DE' --target 'EN' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 800 --pool_ratio 0.2 --filename 'results-twitch.txt'
python sagda.py --source 'DE' --target 'EN' --nhid 64 --num_layers 1 --lr 0.001 --weight_decay 0.001 --epochs 800 --adv_dim 40 --filename 'results-twitch.txt'

echo "Task EN->RU"
echo "=========="

python grade.py --source 'EN' --target 'RU' --nhid 64 --num_layers 1 --lr 0.0003 --weight_decay 0.001 --epochs 1000 --dropout_ratio 0.2 --weight 0.02 --filename 'results-twitch.txt'
python strurw.py --source 'EN' --target 'RU' --nhid 64 --num_layers 3 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --lamb 0.6 --filename 'results-twitch.txt'
python asn.py --source 'EN' --target 'RU' --nhid 64 --hid_dim_vae 64 --lr 0.001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.5 --lambda_r 0.1 --lambda_d 0.5 --lambda_f 0.0001 --filename 'results-twitch.txt'
python acdne.py --source 'EN' --target 'RU' --nhid 64 --lr 0.001 --weight_decay 0.01 --epochs 200 --dropout_ratio 0.2 --pair_weight 0.0001 --filename 'results-twitch.txt'
python adagcn.py --source 'EN' --target 'RU' --nhid 64 --num_layers 2 --lr 0.01 --weight_decay 0.01 --epochs 800 --dropout_ratio 0.4 --domain_weight 0.1 --filename 'results-twitch.txt'
python udagcn.py --source 'EN' --target 'RU' --nhid 64 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.4 --filename 'results-twitch.txt'
python specreg.py --source 'EN' --target 'RU' --nhid 64 --num_layers 5 --lr 0.003 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.1 --gamma_adv 0.1 --gamma_smooth 0.001 --gamma_mfr 0.001 --filename 'results-twitch.txt'
python a2gnn.py --source 'EN' --target 'RU' --nhid 64 --num_layers 2 --lr 0.01 --weight_decay 0.005 --epochs 800 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 10 --weight 10 --filename 'results-twitch.txt'
python pairalign.py --source 'EN' --target 'RU' --nhid 64 --num_layers 2 --lr 0.003 --weight_decay 0.003 --epochs 800 --dropout_ratio 0.0 --rw_lmda 1 --ls_lambda 3.0 --lw_lambda 0.01 --filename 'results-twitch.txt'

python kbl.py --source 'EN' --target 'RU' --nhid 64 --num_layers 3 --lr 0.001 --weight_decay 0.01 --epochs 500 --k_cross 20 --k_within 10 --filename 'results-twitch.txt'
python cwgcn.py --source 'EN' --target 'RU' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.0001 --epochs 800 --filename 'results-twitch.txt'
python dane.py --source 'EN' --target 'RU' --nhid 64 --num_layers 3 --lr 0.001 --weight_decay 0.001 --epochs 800 --filename 'results-twitch.txt'
python dgda.py --source 'EN' --target 'RU' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 800 --m_w 0.5 --beta 0.5 --filename 'results-twitch.txt'
python dmgnn.py --source 'EN' --target 'RU' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pair_weight 0.1 --filename 'results-twitch.txt'
python jhgda.py --source 'EN' --target 'RU' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 800 --pool_ratio 0.2 --filename 'results-twitch.txt'
python sagda.py --source 'EN' --target 'RU' --nhid 64 --num_layers 1 --lr 0.001 --weight_decay 0.001 --epochs 800 --adv_dim 40 --filename 'results-twitch.txt'

echo "Task EN->DE"
echo "=========="

python grade.py --source 'EN' --target 'DE' --nhid 64 --num_layers 1 --lr 0.0003 --weight_decay 0.001 --epochs 1000 --dropout_ratio 0.2 --weight 0.02 --filename 'results-twitch.txt'
python strurw.py --source 'EN' --target 'DE' --nhid 64 --num_layers 3 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --lamb 0.6 --filename 'results-twitch.txt'
python asn.py --source 'EN' --target 'DE' --nhid 64 --hid_dim_vae 64 --lr 0.001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.5 --lambda_r 0.1 --lambda_d 0.5 --lambda_f 0.0001 --filename 'results-twitch.txt'
python acdne.py --source 'EN' --target 'DE' --nhid 64 --lr 0.001 --weight_decay 0.01 --epochs 200 --dropout_ratio 0.2 --pair_weight 0.0001 --filename 'results-twitch.txt'
python adagcn.py --source 'EN' --target 'DE' --nhid 64 --num_layers 2 --lr 0.01 --weight_decay 0.01 --epochs 800 --dropout_ratio 0.4 --domain_weight 0.1 --filename 'results-twitch.txt'
python udagcn.py --source 'EN' --target 'DE' --nhid 64 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.4 --filename 'results-twitch.txt'
python specreg.py --source 'EN' --target 'DE' --nhid 64 --num_layers 5 --lr 0.003 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.1 --gamma_adv 0.1 --gamma_smooth 0.001 --gamma_mfr 0.001 --filename 'results-twitch.txt'
python a2gnn.py --source 'EN' --target 'DE' --nhid 64 --num_layers 2 --lr 0.01 --weight_decay 0.005 --epochs 800 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 10 --weight 10 --filename 'results-twitch.txt'
python pairalign.py --source 'EN' --target 'DE' --nhid 64 --num_layers 2 --lr 0.003 --weight_decay 0.003 --epochs 800 --dropout_ratio 0.0 --rw_lmda 1 --ls_lambda 3.0 --lw_lambda 0.01 --filename 'results-twitch.txt'

python kbl.py --source 'EN' --target 'DE' --nhid 64 --num_layers 3 --lr 0.001 --weight_decay 0.01 --epochs 500 --k_cross 20 --k_within 10 --filename 'results-twitch.txt'
python cwgcn.py --source 'EN' --target 'DE' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.0001 --epochs 800 --filename 'results-twitch.txt'
python dane.py --source 'EN' --target 'DE' --nhid 64 --num_layers 3 --lr 0.001 --weight_decay 0.001 --epochs 800 --filename 'results-twitch.txt'
python dgda.py --source 'EN' --target 'DE' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 800 --m_w 0.5 --beta 0.5 --filename 'results-twitch.txt'
python dmgnn.py --source 'EN' --target 'DE' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pair_weight 0.1 --filename 'results-twitch.txt'
python jhgda.py --source 'EN' --target 'DE' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 800 --pool_ratio 0.2 --filename 'results-twitch.txt'
python sagda.py --source 'EN' --target 'DE' --nhid 64 --num_layers 1 --lr 0.001 --weight_decay 0.001 --epochs 800 --adv_dim 40 --filename 'results-twitch.txt'

echo "Task EN->PT"
echo "=========="

python grade.py --source 'EN' --target 'PT' --nhid 64 --num_layers 1 --lr 0.0003 --weight_decay 0.001 --epochs 1000 --dropout_ratio 0.2 --weight 0.02 --filename 'results-twitch.txt'
python strurw.py --source 'EN' --target 'PT' --nhid 64 --num_layers 3 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --lamb 0.6 --filename 'results-twitch.txt'
python asn.py --source 'EN' --target 'PT' --nhid 64 --hid_dim_vae 64 --lr 0.001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.5 --lambda_r 0.1 --lambda_d 0.5 --lambda_f 0.0001 --filename 'results-twitch.txt'
python acdne.py --source 'EN' --target 'PT' --nhid 64 --lr 0.001 --weight_decay 0.01 --epochs 200 --dropout_ratio 0.2 --pair_weight 0.0001 --filename 'results-twitch.txt'
python adagcn.py --source 'EN' --target 'PT' --nhid 64 --num_layers 2 --lr 0.01 --weight_decay 0.01 --epochs 800 --dropout_ratio 0.4 --domain_weight 0.1 --filename 'results-twitch.txt'
python udagcn.py --source 'EN' --target 'PT' --nhid 64 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.4 --filename 'results-twitch.txt'
python specreg.py --source 'EN' --target 'PT' --nhid 64 --num_layers 5 --lr 0.003 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.1 --gamma_adv 0.1 --gamma_smooth 0.001 --gamma_mfr 0.001 --filename 'results-twitch.txt'
python a2gnn.py --source 'EN' --target 'PT' --nhid 64 --num_layers 2 --lr 0.01 --weight_decay 0.005 --epochs 800 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 10 --weight 10 --filename 'results-twitch.txt'
python pairalign.py --source 'EN' --target 'PT' --nhid 64 --num_layers 2 --lr 0.003 --weight_decay 0.003 --epochs 800 --dropout_ratio 0.0 --rw_lmda 1 --ls_lambda 3.0 --lw_lambda 0.01 --filename 'results-twitch.txt'

python kbl.py --source 'EN' --target 'PT' --nhid 64 --num_layers 3 --lr 0.001 --weight_decay 0.01 --epochs 500 --k_cross 20 --k_within 10 --filename 'results-twitch.txt'
python cwgcn.py --source 'EN' --target 'PT' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.0001 --epochs 800 --filename 'results-twitch.txt'
python dane.py --source 'EN' --target 'PT' --nhid 64 --num_layers 3 --lr 0.001 --weight_decay 0.001 --epochs 800 --filename 'results-twitch.txt'
python dgda.py --source 'EN' --target 'PT' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 800 --m_w 0.5 --beta 0.5 --filename 'results-twitch.txt'
python dmgnn.py --source 'EN' --target 'PT' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pair_weight 0.1 --filename 'results-twitch.txt'
python jhgda.py --source 'EN' --target 'PT' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 800 --pool_ratio 0.2 --filename 'results-twitch.txt'
python sagda.py --source 'EN' --target 'PT' --nhid 64 --num_layers 1 --lr 0.001 --weight_decay 0.001 --epochs 800 --adv_dim 40 --filename 'results-twitch.txt'

echo "Task EN->ES"
echo "=========="

python grade.py --source 'EN' --target 'ES' --nhid 64 --num_layers 1 --lr 0.0003 --weight_decay 0.001 --epochs 1000 --dropout_ratio 0.2 --weight 0.02 --filename 'results-twitch.txt'
python strurw.py --source 'EN' --target 'ES' --nhid 64 --num_layers 3 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --lamb 0.6 --filename 'results-twitch.txt'
python asn.py --source 'EN' --target 'ES' --nhid 64 --hid_dim_vae 64 --lr 0.001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.5 --lambda_r 0.1 --lambda_d 0.5 --lambda_f 0.0001 --filename 'results-twitch.txt'
python acdne.py --source 'EN' --target 'ES' --nhid 64 --lr 0.001 --weight_decay 0.01 --epochs 200 --dropout_ratio 0.2 --pair_weight 0.0001 --filename 'results-twitch.txt'
python adagcn.py --source 'EN' --target 'ES' --nhid 64 --num_layers 2 --lr 0.01 --weight_decay 0.01 --epochs 800 --dropout_ratio 0.4 --domain_weight 0.1 --filename 'results-twitch.txt'
python udagcn.py --source 'EN' --target 'ES' --nhid 64 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.4 --filename 'results-twitch.txt'
python specreg.py --source 'EN' --target 'ES' --nhid 64 --num_layers 5 --lr 0.003 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.1 --gamma_adv 0.1 --gamma_smooth 0.001 --gamma_mfr 0.001 --filename 'results-twitch.txt'
python a2gnn.py --source 'EN' --target 'ES' --nhid 64 --num_layers 2 --lr 0.01 --weight_decay 0.005 --epochs 800 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 10 --weight 10 --filename 'results-twitch.txt'
python pairalign.py --source 'EN' --target 'ES' --nhid 64 --num_layers 2 --lr 0.003 --weight_decay 0.003 --epochs 800 --dropout_ratio 0.0 --rw_lmda 1 --ls_lambda 3.0 --lw_lambda 0.01 --filename 'results-twitch.txt'

python kbl.py --source 'EN' --target 'ES' --nhid 64 --num_layers 3 --lr 0.001 --weight_decay 0.01 --epochs 500 --k_cross 20 --k_within 10 --filename 'results-twitch.txt'
python cwgcn.py --source 'EN' --target 'ES' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.0001 --epochs 800 --filename 'results-twitch.txt'
python dane.py --source 'EN' --target 'ES' --nhid 64 --num_layers 3 --lr 0.001 --weight_decay 0.001 --epochs 800 --filename 'results-twitch.txt'
python dgda.py --source 'EN' --target 'ES' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 800 --m_w 0.5 --beta 0.5 --filename 'results-twitch.txt'
python dmgnn.py --source 'EN' --target 'ES' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pair_weight 0.1 --filename 'results-twitch.txt'
python jhgda.py --source 'EN' --target 'ES' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 800 --pool_ratio 0.2 --filename 'results-twitch.txt'
python sagda.py --source 'EN' --target 'ES' --nhid 64 --num_layers 1 --lr 0.001 --weight_decay 0.001 --epochs 800 --adv_dim 40 --filename 'results-twitch.txt'

echo "Task EN->FR"
echo "=========="

python grade.py --source 'EN' --target 'FR' --nhid 64 --num_layers 1 --lr 0.0003 --weight_decay 0.001 --epochs 1000 --dropout_ratio 0.2 --weight 0.02 --filename 'results-twitch.txt'
python strurw.py --source 'EN' --target 'FR' --nhid 64 --num_layers 3 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --lamb 0.6 --filename 'results-twitch.txt'
python asn.py --source 'EN' --target 'FR' --nhid 64 --hid_dim_vae 64 --lr 0.001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.5 --lambda_r 0.1 --lambda_d 0.5 --lambda_f 0.0001 --filename 'results-twitch.txt'
python acdne.py --source 'EN' --target 'FR' --nhid 64 --lr 0.001 --weight_decay 0.01 --epochs 200 --dropout_ratio 0.2 --pair_weight 0.0001 --filename 'results-twitch.txt'
python adagcn.py --source 'EN' --target 'FR' --nhid 64 --num_layers 2 --lr 0.01 --weight_decay 0.01 --epochs 800 --dropout_ratio 0.4 --domain_weight 0.1 --filename 'results-twitch.txt'
python udagcn.py --source 'EN' --target 'FR' --nhid 64 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.4 --filename 'results-twitch.txt'
python specreg.py --source 'EN' --target 'FR' --nhid 64 --num_layers 5 --lr 0.003 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.1 --gamma_adv 0.1 --gamma_smooth 0.001 --gamma_mfr 0.001 --filename 'results-twitch.txt'
python a2gnn.py --source 'EN' --target 'FR' --nhid 64 --num_layers 2 --lr 0.01 --weight_decay 0.005 --epochs 800 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 10 --weight 10 --filename 'results-twitch.txt'
python pairalign.py --source 'EN' --target 'FR' --nhid 64 --num_layers 2 --lr 0.003 --weight_decay 0.003 --epochs 800 --dropout_ratio 0.0 --rw_lmda 1 --ls_lambda 3.0 --lw_lambda 0.01 --filename 'results-twitch.txt'

python kbl.py --source 'EN' --target 'FR' --nhid 64 --num_layers 3 --lr 0.001 --weight_decay 0.01 --epochs 500 --k_cross 20 --k_within 10 --filename 'results-twitch.txt'
python cwgcn.py --source 'EN' --target 'FR' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.0001 --epochs 800 --filename 'results-twitch.txt'
python dane.py --source 'EN' --target 'FR' --nhid 64 --num_layers 3 --lr 0.001 --weight_decay 0.001 --epochs 800 --filename 'results-twitch.txt'
python dgda.py --source 'EN' --target 'FR' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 800 --m_w 0.5 --beta 0.5 --filename 'results-twitch.txt'
python dmgnn.py --source 'EN' --target 'FR' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pair_weight 0.1 --filename 'results-twitch.txt'
python jhgda.py --source 'EN' --target 'FR' --nhid 64 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 800 --pool_ratio 0.2 --filename 'results-twitch.txt'
python sagda.py --source 'EN' --target 'FR' --nhid 64 --num_layers 1 --lr 0.001 --weight_decay 0.001 --epochs 800 --adv_dim 40 --filename 'results-twitch.txt'

