#!/bin/sh

echo "Task D->C"
echo "=========="
python grade.py --source 'DBLPv7' --target 'Citationv1' --nhid 128 --num_layers 5 --lr 0.001 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.5 --weight 0.005 --filename 'results-citation.txt'
python strurw.py --source 'DBLPv7' --target 'Citationv1' --nhid 128 --num_layers 2 --lr 0.003 --weight_decay 0.01 --epochs 200 --dropout_ratio 0.2 --lamb 0.8 --filename 'results-citation.txt'
python asn.py --source 'DBLPv7' --target 'Citationv1' --nhid 128 --hid_dim_vae 128 --lr 0.001 --weight_decay 0.001 --epochs 100 --dropout_ratio 0.3 --lambda_r 0.1 --lambda_d 1.0 --lambda_f 0.001 --filename 'results-citation.txt'
python acdne.py --source 'DBLPv7' --target 'Citationv1' --nhid 128 --lr 0.001 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.5 --pair_weight 0.1 --filename 'results-citation.txt'
python adagcn.py --source 'DBLPv7' --target 'Citationv1' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --domain_weight 0.1 --filename 'results-citation.txt'
python udagcn.py --source 'DBLPv7' --target 'Citationv1' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 400 --dropout_ratio 0.4 --filename 'results-citation.txt'
python specreg.py --source 'DBLPv7' --target 'Citationv1' --nhid 128 --num_layers 4 --lr 0.003 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.1 --gamma_adv 0.1 --gamma_smooth 0.001 --gamma_mfr 0.001 --filename 'results-citation.txt'
python a2gnn.py --source 'DBLPv7' --target 'Citationv1' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.005 --epochs 200 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 10 --weight 10 --filename 'results-citation.txt'
python pairalign.py --source 'DBLPv7' --target 'Citationv1' --nhid 128 --num_layers 2 --lr 0.003 --weight_decay 0.003 --epochs 200 --dropout_ratio 0.0 --rw_lmda 1 --ls_lambda 3.0 --lw_lambda 0.01 --filename 'results-citation.txt'

python kbl.py --source 'DBLPv7' --target 'Citationv1' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.01 --epochs 200 --k_cross 20 --k_within 10 --filename 'results-citation.txt'
python cwgcn.py --source 'DBLPv7' --target 'Citationv1' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.0001 --epochs 200 --filename 'results-citation.txt'
python dane.py --source 'DBLPv7' --target 'Citationv1' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.001 --epochs 200 --filename 'results-citation.txt'
python dgda.py --source 'DBLPv7' --target 'Citationv1' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --m_w 0.5 --beta 0.5 --filename 'results-citation.txt'
python dmgnn.py --source 'DBLPv7' --target 'Citationv1' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pair_weight 0.1 --filename 'results-citation.txt'
python jhgda.py --source 'DBLPv7' --target 'Citationv1' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pool_ratio 0.2 --filename 'results-citation.txt'
python sagda.py --source 'DBLPv7' --target 'Citationv1' --nhid 128 --num_layers 1 --lr 0.001 --weight_decay 0.001 --epochs 200 --adv_dim 40 --filename 'results-citation.txt'

echo "Task D->A"
echo "=========="
python grade.py --source 'DBLPv7' --target 'ACMv9' --nhid 128 --num_layers 5 --lr 0.001 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.2 --weight 0.002 --filename 'results-citation.txt'
python strurw.py --source 'DBLPv7' --target 'ACMv9' --nhid 128 --num_layers 2 --lr 0.003 --weight_decay 0.01 --epochs 200 --dropout_ratio 0.2 --lamb 0.8 --filename 'results-citation.txt'
python asn.py --source 'DBLPv7' --target 'ACMv9' --nhid 128 --hid_dim_vae 128 --lr 0.001 --weight_decay 0.001 --epochs 100 --dropout_ratio 0.3 --lambda_r 0.1 --lambda_d 1.0 --lambda_f 0.001 --filename 'results-citation.txt'
python acdne.py --source 'DBLPv7' --target 'ACMv9' --nhid 128 --lr 0.001 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.5 --pair_weight 0.1 --filename 'results-citation.txt'
python adagcn.py --source 'DBLPv7' --target 'ACMv9' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --domain_weight 0.1 --filename 'results-citation.txt'
python udagcn.py --source 'DBLPv7' --target 'ACMv9' --nhid 128 --num_layers 3 --lr 0.0001 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.4 --filename 'results-citation.txt'
python specreg.py --source 'DBLPv7' --target 'ACMv9' --nhid 128 --num_layers 4 --lr 0.003 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.1 --gamma_adv 0.1 --gamma_smooth 0.001 --gamma_mfr 0.001 --filename 'results-citation.txt'
python a2gnn.py --source 'DBLPv7' --target 'ACMv9' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.005 --epochs 200 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 10 --weight 10 --filename 'results-citation.txt'
python pairalign.py --source 'DBLPv7' --target 'ACMv9' --nhid 128 --num_layers 2 --lr 0.005 --weight_decay 0.003 --epochs 200 --dropout_ratio 0.0 --rw_lmda 1 --ls_lambda 3.0 --lw_lambda 0.01 --filename 'results-citation.txt'

python kbl.py --source 'DBLPv7' --target 'ACMv9' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.01 --epochs 200 --k_cross 20 --k_within 10 --filename 'results-citation.txt'
python cwgcn.py --source 'DBLPv7' --target 'ACMv9' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.0001 --epochs 200 --filename 'results-citation.txt'
python dane.py --source 'DBLPv7' --target 'ACMv9' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.0 --epochs 200 --filename 'results-citation.txt'
python dgda.py --source 'DBLPv7' --target 'ACMv9' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --m_w 0.1 --beta 0.5 --filename 'results-citation.txt'
python dmgnn.py --source 'DBLPv7' --target 'ACMv9' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pair_weight 0.1 --filename 'results-citation.txt'
python jhgda.py --source 'DBLPv7' --target 'ACMv9' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pool_ratio 0.2 --filename 'results-citation.txt'
python sagda.py --source 'DBLPv7' --target 'ACMv9' --nhid 128 --num_layers 1 --lr 0.0001 --weight_decay 0.001 --epochs 800 --adv_dim 40 --filename 'results-citation.txt'

echo "Task C->D"
echo "=========="
python grade.py --source 'Citationv1' --target 'DBLPv7' --nhid 128 --num_layers 5 --lr 0.001 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.5 --weight 0.01 --filename 'results-citation.txt'
python strurw.py --source 'Citationv1' --target 'DBLPv7' --nhid 128 --num_layers 2 --lr 0.003 --weight_decay 0.01 --epochs 200 --dropout_ratio 0.2 --lamb 0.8 --filename 'results-citation.txt'
python asn.py --source 'Citationv1' --target 'DBLPv7' --nhid 128 --hid_dim_vae 128 --lr 0.001 --weight_decay 0.001 --epochs 100 --dropout_ratio 0.3 --lambda_r 0.1 --lambda_d 1.0 --lambda_f 0.001 --filename 'results-citation.txt'
python acdne.py --source 'Citationv1' --target 'DBLPv7' --nhid 128 --lr 0.001 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.5 --pair_weight 0.1 --filename 'results-citation.txt'
python adagcn.py --source 'Citationv1' --target 'DBLPv7' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --filename 'results-citation.txt'
python udagcn.py --source 'Citationv1' --target 'DBLPv7' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 400 --dropout_ratio 0.4 --filename 'results-citation.txt'
python specreg.py --source 'Citationv1' --target 'DBLPv7' --nhid 128 --num_layers 4 --lr 0.003 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.1 --gamma_adv 0.1 --gamma_smooth 0.001 --gamma_mfr 0.001 --filename 'results-citation.txt'
python a2gnn.py --source 'Citationv1' --target 'DBLPv7' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.005 --epochs 200 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 10 --weight 10 --filename 'results-citation.txt'
python pairalign.py --source 'Citationv1' --target 'DBLPv7' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.003 --epochs 200 --dropout_ratio 0.0 --rw_lmda 1 --ls_lambda 3.0 --lw_lambda 0.01 --filename 'results-citation.txt'

python kbl.py --source 'Citationv1' --target 'DBLPv7' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.01 --epochs 200 --k_cross 20 --k_within 10 --filename 'results-citation.txt'
python cwgcn.py --source 'Citationv1' --target 'DBLPv7' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.0001 --epochs 200 --filename 'results-citation.txt'
python dane.py --source 'Citationv1' --target 'DBLPv7' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.0 --epochs 200 --filename 'results-citation.txt'
python dgda.py --source 'Citationv1' --target 'DBLPv7' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --m_w 0.5 --beta 0.5 --filename 'results-citation.txt'
python dmgnn.py --source 'Citationv1' --target 'DBLPv7' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pair_weight 0.1 --filename 'results-citation.txt'
python jhgda.py --source 'Citationv1' --target 'DBLPv7' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pool_ratio 0.2 --filename 'results-citation.txt'
python sagda.py --source 'Citationv1' --target 'DBLPv7' --nhid 128 --num_layers 1 --lr 0.0001 --weight_decay 0.001 --epochs 400 --adv_dim 40 --filename 'results-citation.txt'

echo "Task C->A"
echo "=========="
python grade.py --source 'Citationv1' --target 'ACMv9' --nhid 128 --num_layers 4 --lr 0.001 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.5 --weight 0.002 --filename 'results-citation.txt'
python strurw.py --source 'Citationv1' --target 'ACMv9' --nhid 128 --num_layers 2 --lr 0.003 --weight_decay 0.01 --epochs 200 --dropout_ratio 0.2 --lamb 0.8 --filename 'results-citation.txt'
python asn.py --source 'Citationv1' --target 'ACMv9' --nhid 128 --hid_dim_vae 128 --lr 0.001 --weight_decay 0.001 --epochs 100 --dropout_ratio 0.3 --lambda_r 0.1 --lambda_d 1.0 --lambda_f 0.001 --filename 'results-citation.txt'
python acdne.py --source 'Citationv1' --target 'ACMv9' --nhid 128 --lr 0.001 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.5 --pair_weight 0.1 --filename 'results-citation.txt'
python adagcn.py --source 'Citationv1' --target 'ACMv9' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --filename 'results-citation.txt'
python udagcn.py --source 'Citationv1' --target 'ACMv9' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 400 --dropout_ratio 0.4 --filename 'results-citation.txt'
python specreg.py --source 'Citationv1' --target 'ACMv9' --nhid 128 --num_layers 4 --lr 0.003 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.1 --gamma_adv 0.1 --gamma_smooth 0.001 --gamma_mfr 0.001 --filename 'results-citation.txt'
python a2gnn.py --source 'Citationv1' --target 'ACMv9' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.005 --epochs 200 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 10 --weight 10 --filename 'results-citation.txt'
python pairalign.py --source 'Citationv1' --target 'ACMv9' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.003 --epochs 200 --dropout_ratio 0.0 --rw_lmda 1 --ls_lambda 3.0 --lw_lambda 0.01 --filename 'results-citation.txt'

python kbl.py --source 'Citationv1' --target 'ACMv9' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.01 --epochs 200 --k_cross 20 --k_within 10 --filename 'results-citation.txt'
python cwgcn.py --source 'Citationv1' --target 'ACMv9' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.0001 --epochs 200 --filename 'results-citation.txt'
python dane.py --source 'Citationv1' --target 'ACMv9' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.0 --epochs 200 --filename 'results-citation.txt'
python dgda.py --source 'Citationv1' --target 'ACMv9' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.001 --epochs 200 --m_w 0.5 --beta 0.5 --d_w 0.1 --filename 'results-citation.txt'
python dmgnn.py --source 'Citationv1' --target 'ACMv9' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pair_weight 0.1 --filename 'results-citation.txt'
python jhgda.py --source 'Citationv1' --target 'ACMv9' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pool_ratio 0.2 --filename 'results-citation.txt'
python sagda.py --source 'Citationv1' --target 'ACMv9' --nhid 128 --num_layers 1 --lr 0.0001 --weight_decay 0.001 --epochs 800 --adv_dim 40 --filename 'results-citation.txt'

echo "Task A->D"
echo "=========="
python grade.py --source 'ACMv9' --target 'DBLPv7' --nhid 128 --num_layers 5 --lr 0.001 --weight_decay 0.0 --epochs 300 --dropout_ratio 0.5 --weight 0.002 --filename 'results-citation.txt'
python strurw.py --source 'ACMv9' --target 'DBLPv7' --nhid 128 --num_layers 2 --lr 0.003 --weight_decay 0.01 --epochs 200 --dropout_ratio 0.2 --lamb 0.8 --filename 'results-citation.txt'
python asn.py --source 'ACMv9' --target 'DBLPv7' --nhid 128 --hid_dim_vae 128 --lr 0.001 --weight_decay 0.001 --epochs 100 --dropout_ratio 0.3 --lambda_r 0.1 --lambda_d 1.0 --lambda_f 0.001 --filename 'results-citation.txt'
python acdne.py --source 'ACMv9' --target 'DBLPv7' --nhid 128 --lr 0.001 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.5 --pair_weight 0.1 --filename 'results-citation.txt'
python adagcn.py --source 'ACMv9' --target 'DBLPv7' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --filename 'results-citation.txt'
python udagcn.py --source 'ACMv9' --target 'DBLPv7' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 400 --dropout_ratio 0.4 --filename 'results-citation.txt'
python specreg.py --source 'ACMv9' --target 'DBLPv7' --nhid 128 --num_layers 4 --lr 0.003 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.1 --gamma_adv 0.1 --gamma_smooth 0.001 --gamma_mfr 0.001 --filename 'results-citation.txt'
python a2gnn.py --source 'ACMv9' --target 'DBLPv7' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.005 --epochs 200 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 10 --weight 10 --filename 'results-citation.txt'
python pairalign.py --source 'ACMv9' --target 'DBLPv7' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.003 --epochs 200 --dropout_ratio 0.0 --rw_lmda 1 --ls_lambda 3.0 --lw_lambda 0.01 --filename 'results-citation.txt'

python kbl.py --source 'ACMv9' --target 'DBLPv7' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.01 --epochs 200 --k_cross 20 --k_within 10 --filename 'results-citation.txt'
python cwgcn.py --source 'ACMv9' --target 'DBLPv7' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.0001 --epochs 200 --filename 'results-citation.txt'
python dane.py --source 'ACMv9' --target 'DBLPv7' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.0 --epochs 200 --filename 'results-citation.txt'
python dgda.py --source 'ACMv9' --target 'DBLPv7' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --m_w 0.5 --beta 0.5 --d_w 0.1 --filename 'results-citation.txt'
python dmgnn.py --source 'ACMv9' --target 'DBLPv7' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pair_weight 0.1 --filename 'results-citation.txt'
python jhgda.py --source 'ACMv9' --target 'DBLPv7' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pool_ratio 0.2 --filename 'results-citation.txt'
python sagda.py --source 'ACMv9' --target 'DBLPv7' --nhid 128 --num_layers 1 --lr 0.0001 --weight_decay 0.001 --epochs 800 --adv_dim 40 --filename 'results-citation.txt'

echo "Task A->C"
echo "=========="
python grade.py --source 'ACMv9' --target 'Citationv1' --nhid 128 --num_layers 5 --lr 0.001 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.5 --weight 0.005 --filename 'results-citation.txt'
python strurw.py --source 'ACMv9' --target 'Citationv1' --nhid 128 --num_layers 2 --lr 0.003 --weight_decay 0.01 --epochs 200 --dropout_ratio 0.2 --lamb 0.8 --filename 'results-citation.txt'
python asn.py --source 'ACMv9' --target 'Citationv1' --nhid 128 --hid_dim_vae 128 --lr 0.001 --weight_decay 0.001 --epochs 100 --dropout_ratio 0.3 --lambda_r 0.1 --lambda_d 1.0 --lambda_f 0.001 --filename 'results-citation.txt'
python acdne.py --source 'ACMv9' --target 'Citationv1' --nhid 128 --lr 0.001 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.5 --pair_weight 0.1 --filename 'results-citation.txt'
python adagcn.py --source 'ACMv9' --target 'Citationv1' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --filename 'results-citation.txt'
python udagcn.py --source 'ACMv9' --target 'Citationv1' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 400 --dropout_ratio 0.4 --filename 'results-citation.txt'
python specreg.py --source 'ACMv9' --target 'Citationv1' --nhid 128 --num_layers 4 --lr 0.003 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.1 --gamma_adv 0.1 --gamma_smooth 0.001 --gamma_mfr 0.001 --filename 'results-citation.txt'
python a2gnn.py --source 'ACMv9' --target 'Citationv1' --nhid 128 --num_layers 1 --lr 0.005 --weight_decay 0.005 --epochs 200 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 5 --weight 5 --filename 'results-citation.txt'
python pairalign.py --source 'ACMv9' --target 'Citationv1' --nhid 128 --num_layers 2 --lr 0.003 --weight_decay 0.001 --epochs 400 --dropout_ratio 0.0 --rw_lmda 1 --ls_lambda 5.0 --lw_lambda 0.01 --filename 'results-citation.txt'

python kbl.py --source 'ACMv9' --target 'Citationv1' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.01 --epochs 200 --k_cross 20 --k_within 10 --filename 'results-citation.txt'
python cwgcn.py --source 'ACMv9' --target 'Citationv1' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.0001 --epochs 200 --filename 'results-citation.txt'
python dane.py --source 'ACMv9' --target 'Citationv1' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.0 --epochs 200 --filename 'results-citation.txt'
python dgda.py --source 'ACMv9' --target 'Citationv1' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --m_w 0.1 --beta 0.5 --filename 'results-citation.txt'
python dmgnn.py --source 'ACMv9' --target 'Citationv1' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pair_weight 0.1 --filename 'results-citation.txt'
python jhgda.py --source 'ACMv9' --target 'Citationv1' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pool_ratio 0.2 --filename 'results-citation.txt'
python sagda.py --source 'ACMv9' --target 'Citationv1' --nhid 128 --num_layers 1 --lr 0.0001 --weight_decay 0.001 --epochs 800 --adv_dim 40 --filename 'results-citation.txt'
