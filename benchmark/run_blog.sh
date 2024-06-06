#!/bin/sh

echo "Task Blog2->Blog1"
echo "=========="
python grade.py --source 'Blog2' --target 'Blog1' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 300 --dropout_ratio 0.2 --weight 0.01 --filename 'results-blog.txt'
python strurw.py --source 'Blog2' --target 'Blog1' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.4 --lamb 0.6 --filename 'results-blog.txt'
python asn.py --source 'Blog2' --target 'Blog1' --nhid 128 --hid_dim_vae 128 --lr 0.0003 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.2 --lambda_r 0.01 --lambda_d 0.5 --lambda_f 0.0001 --filename 'results-blog.txt'
python acdne.py --source 'Blog2' --target 'Blog1' --nhid 128 --lr 0.0001 --weight_decay 0.01 --epochs 300 --dropout_ratio 0.1 --pair_weight 0.03 --step 1 --filename 'results-blog.txt'
python adagcn.py --source 'Blog2' --target 'Blog1' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --domain_weight 0.1 --filename 'results-blog.txt'
python udagcn.py --source 'Blog2' --target 'Blog1' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 400 --dropout_ratio 0.4 --filename 'results-blog.txt'
python specreg.py --source 'Blog2' --target 'Blog1' --nhid 128 --num_layers 4 --lr 0.003 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.1 --gamma_adv 0.1 --gamma_smooth 0.001 --gamma_mfr 0.001 --filename 'results-blog.txt'
python a2gnn.py --source 'Blog2' --target 'Blog1' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.005 --epochs 200 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 10 --weight 10 --filename 'results-blog.txt'
python pairalign.py --source 'Blog2' --target 'Blog1' --nhid 128 --num_layers 2 --lr 0.003 --weight_decay 0.003 --epochs 200 --dropout_ratio 0.0 --rw_lmda 1 --ls_lambda 3.0 --lw_lambda 0.01 --filename 'results-blog.txt'

python kbl.py --source 'Blog2' --target 'Blog1' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.01 --epochs 200 --k_cross 20 --k_within 10 --filename 'results-blog.txt'
python cwgcn.py --source 'Blog2' --target 'Blog1' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.0001 --epochs 200 --filename 'results-blog.txt'
python dane.py --source 'Blog2' --target 'Blog1' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.001 --epochs 200 --filename 'results-blog.txt'
python dgda.py --source 'Blog2' --target 'Blog1' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --m_w 0.5 --beta 0.5 --filename 'results-blog.txt'
python dmgnn.py --source 'Blog2' --target 'Blog1' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pair_weight 0.1 --filename 'results-blog.txt'
python jhgda.py --source 'Blog2' --target 'Blog1' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pool_ratio 0.2 --filename 'results-blog.txt'
python sagda.py --source 'Blog2' --target 'Blog1' --nhid 128 --num_layers 1 --lr 0.001 --weight_decay 0.001 --epochs 200 --adv_dim 40 --filename 'results-blog.txt'

echo "Task Blog1->Blog2"
echo "=========="
python grade.py --source 'Blog1' --target 'Blog2' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.5 --weight 0.01 --filename 'results-blog.txt'
python strurw.py --source 'Blog1' --target 'Blog2' --nhid 128 --num_layers 1 --lr 0.001 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.4 --lamb 0.6 --filename 'results-blog.txt'
python asn.py --source 'Blog1' --target 'Blog2' --nhid 128 --hid_dim_vae 128 --lr 0.0003 --weight_decay 0.01 --epochs 300 --dropout_ratio 0.2 --lambda_r 0.01 --lambda_d 0.5 --lambda_f 0.0001 --filename 'results-blog.txt'
python acdne.py --source 'Blog1' --target 'Blog2' --nhid 128 --lr 0.0003 --weight_decay 0.01 --epochs 300 --dropout_ratio 0.0 --pair_weight 0.01 --step 1 --filename 'results-blog.txt'
python adagcn.py --source 'Blog1' --target 'Blog2' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.01 --epochs 400 --dropout_ratio 0.4 --domain_weight 0.1 --filename 'results-blog.txt'
python udagcn.py --source 'Blog1' --target 'Blog2' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 400 --dropout_ratio 0.4 --filename 'results-blog.txt'
python specreg.py --source 'Blog1' --target 'Blog2' --nhid 128 --num_layers 4 --lr 0.003 --weight_decay 0.001 --epochs 200 --dropout_ratio 0.1 --gamma_adv 0.1 --gamma_smooth 0.001 --gamma_mfr 0.001 --filename 'results-blog.txt'
python a2gnn.py --source 'Blog1' --target 'Blog2' --nhid 128 --num_layers 2 --lr 0.01 --weight_decay 0.005 --epochs 200 --dropout_ratio 0.5 --s_pnums 0 --t_pnums 10 --weight 10 --filename 'results-blog.txt'
python pairalign.py --source 'Blog1' --target 'Blog2' --nhid 128 --num_layers 2 --lr 0.003 --weight_decay 0.003 --epochs 200 --dropout_ratio 0.0 --rw_lmda 1 --ls_lambda 3.0 --lw_lambda 0.01 --filename 'results-blog.txt'

python kbl.py --source 'Blog1' --target 'Blog2' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.01 --epochs 200 --k_cross 20 --k_within 10 --filename 'results-blog.txt'
python cwgcn.py --source 'Blog1' --target 'Blog2' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.0001 --epochs 200 --filename 'results-blog.txt'
python dane.py --source 'Blog1' --target 'Blog2' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.001 --epochs 200 --filename 'results-blog.txt'
python dgda.py --source 'Blog1' --target 'Blog2' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --m_w 0.5 --beta 0.5 --filename 'results-blog.txt'
python dmgnn.py --source 'Blog1' --target 'Blog2' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pair_weight 0.1 --filename 'results-blog.txt'
python jhgda.py --source 'Blog1' --target 'Blog2' --nhid 128 --num_layers 2 --lr 0.001 --weight_decay 0.001 --epochs 200 --pool_ratio 0.2 --filename 'results-blog.txt'
python sagda.py --source 'Blog1' --target 'Blog2' --nhid 128 --num_layers 1 --lr 0.001 --weight_decay 0.001 --epochs 200 --adv_dim 40 --filename 'results-blog.txt'
