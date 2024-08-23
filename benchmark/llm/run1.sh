echo "Task arxiv-1950-2016 -> arxiv-2016-2018"
echo "Original ogbn features"

python a2gnn.py --source 'arxiv-1950-2016' --target 'arxiv-2016-2018' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.0 --epochs 800 --dropout_ratio 0.1 --s_pnums 0 --t_pnums 5 --weight 0.1 --filename 'results-llm-1.txt' --device 'cuda:1'
python udagcn.py --source 'arxiv-1950-2016' --target 'arxiv-2016-2018' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.4 --filename 'results-llm-1.txt' --device 'cuda:1'
python kbl.py --source 'arxiv-1950-2016' --target 'arxiv-2016-2018' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.01 --epochs 800 --k_cross 20 --k_within 10 --filename 'results-llm-1.txt' --device 'cuda:1'
python grade.py --source 'arxiv-1950-2016' --target 'arxiv-2016-2018' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.0 --epochs 800 --dropout_ratio 0.5 --weight 0.01 --filename 'results-llm-1.txt' --device 'cuda:1'
python adagcn.py --source 'arxiv-1950-2016' --target 'arxiv-2016-2018' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.0 --epochs 800 --dropout_ratio 0.4 --domain_weight 1.0 --filename 'results-llm-1.txt' --device 'cuda:1'

echo "Task arxiv-1950-2016 -> arxiv-2018-2020"
echo "Original ogbn features"

python a2gnn.py --source 'arxiv-1950-2016' --target 'arxiv-2018-2020' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.0 --epochs 800 --dropout_ratio 0.1 --s_pnums 0 --t_pnums 5 --weight 0.1 --filename 'results-llm-1.txt' --device 'cuda:1'
python udagcn.py --source 'arxiv-1950-2016' --target 'arxiv-2018-2020' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.4 --filename 'results-llm-1.txt' --device 'cuda:1'
python kbl.py --source 'arxiv-1950-2016' --target 'arxiv-2018-2020' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.01 --epochs 800 --k_cross 20 --k_within 10 --filename 'results-llm-1.txt' --device 'cuda:1'
python grade.py --source 'arxiv-1950-2016' --target 'arxiv-2018-2020' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.0 --epochs 800 --dropout_ratio 0.5 --weight 0.01 --filename 'results-llm-1.txt' --device 'cuda:1'
python adagcn.py --source 'arxiv-1950-2016' --target 'arxiv-2018-2020' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.0 --epochs 800 --dropout_ratio 0.4 --domain_weight 1.0 --filename 'results-llm-1.txt' --device 'cuda:1'

echo "Task arxiv-2016-2018 -> arxiv-2018-2020"
echo "Original ogbn features"

python a2gnn.py --source 'arxiv-2016-2018' --target 'arxiv-2018-2020' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.0 --epochs 800 --dropout_ratio 0.1 --s_pnums 0 --t_pnums 5 --weight 0.1 --filename 'results-llm-1.txt' --device 'cuda:1'
python udagcn.py --source 'arxiv-2016-2018' --target 'arxiv-2018-2020' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.4 --filename 'results-llm-1.txt' --device 'cuda:1'
python kbl.py --source 'arxiv-2016-2018' --target 'arxiv-2018-2020' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.01 --epochs 800 --k_cross 20 --k_within 10 --filename 'results-llm-1.txt' --device 'cuda:1'
python grade.py --source 'arxiv-2016-2018' --target 'arxiv-2018-2020' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.0 --epochs 800 --dropout_ratio 0.5 --weight 0.01 --filename 'results-llm-1.txt' --device 'cuda:1'
python adagcn.py --source 'arxiv-2016-2018' --target 'arxiv-2018-2020' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.0 --epochs 800 --dropout_ratio 0.4 --domain_weight 1.0 --filename 'results-llm-1.txt' --device 'cuda:1'