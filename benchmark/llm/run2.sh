echo "Task arxiv-1950-2016 -> arxiv-2016-2018"
echo "LLM enhanced text with word2vec embedding"

python a2gnn.py --source 'llm-arxiv-1950-2016' --target 'llm-arxiv-2016-2018' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.0 --epochs 800 --dropout_ratio 0.1 --s_pnums 0 --t_pnums 5 --weight 0.1 --filename 'results-llm-2.txt' --device 'cuda:2'
python udagcn.py --source 'llm-arxiv-1950-2016' --target 'llm-arxiv-2016-2018' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.4 --filename 'results-llm-2.txt' --device 'cuda:2'
python kbl.py --source 'llm-arxiv-1950-2016' --target 'llm-arxiv-2016-2018' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.0 --epochs 800 --k_cross 20 --k_within 10 --filename 'results-llm-2.txt' --device 'cuda:2'
python grade.py --source 'llm-arxiv-1950-2016' --target 'llm-arxiv-2016-2018' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.0 --epochs 800 --dropout_ratio 0.5 --weight 0.01 --filename 'results-llm-2.txt' --device 'cuda:2'
python adagcn.py --source 'llm-arxiv-1950-2016' --target 'llm-arxiv-2016-2018' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.0 --epochs 800 --dropout_ratio 0.4 --domain_weight 1.0 --filename 'results-llm-2.txt' --device 'cuda:2'


echo "Task arxiv-1950-2016 -> arxiv-2018-2020"
echo "LLM enhanced text with word2vec embedding"

python a2gnn.py --source 'llm-arxiv-1950-2016' --target 'llm-arxiv-2018-2020' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.0 --epochs 800 --dropout_ratio 0.1 --s_pnums 0 --t_pnums 5 --weight 0.1 --filename 'results-llm-2.txt' --device 'cuda:2'
python udagcn.py --source 'llm-arxiv-1950-2016' --target 'llm-arxiv-2018-2020' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.4 --filename 'results-llm-2.txt' --device 'cuda:2'
python kbl.py --source 'llm-arxiv-1950-2016' --target 'llm-arxiv-2018-2020' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.0 --epochs 800 --k_cross 20 --k_within 10 --filename 'results-llm-2.txt' --device 'cuda:2'
python grade.py --source 'llm-arxiv-1950-2016' --target 'llm-arxiv-2018-2020' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.0 --epochs 800 --dropout_ratio 0.5 --weight 0.01 --filename 'results-llm-2.txt' --device 'cuda:2'
python adagcn.py --source 'llm-arxiv-1950-2016' --target 'llm-arxiv-2018-2020' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.0 --epochs 800 --dropout_ratio 0.4 --domain_weight 1.0 --filename 'results-llm-2.txt' --device 'cuda:2'

echo "Task arxiv-2016-2018 -> arxiv-2018-2020"
echo "LLM enhanced text with word2vec embedding"

python a2gnn.py --source 'llm-arxiv-2016-2018' --target 'llm-arxiv-2018-2020' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.0 --epochs 800 --dropout_ratio 0.1 --s_pnums 0 --t_pnums 5 --weight 0.1 --filename 'results-llm-2.txt' --device 'cuda:2'
python udagcn.py --source 'llm-arxiv-2016-2018' --target 'llm-arxiv-2018-2020' --nhid 128 --num_layers 2 --lr 0.0001 --weight_decay 0.001 --epochs 800 --dropout_ratio 0.4 --filename 'results-llm-2.txt' --device 'cuda:2'
python kbl.py --source 'llm-arxiv-2016-2018' --target 'llm-arxiv-2018-2020' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.0 --epochs 800 --k_cross 20 --k_within 10 --filename 'results-llm-2.txt' --device 'cuda:2'
python grade.py --source 'llm-arxiv-2016-2018' --target 'llm-arxiv-2018-2020' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.0 --epochs 800 --dropout_ratio 0.5 --weight 0.01 --filename 'results-llm-2.txt' --device 'cuda:2'
python adagcn.py --source 'llm-arxiv-2016-2018' --target 'llm-arxiv-2018-2020' --nhid 128 --num_layers 3 --lr 0.001 --weight_decay 0.0 --epochs 800 --dropout_ratio 0.4 --domain_weight 1.0 --filename 'results-llm-2.txt' --device 'cuda:2'
