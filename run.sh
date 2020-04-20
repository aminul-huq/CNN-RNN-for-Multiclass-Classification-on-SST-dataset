#RNN implementation with dropout set to 0 and hidden dimension set to 256
python train.py --model=0  --lr=0.001 --itr=10 --dropout=0 --hidden_dim=256 --device='cuda' --n_filters=5 --filter_size=1

#RNN implementation with dropout set to 0.2 and hidden dimension set to 256
python train.py --model=0  --lr=0.001 --itr=10 --dropout=0.2 --hidden_dim=256 --device='cuda' --n_filters=5 --filter_size=1

#RNN implementation with dropout set to 0.2 and hidden dimension set to 512
python train.py --model=0  --lr=0.001 --itr=10 --dropout=0.2 --hidden_dim=512 --device='cuda' --n_filters=5 --filter_size=1

#CNN implementation with dropout set to 0 and filter_size/kernel_size set to 1
python train.py --model=1  --lr=0.001 --itr=10 --dropout=0 --hidden_dim=256 --device='cuda' --n_filters=5 --filter_size=1

#CNN implementation with dropout set to 0.2 and filter_size/kernel_size set to 1
python train.py --model=1  --lr=0.001 --itr=10 --dropout=0.2 --hidden_dim=256 --device='cuda' --n_filters=5 --filter_size=1

#CNN implementation with dropout set to 0.2 and filter_size/kernel_size set to 3
python train.py --model=1  --lr=0.001 --itr=10 --dropout=0.2 --hidden_dim=256 --device='cuda' --n_filters=5 --filter_size=3
