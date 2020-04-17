import torch
from utils import *
from models import *
import argparse
import torch.optim as optim

train_iterator, valid_iterator, test_iterator, TEXT = get_SSTdata()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Training on SST dataset using RNN')

    # model hyper-parameter variables
    parser.add_argument('--model', default=0, metavar='model', type=int, help='0 for RNN 1 for CNN 2 for three layer CNN')
    parser.add_argument('--lr', default=0.001, metavar='lr', type=float, help='Learning rate')
    parser.add_argument('--itr', default=5, metavar='iter', type=int, help='Number of iterations')
    parser.add_argument('--dropout', default=0.5, metavar='dropout', type=float, help='Dropout Value')
    parser.add_argument('--hidden_dim', default=256, metavar='hidden_dim', type=int, help='Number hidden units')
    parser.add_argument('--device', default='cpu', metavar='device', type=str, help='Dropout Value')
    parser.add_argument('--n_filters', default=100, metavar='n_filters', type=int, help='No. of filters')
    parser.add_argument('--filter_size', default=3, metavar='filter_size', type=int, help='filters sizes')
    
    args = parser.parse_args()

    
    if args.device=='cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print("Using ",device)
    
    DROPOUT = args.dropout
    HIDDEN_DIM = args.hidden_dim
    num_epochs = args.itr
    lr = args.lr
    N_FILTERS = args.n_filters
    FILTER_SIZES = args.filter_size
    
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 300
    OUTPUT_DIM = 5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    
    if args.model == 0:
        print("Using RNN as Model")
        model = RNN(INPUT_DIM, EMBEDDING_DIM,DROPOUT, HIDDEN_DIM, OUTPUT_DIM).to(device)
    elif args.model == 1:
        print("Using CNN as Model")
        model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX).to(device)
         
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()


    history = train_eval(model, train_iterator, valid_iterator, num_epochs, optimizer, criterion, device)
    test(model,test_iterator,criterion,device)