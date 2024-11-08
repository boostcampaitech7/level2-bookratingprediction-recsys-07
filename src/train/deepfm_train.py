from preprocess import *
import torch
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from models import *
import ast
from torch import nn
from loss import RMSELoss
import torch.optim as optim
from tqdm import tqdm
import random 
import os
from hyperopt import fmin, tpe, Trials, hp, STATUS_OK
import json
# def train_eval(model,train_dataloader,optimiser):
#     model.train()  # Set model to training mode
#     train_loss = 0.0
#     for X_batch, y_batch in tqdm(train_dataloader):
#         X_batch, y_batch = X_batch.to(args.device), y_batch.to(args.device)

#             # Forward pass
#             outputs = model(X_batch)
#             loss = criterion(outputs, y_batch.float())

#             # Backward pass and optimization
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item() * X_batch.size(0)

#         # Calculate average training loss
#         train_loss /= len(train_dataloader.dataset)
# def validate_eval():
    
# def predict_eval():

def seed_everything(seed):
        '''
        [description]
        seed 값을 고정시키는 함수입니다.

        [arguments]
        seed : seed 값
        '''
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def deepfm_train(args):
    seed_everything(args.seed)
    print(f'--------------- Load Data ---------------')
    data=preprocess_context_data(args.filepath)
    print(f'--------------- Split Data ---------------')
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                            data['train'].drop(['rating'], axis=1),
                                                            data['train']['rating'],
                                                            test_size=args.ratio,
                                                            random_state=args.seed,
                                                            shuffle=True
                                                            )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid

    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values)) if args.ratio != 0 else None
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0) if args.ratio != 0 else None
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f'--------------- INIT model ---------------')
    model = DeepFM(args,data)
    model = model.to(args.device)
    criterion = RMSELoss().to(args.device) # 방금 args.device추가 
    optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=0.0001)

    print(f'---------------TRAINING ---------------')
    for epoch in range(args.epochs):
        model.train()  # Set model to training mode
        train_loss = 0.0
        for X_batch, y_batch in tqdm(train_dataloader):
            X_batch, y_batch = X_batch.to(args.device), y_batch.to(args.device)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.float())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        # Calculate average training loss
        train_loss /= len(train_dataloader.dataset)

        # Validation (if validation set is used)
        if valid_dataloader:
            model.eval()  # Set model to evaluation mode
            valid_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in tqdm(valid_dataloader):
                    X_batch, y_batch = X_batch.to(args.device), y_batch.to(args.device)

                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch.float())
                    valid_loss += loss.item() * X_batch.size(0)

            valid_loss /= len(valid_dataloader.dataset)
            print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}")
        else:
            print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.4f}")



def objective(params):
    # Update args with hyperparameters
    global data

    args.lr = params['lr']
    args.dropout = params['dropout']
    args.weight_decay = params['weight_decay']
    args.epochs = params['epochs']
    args.embed_dim = params['embed_dim']

    # Seed everything
    seed_everything(args.seed)

    # Load and preprocess data
    X_train, X_valid, y_train, y_valid = train_test_split(
        data['train'].drop(['rating'], axis=1),
        data['train']['rating'],
        test_size=args.ratio,
        random_state=args.seed,
        shuffle=True
    )

    # Update data dictionary
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values)) if args.ratio != 0 else None
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0) if args.ratio != 0 else None

    # Initialize model, loss, and optimizer
    model = DeepFM(args, data).to(args.device)
    criterion = RMSELoss().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop with tqdm progress bar
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        print(f"Epoch {epoch + 1}/{args.epochs}")
        for X_batch, y_batch in tqdm(train_dataloader, desc="Training", leave=False):
            X_batch, y_batch = X_batch.to(args.device), y_batch.to(args.device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        # Validation loss with tqdm progress bar
        if valid_dataloader:
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in tqdm(valid_dataloader, desc="Validation", leave=False):
                    X_batch, y_batch = X_batch.to(args.device), y_batch.to(args.device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch.float())
                    valid_loss += loss.item() * X_batch.size(0)

            valid_loss /= len(valid_dataloader.dataset)
            print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss / len(train_dataloader.dataset):.4f}, Validation Loss: {valid_loss:.4f}")
        else:
            print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss / len(train_dataloader.dataset):.4f}")

    return {'loss': valid_loss, 'status': STATUS_OK}


if __name__== "__main__":

    parser = argparse.ArgumentParser(description='parser')
    
    arg=parser.add_argument
    
    arg('--filepath','-fp',type=str,help='File path를 지정합니다')
    arg('--seed',type=int,
        help='초기 사용 시드를 고정합니다')
    arg('--device', type=str,
        choices=['cude','cpu'],help='사용할 디바이스를 선택가능')
    arg('--lr',type=int,
        help='learning rate 지정')
    arg('--ratio',type=int,
        help='train valid dataset ratio')
    arg('--mlp_dims', type=list,
        help='the list of mlp_dims')
    arg('--embed_dim',type=int,
        help='embedding dimension')
    arg('--batchnorm',type=ast.literal_eval,
        help='batchnorm 설정')
    arg('--dropout',type=float,
        help='dropout 비율 설정')
    arg('--batch_size',type=int,
        help='batchsize설정')
    arg('--epochs',type=int,
        help='ephochs 정하기')
    arg('--tuning',type=ast.literal_eval,
        help='tuning ok?')
    arg('--weight_decay',type=float,
        help='weight_decay')
    
    
    args=parser.parse_args()    

    args.filepath = "/data/ephemeral/home/data"
    args.seed = args.seed if args.seed is not None else 42
    args.device = args.device or 'cuda'
    args.lr = args.lr if args.lr is not None else 0.05
    args.ratio = args.ratio if args.ratio is not None else 0.2
    args.embed_dim = args.embed_dim if args.embed_dim is not None else 16
    args.batchnorm = args.batchnorm if args.batchnorm is not None else True
    args.dropout = args.dropout if args.dropout is not None else 0.2
    args.epochs = args.epochs if args.epochs is not None else 20
    args.batch_size = args.batch_size if args.batch_size is not None else 1024
    args.tuning= args.tuning if args.tuning is not None else False
    args.weight_decay= args.weight_decay if args.weight_decay is not None else 1e-4

    # Convert mlp_dims from comma-separated string to a list of integers
    if args.mlp_dims:
        args.mlp_dims = [int(dim) for dim in args.mlp_dims.split(',')]
    else:
        args.mlp_dims = [16, 32]
    
    if not args.tuning:
        deepfm_train(args)
    else:
        seed_everything(args.seed)
        print(f'--------------- Data Loading ---------------')
        data = preprocess_context_data(args.filepath)
        space = {}
        space['lr'] = hp.uniform('lr', 0.01, 0.2)
        space['dropout'] = hp.uniform('dropout', 0.01, 0.2)
        space['weight_decay'] = hp.uniform('weight_decay',5e-5, 1e-3)
        space['epochs'] = hp.choice('epochs', [10, 20, 30, 50])
        space['embed_dim'] = hp.choice('embed_dim',[8,16,32])
        print(f'--------------- Tunning ---------------')
        rstate = np.random.default_rng(42)
        trials = Trials()
        best_params = fmin(
                fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=40,
                trials=trials
                )
        # best_params_dir = '/data/ephemeral/home/nate/level2-bookratingprediction-recsys-07/saved'
        # os.makedirs(best_params_dir, exist_ok=True)            
        # best_params_path = os.path.join(best_params_dir, "best_params.json")
        # # Save the best hyperparameters to the specified folder
        # with open(best_params_path, "w") as f:
        #     json.dump(best_params, f)
        # print(f"Best parameters saved to {best_params_path}")
        print(f'--------------- train with best_params ---------------')
        args.lr = best_params['lr']
        args.dropout = best_params['dropout']
        args.weight_decay = best_params['weight_decay']
        args.epochs = [10, 20, 30, 50][best_params['epochs']]  # Map index to value
        args.embed_dim = [8, 16, 32][best_params['embed_dim']]  # Map index to valu

        train_dataset = TensorDataset(
            torch.LongTensor(data['train'].drop('rating', axis=1).values),  # Convert to NumPy array
            torch.LongTensor(data['train']['rating'].values)  # Convert to NumPy array
            )
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        model = DeepFM(args, data).to(args.device)
        criterion = RMSELoss().to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_dataloader:
                X_batch, y_batch = X_batch.to(args.device), y_batch.to(args.device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)
            print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss / len(train_dataloader.dataset):.4f}")


        print(f'--------------- PREDICT ---------------')
        model.eval()
        test_dataset = TensorDataset(torch.LongTensor(data['test'].values))
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        predictions = []
        with torch.no_grad():
            for X_batch in test_dataloader:
                X_batch = X_batch[0].to(args.device)  # Only the features are needed for prediction
                outputs = model(X_batch)
                predictions.extend(outputs.tolist())  # Collect predictions
        
        submission = data['sub']
        submission['rating'] = predictions
        submission.to_csv('/data/ephemeral/home/nate/level2-bookratingprediction-recsys-07/saved/submission.csv', index=False)
       
        

        