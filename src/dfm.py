import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import tqdm
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

from dfm_dataset import Market_User
from Deep_factorization_machine import DeepFactorizationMachineModel

#parameters
TRAIN_MODE = False
DATASET_NAME = 'Market_User'
DATASET_PATH_TRAIN = '../generated_data/data_5000_train.dat'
DATASET_PATH_TEST  = '../generated_data/data_5000_test.dat'
MODEL_NAME = 'dfm'
DEVICE = 'cuda:0'
MODEL_SAVE_DIR = '../model'
PRETRAINED_MODEL_PATH = '../model_lr_0.001/dfm_epoch_100.pt'
RECOMMENDED_MARKET_COUNT = 10
VERBOSE = 50
NUM_WORKER = 4

DATASET_MARKET_COUNT = 75 # MARKET COUNT
TRAIN_BATCH_SIZE = 2048
TEST_BATCH_SIZE = DATASET_MARKET_COUNT
EPOCH = 150
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-6

#DFM PARAMS
EMBEDDED_DIMENSIONS = 75
MLP_HIDDEN_LAYER_NEURON_SIZES = (128, 64, 32, 32)
DROPOUT = 0.2

def print_loss(path):
    df = pd.read_csv(path)
    values = df.values
    epoch = values[2:, 0]
    loss = values[2:, 1]
    plt.title('MODEL LOSS')
    plt.xlabel('EPOCHS')
    plt.ylabel('MEAN SQUARE ERROR')
    plt.xticks(np.arange(0, 150, 10))
    plt.plot(epoch, loss, '-')
    plt.show()

def print_acc(path):
    values = pd.read_csv(path).values
    recommendation_count = values[:, 0]
    hit = values[:, 1]
    plt.title('Increase in accuracy by number of predictions')
    plt.xticks(np.arange(0, 71))
    plt.yticks(np.arange(0, 101, 5))
    plt.vlines(recommendation_count, 0, hit, linestyles='dashed', colors='gray')
    plt.hlines(hit, 0, recommendation_count, linestyles='dashed', colors='gray')
    plt.plot(recommendation_count, hit, '.')
    plt.show()


def get_dataset(name, path, train):
    if name == 'Market_User':
        return Market_User(path, train=train)
    else:
        raise ValueError(f'unknown dataset name: {name}')


def get_model(name, dataset):
    """
    Define your model and create instance if you want
    """
    field_dims = dataset.field_dims
    if name == 'dfm':
        return DeepFactorizationMachineModel(field_dims,
                                             embed_dim=EMBEDDED_DIMENSIONS,
                                             mlp_dims=MLP_HIDDEN_LAYER_NEURON_SIZES,
                                             dropout=DROPOUT)
    else:
        raise ValueError('unknown model name: ' + name)


class EarlyStopper(object):
    """
    Simple early stopper for creating checkpoint
    """
    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def train(model, optimizer, data_loader, criterion, device, epoch, log_interval=100):
    model.train()
    total_loss = 0
    data_provider = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(data_provider):
        # load features and targets to gpu
        fields, target = fields.to(device), target.to(device)
        # predict
        y = model(fields)
        # calculate loss for predictions
        loss = criterion(y, target.float())
        # set model to zero grad
        model.zero_grad()
        # backpropagation for weight adjusting
        loss.backward()
        optimizer.step()
        # add loss to total loss and update loss output
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            data_provider.set_postfix(loss=total_loss / log_interval)
            with open('../result/train_loss_2.txt', 'a') as result_txt:
                _result = f'Epoch : {epoch} - Loss : {total_loss / log_interval}'
                result_txt.write(_result + '\n')
            total_loss = 0


def calculate_mse_loss(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            # load features and targets to gpu
            fields, target = fields.to(device), target.to(device)
            # predict
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return mean_squared_error(targets, predicts)


def test_hit_ratio(model, data_loader, device, dataset_market_count, recommended_market_count=10):
    model.eval()
    acc = 0
    sample = 0
    with torch.no_grad():
        for fields, target, visited in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            #load features and targets to gpu
            fields, target = fields.to(device), target.to(device)
            #predict
            y = model(fields)
            #get predictions to cpu
            targets = np.array(target.to('cpu'))
            predictions = np.array(y.to('cpu'))
            visited_market = np.array(visited.to('cpu'))

            #create mask array and give 0 score for visited markets
            mask = np.ones(shape=(dataset_market_count, 1), dtype='int')
            for visit in visited_market[0]:
                if visit != -1:
                    mask[visit] = 0

            #mask visited markets scores
            predictions = predictions * mask.reshape(1, -1)
            #get top n scored market for prediction
            max_rated = np.argsort(-predictions)[0][:recommended_market_count]
            #if top scored markets contains target market this means we got hit
            if targets[0] in max_rated:
                acc += 1
            sample +=1
    return acc * 100 / sample


def main(dataset_name, dataset_market_count, train_dataset_path, test_dataset_path, model_name, num_worker,
         epoch, learning_rate, train_batch_size, test_batch_size, weight_decay, device,
         recommended_market_count, save_dir, verbose):

    #create save dir for pretrained model
    model_save_dir = f'{save_dir}_lr_{learning_rate}/{model_name}_epoch_{epoch}.pt'

    #load the dataset
    device = torch.device(device)
    dataset_train = get_dataset(dataset_name, train_dataset_path, train=True)
    dataset_test = get_dataset(dataset_name, test_dataset_path, train=False)


    train_data_loader = DataLoader(dataset_train, batch_size=train_batch_size, num_workers=num_worker)
    valid_data_loader = DataLoader(dataset_test, batch_size=test_batch_size, num_workers=num_worker)

    #create model instance and define loss and optimizer
    model = get_model(model_name, dataset_train).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #train process
    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, criterion, device, epoch_i)
        '''if epoch_i != 0 and epoch_i % verbose == 0:
            acc = test_hit_ratio(model, valid_data_loader, device, dataset_market_count, recommended_market_count)
            print(f'\nepoch: {epoch_i} - Test {recommended_market_count} score : {acc}:')
            torch.save(model.state_dict(),
                       f'{save_dir}_lr_{learning_rate}/{model_name}_epoch_{epoch_i}.pt')'''

    #evaluate and save model
    acc = test_hit_ratio(model, valid_data_loader, device, dataset_market_count, recommended_market_count)
    print(f'Test {recommended_market_count} score : {acc}')
    torch.save(model.state_dict(), model_save_dir)


def predict(model_name, device, dataset_name, train_dataset_path, test_dataset_path,
            test_batch_size, pretrained_model_path, num_worker, dataset_market_count, recommended_market_count):
    #load the dataset
    dataset_train = get_dataset(dataset_name, train_dataset_path, train=True)
    dataset_test = get_dataset(dataset_name, test_dataset_path, train=False)
    test_data_loader = DataLoader(dataset_test, batch_size=test_batch_size, num_workers=num_worker)

    #get pretrained model
    model = get_model(model_name, dataset_train).to(device)
    model.load_state_dict(torch.load(pretrained_model_path))

    with open('../result/dfm_results.txt', 'a') as result_txt:
        recommendation_count = 1
        acc = 0
        while recommendation_count < 75 and acc < 100.0:
            acc = test_hit_ratio(model, test_data_loader, device, dataset_market_count, recommendation_count)
            result = f'{recommendation_count},{acc}'
            result_txt.write(result + '\n')
            recommendation_count += 1
    '''#evaluate
    acc_valid = test_hit_ratio(model, test_data_loader, device, dataset_market_count, recommended_market_count)
    print(f'\nTest {recommended_market_count} score: {acc_valid}\n')

    acc_valid = test_hit_ratio(model, test_data_loader, device, dataset_market_count, recommended_market_count + 5)
    print(f'\nTest {recommended_market_count + 5} score: {acc_valid}\n')

    acc_valid = test_hit_ratio(model, test_data_loader, device, dataset_market_count, recommended_market_count + 10)
    print(f'\nTest {recommended_market_count + 10} score: {acc_valid}\n')
    return acc_valid'''


if __name__ == '__main__':
    if TRAIN_MODE:
        main(DATASET_NAME, DATASET_MARKET_COUNT, DATASET_PATH_TRAIN, DATASET_PATH_TEST, MODEL_NAME, NUM_WORKER,
             EPOCH, LEARNING_RATE, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, WEIGHT_DECAY, DEVICE,
             RECOMMENDED_MARKET_COUNT, MODEL_SAVE_DIR, VERBOSE)
    else:
        predict(MODEL_NAME, DEVICE, DATASET_NAME, DATASET_PATH_TRAIN, DATASET_PATH_TEST, TEST_BATCH_SIZE,
                PRETRAINED_MODEL_PATH, NUM_WORKER, DATASET_MARKET_COUNT, RECOMMENDED_MARKET_COUNT)