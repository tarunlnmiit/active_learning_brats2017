# Python libraries
import argparse
import csv
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import lib.medloaders as medical_loaders
import lib.medzoo as medzoo
import lib.train as train
# Lib files
import lib.utils as utils
from lib.losses3D import DiceLoss, BCEWithLogitsLossPadding, BCEDiceLoss

import lib.active_learning.committee as al

from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint, TrainEndCheckpoint, EarlyStopping

from modAL.models import ActiveLearner
from modAL.disagreement import KL_max_disagreement, max_disagreement_sampling

from sklearn.metrics import average_precision_score
from statistics import mean, mode


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 1777777

device = "cuda" if torch.cuda.is_available() else "cpu"
                                                                                           

def main():
    args = get_arguments()

    utils.reproducibility(args, seed)
    utils.make_dirs(args.save)

    models = ['VNET', 'UNET3D', 'VNET2']
    classifiers = {}

    training_generator, val_generator = medical_loaders.generate_datasets(args,
                                                                                               path='/content/drive/MyDrive/Colab Notebooks/active_learning_brats2017/datasets')
    
    print(len(training_generator))
    X_train, y_train= next(iter(training_generator))
    X_val, y_val = next(iter(val_generator))

    print('0', X_train.shape,y_train.shape)
    print('1', X_val.shape,y_val.shape)

    # output_final = []
    # for item in y_train:
    #     unique_label = torch.from_numpy(np.where(item > 1, 0, item))
    #     # print(unique_label.shape)
    #     output_final.append(unique_label)
    # result = torch.stack(output_final, dim=0)  
    # print(np.unique(result[0]))
    # print(result.shape)

    # val_final = []
    # for item in y_val:
    #     unique_label = torch.from_numpy(np.where(item > 1, 0, item))
    #     # print(unique_label.shape)
    #     val_final.append(unique_label)
    # result_val = torch.stack(val_final, dim=0)  
    # print(np.unique(result_val[0]))
    # print(result_val.shape)

    # y_train = result
    # y_val = result_val


    # temp = []
    # for item in X_val:
    #     # print(len(item))
    #     for i in range(0, len(item), 4):
    #         temp2 = item[i:i+4]
    #         # print(temp2.shape)
    #         temp.append(torch.Tensor(temp2))

    # temp = torch.stack(temp)
    # print('2', temp.shape)

    # X_val = temp

    temp = []
    for item in y_val:
        # print(item.shape)
        temp2 = torch.stack([item] * 2)
        # print(temp2.shape)
        temp.append(temp2)

    temp = torch.stack(temp)
    print('3', temp.shape)
    y_val = temp
    # X_val = X_val.reshape(32,1,32, 32, 32)
    # y_val= y_val.reshape(32,2,32, 32, 32)
    # print(X_val.shape,y_val.shape)

    # # print(X_train.shape,y_train.shape)

    # temp = []
    # for item in X_train:
    #     # print(len(item))
    #     for i in range(0, len(item), 4):
    #         temp2 = item[i:i+4]
    #         # print(temp2.shape)
    #         temp.append(torch.Tensor(temp2))

    # temp = torch.stack(temp)
    # # print(temp.shape)
    # # for item in y_train:
    # #     print(len(item), item.shape)
    # X_train = temp
    # X_train = torch.squeeze(X_train, 2)

    temp = []
    for item in y_train:
        # print(item.shape)
        temp2 = torch.stack([item] * 2)
        # print(temp2.shape)
        temp.append(temp2)

    temp = torch.stack(temp)
    print('4', temp.shape)
    y_train = temp

    print('5', X_train.shape,y_train.shape)

    # X_train = X_train.reshape(args.batchSz,1,32,32,32)
    # y_train= y_train.reshape(args.batchSz,2,32,32,32)

    X_val = X_val.reshape(128,1,32,32,32)
    # y_val = y_val.reshape(128,2,32,32,32)

    # print(len(X_train))

    n_initial = args.batchSz//2
    initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
    X_initial = X_train[initial_idx]
    y_initial = y_train[initial_idx]
    print('6', np.unique(y_initial))
    # y_initial = np.where(y_initial < 3.0, np.float(0.0), np.float(3.0))

    # print(X_train.shape,y_train.shape)
    X_pool = np.delete(X_train, initial_idx, axis=0)
    y_pool = np.delete(y_train, initial_idx, axis=0)
    # y_pool = np.where(y_pool < 4.0, 0.0, 4.0)
    print('7', X_pool.shape,y_pool.shape)
    print('8', X_initial.shape,y_initial.shape)

    cp = Checkpoint(dirname='exp1')
    train_end_cp = TrainEndCheckpoint(dirname='exp1')
    monitor = lambda net: all(net.history[-1, ('train_loss_best')])
    es = EarlyStopping('train_loss')
    for model_name in models:
        args.model = model_name
        args.classes = 2
        args.inChannels = 1
        model, optimizer = medzoo.create_model(args)
        # criterion = DiceLoss(classes=args.classes)  # ,skip_index_after=2,weight=torch.tensor([0.00001,1,1,1]).cuda())
        criterion = CustomDiceLoss
        optimizer = torch.optim.Adam

        if args.cuda:
            model = model.cuda()
            print("Model transferred in GPU.....")

        classifier = NeuralNetClassifier(model,
                                 max_epochs=args.nEpochs,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 train_split=None,
                                 verbose=1,
                                 device=device,
                                #  callbacks=[cp, train_end_cp]
                                 )

        # classifier.fit(train_loader)

        classifiers[model_name] = classifier


    learners = {}
    for k, v in classifiers.items():
        learners[k] = ActiveLearner(
            estimator=v,
            X_training=np.array(X_initial), y_training=np.array(y_initial),)

    committee = al.CustomCommittee(learner_list=list(learners.values()))

    no_querry = args.nQuery
    random_result = {}
    KLD_result = {}
    JSD_result = {}
    entropy_result = {}

    print('Length of Pool before teaching: ', len(X_pool))

    with open('results/results-{}-{}.csv'.format(no_querry, args.nEpochs), 'w', newline='') as file:
      fieldnames = ['type', 'indexes', 'avg_precision_committee', "avg_list" ]
      writer = csv.DictWriter(file, fieldnames=fieldnames)
      writer.writeheader()
      
      for _ in range(int((len(X_pool)/no_querry)) + 1):
          if no_querry > len(X_pool):
            no_querry = len(X_pool)
          print('Length of Pool entropy: ', len(X_pool))
          indexes, _ = al.consensus_entropy_sampling_custom(committee,  X_pool[:], no_querry)
          X_initial, y_initial, X_pool, y_pool, committee = teach_model(X_pool,
                                                          y_pool,  X_initial, y_initial, 
                                                          indexes, classifiers)
          avg_precision_committee, avg_list = average_precision(committee, X_val, y_val)
          entropy_result['type'] = 'entropy'
          entropy_result["indexes"]= indexes
          entropy_result["avg_precision_committee"] = avg_precision_committee
          entropy_result["avg_list"] = avg_list
          writer.writerow(entropy_result)
          # X_pool = np.delete(X_pool, indexes, axis=0)
          # y_pool = np.delete(y_pool, indexes, axis=0)

      X_initial = X_train[initial_idx]
      y_initial = y_train[initial_idx]
      X_pool = np.delete(X_train, initial_idx, axis=0)
      y_pool = np.delete(y_train, initial_idx, axis=0)
      no_querry = args.nQuery

      for _ in range(int((len(X_pool) / no_querry)) + 1):
          if no_querry > len(X_pool):
            no_querry = len(X_pool)
          print('Length of Pool KLD: ', len(X_pool))
          indexes, _ = al.KL_max_disagreement_sampling_custom(committee, X_pool[:], no_querry)
          X_initial, y_initial, X_pool, y_pool, committee = teach_model(X_pool,
                                                          y_pool,  X_initial, y_initial, 
                                                          indexes, classifiers)
          avg_precision_committee, avg_list  = average_precision(committee, X_val, y_val)
          KLD_result['type'] = 'KLD'
          KLD_result["indexes"]= indexes
          KLD_result["avg_precision_committee"] = avg_precision_committee
          KLD_result["avg_list"] = avg_list
          writer.writerow(KLD_result)
          # X_pool = np.delete(X_pool, indexes, axis=0)
          # y_pool = np.delete(y_pool, indexes, axis=0)

      X_initial = X_train[initial_idx]
      y_initial = y_train[initial_idx]
      X_pool = np.delete(X_train, initial_idx, axis=0)
      y_pool = np.delete(y_train, initial_idx, axis=0)
      no_querry = args.nQuery

      for _ in range(int((len(X_pool) / no_querry)) + 1):
          if no_querry > len(X_pool):
            no_querry = len(X_pool)
          print('Length of Pool JSD: ', len(X_pool))
          indexes, _ = al.JSD_max_disagreement_sampling(committee, X_pool[:], no_querry)
          X_initial, y_initial, X_pool, y_pool, committee = teach_model(X_pool,
                                                          y_pool,  X_initial, y_initial, 
                                                          indexes, classifiers)
          avg_precision_committee, avg_list  = average_precision(committee, X_val, y_val)
          JSD_result['type'] = 'JSD'
          JSD_result["indexes"]= indexes
          JSD_result["avg_precision_committee"] = avg_precision_committee
          JSD_result["avg_list"] = avg_list
          writer.writerow(JSD_result)
          # X_pool = np.delete(X_pool, indexes, axis=0)
          # y_pool = np.delete(y_pool, indexes, axis=0)

      X_initial = X_train[initial_idx]
      y_initial = y_train[initial_idx]
      X_pool = np.delete(X_train, initial_idx, axis=0)
      y_pool = np.delete(y_train, initial_idx, axis=0)
      no_querry = args.nQuery

      for _ in range(int((len(X_pool) / no_querry)) + 1):
          if no_querry > len(X_pool):
            no_querry = len(X_pool)
          print('Length of Pool Random: ', len(X_pool))
          indexes = random.sample(range(X_pool.shape[0]), no_querry)
          X_initial, y_initial, X_pool, y_pool, committee = teach_model(X_pool,
                                                          y_pool,  X_initial, y_initial, 
                                                          indexes, classifiers)
          avg_precision_committee, avg_list = average_precision(committee, X_val, y_val)
          random_result['type'] = 'random'
          random_result["indexes"]= indexes
          random_result["avg_precision_committee"] = avg_precision_committee
          random_result["avg_list"] = avg_list
          writer.writerow(random_result)
          # X_pool = np.delete(X_pool, indexes, axis=0)
          # y_pool = np.delete(y_pool, indexes, axis=0) 

    # indexes, samples  = al.JSD_max_disagreement_sampling(committee, X_pool[:], 20) 
    # print("JSD\n", indexes, samples.shape)

    # indexes, samples  = al.KL_max_disagreement_sampling_custom(committee, X_pool[:], 20) 
    # print("KLD\n", indexes, samples.shape)

    # indexes, samples  = al.consensus_entropy_sampling_custom(committee, X_pool[:], 20) 
    # print("Consensus\n", indexes, samples.shape)
    
    
    


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class CustomDiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(CustomDiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]


def average_precision(committee, X_pool, y_pool):
    avg_list = []
    
    target = y_pool[:][:, 0,  :, :]
    
    for learner in committee:
        pred = learner.predict(X_pool[:])

        avg_precision = average_precision_score(np.array(target.reshape(y_pool.shape[0]*32**3)), 
                                                np.array(pred.reshape(X_pool.shape[0]*32**3)), 
                                                pos_label=0.0) 
        avg_list.append(avg_precision)
  
    pred_committee = committee.predict(X_pool[:])
    avg_precision_committee = average_precision_score(np.array(target.reshape(y_pool.shape[0]*32**3)), 
                                            np.array(pred_committee.reshape(X_pool.shape[0]*32**3)), 
                                            pos_label=0.0)

    return avg_precision_committee, avg_list


def teach_model(X_pool, y_pool, X_initial, y_initial, n_instances,
                classifiers):
    """
    Retrains Committe with new samples.

    Args:
        committee: The committee for which the labels are to be queried.
        X_pool & y_pool: The pool of samples.
        n_instances: Number of samples to be queried.


    Returns:
        New updated Initial samples, new pool of samples and committee;
    """
    learner_list = []
    for idx in n_instances:
        X_initial = np.append(X_initial, X_pool[idx].reshape(1,1,32,32,32), axis=0)
        y_initial = np.append(y_initial, y_pool[idx].reshape(1,2,32,32,32), axis=0)

    learners = {}
    for k, v in classifiers.items():
        learners[k] = ActiveLearner(
            estimator=v,
            X_training=np.array(X_initial), y_training=np.array(y_initial),)

    committee = al.CustomCommittee(learner_list=list(learners.values()))

    X_pool = np.delete(X_pool, n_instances, axis=0)
    y_pool = np.delete(y_pool, n_instances, axis=0)
    return X_initial, y_initial, X_pool, y_pool, committee


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=4)
    parser.add_argument('--dataset_name', type=str, default="brats2017")
    parser.add_argument('--dim', nargs="+", type=int, default=(64, 64, 64))
    parser.add_argument('--nEpochs', type=int, default=100)
    parser.add_argument('--nQuery', type=int, default=200)
    parser.add_argument('--classes', type=int, default=4)
    parser.add_argument('--samples_train', type=int, default=1024)
    parser.add_argument('--samples_val', type=int, default=128)
    parser.add_argument('--inChannels', type=int, default=4)
    parser.add_argument('--inModalities', type=int, default=4)
    parser.add_argument('--threshold', default=0.00000000001, type=float)
    parser.add_argument('--terminal_show_freq', default=50)
    parser.add_argument('--augmentation', action='store_true', default=True)
    parser.add_argument('--normalization', default='full_volume_mean', type=str,
                        help='Tensor normalization: options ,max_min,',
                        choices=('max_min', 'full_volume_mean', 'brats', 'max', 'mean'))
    parser.add_argument('--split', default=0.8, type=float, help='Select percentage of training data(default: 0.8)')
    parser.add_argument('--lr', default=1e-2, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--loadData', default=False)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='VNET',
                        choices=("RESNET3DVAE",'UNET3D',  'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET', "SKIPDENSENET3D",
                  "DENSEVOXELNET",'VNET','VNET2'))
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--log_dir', type=str,
                        default='../runs/')

    args = parser.parse_args()

    args.save = '../saved_models/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(
        utils.datestr(), args.dataset_name)
    return args

if __name__ == '__main__':
    main()
