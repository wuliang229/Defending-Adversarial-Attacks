# <---- need to put the model defination input the models_mnist.py file ---->
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchvision.datasets as dset
import torchvision.transforms as transforms
from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import _imshow
from advertorch.attacks import LinfPGDAttack
import sys
from models_mnist import *

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def generate_attack_samples(model, cln_data, true_label, nb_iter, eps_iter):
    adversary = LinfPGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.25,
        nb_iter=nb_iter, eps_iter=eps_iter, rand_init=True, clip_min=0.0, clip_max=1.0,
        targeted=False)

    adv_untargeted = adversary.perturb(cln_data, true_label)

    adv_targeted_results = []
    adv_target_labels = []
    for target_label in range(0, 10):
        assert target_label >= 0 and target_label <= 10 and type(
            target_label) == int
        target = torch.ones_like(true_label) * target_label
        adversary.targeted = True
        adv_targeted = adversary.perturb(cln_data, target)
        adv_targeted_results.append(adv_targeted)
        adv_target_labels.append(target)

    return adv_targeted_results, adv_target_labels, adv_untargeted


def attack_one_model(model, nb_iter, eps_iter):
    # generate attack samples
    batch_size = 100
    # dataset
    root = '../data'
    if not os.path.exists(root):
        os.mkdir(root)
    test_set = dset.MNIST(root=root, train=False, transform=transforms.ToTensor(), download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    for cln_data, true_labels in test_loader:
        break
    cln_data, true_labels = cln_data.to(device), true_labels.to(device)

    if model_2 is not None:
        adv_targeted_results, adv_target_labels, adv_untargeted = generate_attack_samples(
            model_2, cln_data, true_labels, nb_iter, eps_iter)
    else:
        adv_targeted_results, adv_target_labels, adv_untargeted = generate_attack_samples(
            model, cln_data, true_labels, nb_iter, eps_iter)

    defense_cln_acc = 0.0
    defense_acc = 0.0
    defense_rate = 0.0
    attack_rate = 0.0

    pred_cln = predict_from_logits(model(cln_data))

    for targeted_label in range(len(adv_targeted_results)):
        # make sure label index equals to adv target label
        assert targeted_label == adv_target_labels[targeted_label][0]
        for pred_label, adv_result, true_label in zip(pred_cln, adv_targeted_results[targeted_label], true_labels):
            if true_label == targeted_label:
                continue
            pred_targeted_adv = predict_from_logits(model(adv_result.unsqueeze(0)))
            if pred_label == true_label:
                defense_cln_acc += 1
            if pred_targeted_adv == true_label:
                defense_acc += 1
            if pred_label == pred_targeted_adv:
                defense_rate += 1
            if pred_targeted_adv == targeted_label:
                attack_rate += 1

    defense_cln_acc /= 900
    defense_acc /= 900
    defense_rate /= 900
    attack_rate /= 900

    return defense_cln_acc, defense_acc, defense_rate, attack_rate

# paths
model_name = sys.argv[1]
model_path = os.path.join('..', 'model', model_name)
output_path = os.path.join('..', 'result', model_name + '_metrics.txt')
print(model_path)

# load model
model_1 = torch.load(model_path, map_location=device)
test_model_name = sys.argv[2]
model_2_path = os.path.join('..', 'model', test_model_name)
model_2 = torch.load(model_2_path, map_location=device)
print('model_1: %s | model_2: %s'%(model_path, model_2_path))

parameter_list = [
    {
        'nb_iter': 10,
        'eps_iter': 0.01
    },
    {
        'nb_iter': 10,
        'eps_iter': 0.05
    },
    {
        'nb_iter': 10,
        'eps_iter': 0.1
    },
    {
        'nb_iter': 10,
        'eps_iter': 0.15
    },
    {
        'nb_iter': 10,
        'eps_iter': 0.2
    },
    {
        'nb_iter': 5,
        'eps_iter': 0.01
    },
    {
        'nb_iter': 5,
        'eps_iter': 0.1
    },
    {
        'nb_iter': 5,
        'eps_iter': 0.15
    },
    {
        'nb_iter': 5,
        'eps_iter': 0.2
    },
    {
        'nb_iter': 5,
        'eps_iter': 0.25
    }
    
]
best_parameter_idx = None
distance = 0
for idx, parameters in enumerate(parameter_list):
    print('-- ', idx)
    results_1 = attack_one_model(model_1, parameters['nb_iter'], parameters['eps_iter'])
    results_2 = attack_one_model(model_2, parameters['nb_iter'], parameters['eps_iter'])
    if np.abs(results_1[1]-results_2[1]) > distance:
        best_parameter_idx = idx
    with open(output_path, 'a') as f:
        f.write('model_1: %s | model_2: %s'%(model_path, model_2_path))
        f.write(str(parameters)+'\n')
        f.write('acc_before_attack %.4f | acc_after_attack %.4f | percentage_unchange %.4f percentage_successful_attack %.4f\n' % results_1)
        f.write('acc_before_attack %.4f | acc_after_attack %.4f | percentage_unchange %.4f percentage_successful_attack %.4f\n' % results_2)
        f.write('-'*100+'\n')

print('-'*100)
print('Best Parameter', parameter_list[best_parameter_idx])
    
