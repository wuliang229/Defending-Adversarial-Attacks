# <---- need to put the model defination input the models_mnist.py file ---->
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.datasets as dset
import torchvision.transforms as transforms
from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import _imshow
from advertorch.attacks import JacobianSaliencyMapAttack

from tqdm import tqdm

from models_mnist import *

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def generate_attack_samples(model, cln_data, true_labels):
    '''N. Papernot, P. McDaniel, S. Jha, M. Fredrikson, Z. B. Celik, and
A. Swami, “The limitations of deep learning in adversarial settings,”
in Proceedings of the 1st IEEE European Symposium on Security and
Privacy. IEEE, 2016.'''
    num_classes = 10
    adversary = JacobianSaliencyMapAttack(model,
                                          num_classes=num_classes,
                                          clip_min=0,
                                          clip_max=1,
                                          loss_fn=None,
                                          theta=1,
                                          gamma=0.145,  # according to the paper
                                          comply_cleverhans=False)

    adv_targeted_results = []
    adv_target_labels = []
    avg_distortion_rate = 0.0
    min_distortion_rate = 1.0
    max_distortion_rate = 0.0
    loader = tqdm(range(num_classes), total=len(range(num_classes)))
    for target_label in loader:
        assert target_label >= 0 and target_label <= 10 and type(
            target_label) == int
        target = torch.ones_like(true_labels) * target_label
        adv_targeted = adversary.perturb(cln_data, target)
        # distortion rate
        distortion_rates = [1 - torch.eq(origin_sample, adv_sample).sum().float() / origin_sample.numel() for
                            origin_sample, adv_sample in zip(cln_data, adv_targeted)]
        
        # filter out equal labels
        filtered_rates = np.array(distortion_rates)[np.array(distortion_rates) != 0.0] # distortion rate == 0 if target is the same with true
        avg_distortion_rate += sum(filtered_rates) / len(filtered_rates)
        min_distortion_rate = min(min(filtered_rates), min_distortion_rate)
        max_distortion_rate = max(max(filtered_rates), max_distortion_rate)
        adv_targeted_results.extend(adv_targeted)
        adv_target_labels.extend(target)

    avg_distortion_rate /= num_classes

    return adv_targeted_results, adv_target_labels, min_distortion_rate, max_distortion_rate, avg_distortion_rate


# paths
model_name = sys.argv[1]
model_path = os.path.join('..', 'model', model_name)
output_path = os.path.join('..', 'result', model_name + '_metrics_JSM.txt')
print(model_path)

# load model
model = torch.load(model_path, map_location=device)
model_2 = None
if len(sys.argv) > 2:
    test_model_name = sys.argv[2]
    model_2_path = os.path.join('..', 'model', test_model_name)
    model_2 = torch.load(model_2_path, map_location=device)
    print('testing results on ' + model_2_path)

# generate attack samples
batch_size = 25
# dataset
root = '../data'
if not os.path.exists(root):
    os.mkdir(root)

test_set = dset.MNIST(root=root, train=False, transform=transforms.ToTensor(), download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

idx = 0
adv_targeted_results, adv_target_labels = [], []
min_distortion_rate, max_distortion_rate, avg_distortion_rate = 1, 0, 0

for cln_data, true_labels in test_loader:
    if idx == 4:
        break
    cln_data, true_labels = cln_data.to(device), true_labels.to(device)
    if model_2 is not None:
        adv_targeted_results_batch, adv_target_labels_batch, min_distortion_rate_batch, max_distortion_rate_batch, avg_distortion_rate_batch = generate_attack_samples(
            model_2, cln_data, true_labels)
    else:
        adv_targeted_results_batch, adv_target_labels_batch, min_distortion_rate_batch, max_distortion_rate_batch, avg_distortion_rate_batch = generate_attack_samples(
            model, cln_data, true_labels)
    adv_targeted_results.extend(adv_targeted_results_batch)
    adv_target_labels.extend(adv_target_labels_batch)
    min_distortion_rate = min(min_distortion_rate, min_distortion_rate_batch)
    max_distortion_rate = max(max_distortion_rate, max_distortion_rate_batch)
    avg_distortion_rate += avg_distortion_rate_batch

    idx += 1

avg_distortion_rate /= 4 # 4 x 250 = 1000
print(len(adv_targeted_results), adv_targeted_results[0])

defense_cln_acc = 0.0
defense_acc = 0.0
defense_rate = 0.0
attack_rate = 0.0

pred_cln = predict_from_logits(model(cln_data))

with torch.no_grad():
    loader = tqdm(range(len(adv_targeted_results)), total=len(range(len(adv_targeted_results))))
    for targeted_label in loader:
        # make sure label index equals to adv target label
        assert targeted_label == adv_target_labels[targeted_label][0]
        for pred_label, adv_result, true_label in zip(pred_cln, adv_targeted_results[targeted_label], true_labels):
            if true_label == targeted_label:
                continue
            print(adv_result.is_cuda)
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

with open(output_path, 'w') as f:
    f.write(
        'acc_before_attack %.4f | acc_after_attack %.4f | percentage_unchange %.4f percentage_successful_attack %.4f \n' % (
            defense_cln_acc, defense_acc, defense_rate, attack_rate))
    f.write('min_distortion_rate %.4f | max_distortion_rate %.4f | avg_distortion_rate %.4f' % (
        min_distortion_rate, max_distortion_rate, avg_distortion_rate))
