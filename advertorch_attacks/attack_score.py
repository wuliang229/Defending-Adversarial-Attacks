# <---- need to put the model defination input the models.py file ---->
import torch
import torch.nn as nn
import torch.nn.functional as F

from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import _imshow
from advertorch.attacks import LinfPGDAttack
import sys
from models import *

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def generate_attack_samples(model, cln_data, true_label):
    adversary = LinfPGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.25,
        nb_iter=10, eps_iter=0.15, rand_init=True, clip_min=0.0, clip_max=1.0,
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


# paths
model_path = sys.argv[1]
output_path = model_path + '_metrics.txt'
print(model_path)
# load model
model = torch.load(model_path, map_location='cpu')


# generate attack samples
batch_size = 100
loader = get_mnist_test_loader(batch_size=batch_size)
for cln_data, true_label in loader:
    break
cln_data, true_labels = cln_data.to(device), true_label.to(device)


adv_targeted_results, adv_target_labels, adv_untargeted = generate_attack_samples(
    model, cln_data, true_label)

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

with open(output_path, 'w') as f:
    f.write('cln_acc %.4f | defense_acc %.4f | defense_rate %.4f attack_rate %.4f' % (
        defense_cln_acc, defense_acc, defense_rate, attack_rate))
