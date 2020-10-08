"""Testing script for adaptive_algorithms."""
import numpy as np
from torch.utils.data import DataLoader

from adacvar.util.adaptive_algorithm import Exp3, Exp3Sampler
from adacvar.util.dataset import Dataset
from adacvar.util.learning_rate_decay import AdaGrad, Constant, RobbinsMonro

num_actions = 100
size = 10
alpha = size / num_actions

batch_size = 4
horizon = 1000

losses = np.linspace(0, 1, num_actions)


def _get_initial_eta(alg_name, scheduler_name, batch_size, alpha, horizon):
    eta = np.sqrt(np.log(1 / alpha))
    if alg_name == "exp3":
        eta *= np.sqrt(1 / alpha)

    if scheduler_name != "constant":
        horizon = 1

    return eta * np.sqrt(batch_size / horizon)


def _update_loss(cum_loss, losses_, indexes):
    for i in indexes:
        cum_loss += losses_[i]
    return cum_loss


def _update_count(cum_counts, indexes):
    for i in indexes:
        cum_counts[i] += 1

    return cum_counts


for scheduler_name, learning_rate_class in {
    "constant": Constant,
    "robbins_monro": RobbinsMonro,
    "ada_grad": AdaGrad,
}.items():
    alg_name = "exp3"
    for loader in [
        "direct",
        # 'data_loader'
    ]:
        eta = _get_initial_eta(alg_name, scheduler_name, batch_size, alpha, horizon)
        learning_rate_scheduler = learning_rate_class(eta, num_actions)
        counts = np.zeros(num_actions)
        incurred_loss = 0
        observed_loss = 0

        if loader == "direct":
            algorithm = Exp3(
                num_actions, size, learning_rate_scheduler, 0.01, eps=1e-12
            )
            for t in range(horizon):
                probs = algorithm.probabilities
                idx = np.random.choice(
                    num_actions, size=size, p=probs / np.sum(probs), replace=False
                )
                incurred_loss = _update_loss(incurred_loss, losses, idx)
                idx = np.random.choice(idx, size=batch_size, replace=True)
                observed_loss = _update_loss(observed_loss, losses, idx)
                # idx = np.random.choice(num_actions, size=batch_size,
                #                        p=probs/np.sum(probs), replace=True)
                algorithm.update(losses[idx], idx)
                # if (t+1) % num_actions == 0:
                algorithm.normalize()
                counts = _update_count(counts, idx)

        else:
            dataset = Dataset(np.zeros(num_actions), np.zeros(num_actions))
            sampler = Exp3Sampler(
                batch_size, num_actions, size, learning_rate_scheduler, 0.01, eps=1e-12
            )
            dataloader = DataLoader(dataset, batch_sampler=sampler)

            t = 0
            while t < horizon:
                for _, _, idx in dataloader:
                    t += 1
                    dataloader.batch_sampler.update(losses[idx], idx)
                    #

                    incurred_loss = _update_loss(incurred_loss, losses, idx)
                    counts = _update_count(counts, idx)

                    if (t + 1) % num_actions == 0:
                        dataloader.batch_sampler.normalize()

        print(
            scheduler_name, loader, observed_loss, incurred_loss, counts, np.sum(counts)
        )
