# Siren: A Proactive Attack-agnostic Defense System for Federated Learning

This code accompanies the paper 'Siren: A Proactive Attack-agnostic Defense System for Federated Learning'. Please download Fashion-MNIST dataset to /home/data/ on the user's machine if you want to use it.

Recommended Dependencies: Python-3.5, Tensorflow-1.8, keras, numpy, scipy, scikit-learn. It is possible for our codes to run with newer versions of Tensorflow and Python, but we didn't test it thoroughly.

To run the code, please use:

```english
python main.py
```

While if you want to run the code successfully, you also need to set the following hyper-parameters:

| Parameter   | Function                                               |
| ----------- | ------------------------------------------------------ |
| --gar       | Gradient Aggregation Rule                              |
| --eta       | Learning Rate                                          |
| --k         | Number of agents                                       |
| --C         | Fraction of agents chosen per time step                |
| --E         | Number of epochs for each agent                        |
| --T         | Total number of iterations                             |
| --B         | Batch size at each agent                               |
| --mal_obj   | Single or multiple targets                             |
| --mal_num   | Number of targets                                      |
| --mal_strat | Strategy to follow                                     |
| --mal_boost | Boosting factor                                        |
| --mal_E     | Number of epochs for malicious agent                   |
| --ls        | Ratio of benign to malicious steps in alt. min. attack |
| --rho       | Weighting factor for distance constraint               |
| --attack_type| attack type of malicious clients                      |
| --malicious_proportion| the proportion of malicious clients in the system|
| --non_iidness| the non-iidness of the data distribution on the clients |

* The options of '--attack_type' are: 'sign_flipping', 'label_flipping', 'targeted_model_poisoning', 'stealthy_model_poisoning' and 'none'. We only support four kinds of attacks now.
* The input of '--malicious_proportion' should be a float number between 0 and 1. And this parameter only works when the attack type is not model poisoning.
* The input of '--non_iidness' should also be a float number between 0 and 1. And please use this parameter only when the number of clients in the system is an integral multiple of 10.

If you want to use the same settings as us, here are some examples:

To run federated training with 10 agents and averaging based aggregation with Fashion-MNIST dataset, use

```english
python main.py --dataset=fMNIST --k=10 --C=1.0 --E=5 --T=40 --B=64 --train --model_num=1 --gar=avg
```
While if you want to use CIFAR-10 dataset, please set --dataset=CIFAR-10 and --model_num=0.

To run Siren under single-target targeted model poisoning attack with Fashion-MNIST dataset, use

```
python main.py --dataset=fMNIST --k=10 --C=1.0 --E=5 --T=40 --B=64 --train --model_num=1 --mal --mal_obj=single --gar=siren --attack_type=targeted_model_poisoning
```

After running the code, please check 'output' directory for the results.