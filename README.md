# Siren: Byzantine-robust Federated Learning via Proactive Alarming

This code accompanies the paper 'SIREN: Byzantine-robust Federated Learning via Proactive Alarming', which is accepted by ACM SoCC 2021.

:star2:This is the Tensorflow version of Siren. For Pytorch version, please refer to [Siren_by_Pytorch](https://github.com/ln-y/Siren_by_Pytorch). Thanks @[ln-y](https://github.com/ln-y).

Please download Fashion-MNIST dataset to path like ```/home/data/``` on the user's machine if you want to use it. CIFAR-10 dataset can be downloaded by the program automatically.

Recommended Dependencies: Please check the ```requirements.txt``` file in ```/requirement```. We have tested our system using such setting.

To run the code without any customized settings, please use:

```english
python main.py
```

While if you want to run the code successfully with your customized parameters, you also need to set the following basic hyper-parameters:

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

and SIREN exclusive parameters (if you want to use SIREN):

| Parameter         | Function                                               |
| -----------       | ------------------------------------------------------ |
| --server_c        | threshold $C_s$ used by the server                     |
| --client_c        | threshold $C_s$ used by the client                     |
| --server_prohibit | threshold to trigger the penalty mechanism.            |
| --forgive         | the award value used by the award mechanism            |
| --root_size       | the size of the root test dataset                      |

For more parameters and details of the above parameters, please refer to ```\global_vars.py```

If you want to use the same settings as us, here are some examples:

To run federated training with 10 agents and averaging based aggregation with Fashion-MNIST dataset, use

```english
python main.py --dataset=fMNIST --k=10 --C=1.0 --E=5 --T=40 --B=64 --train --model_num=1 --gar=avg
```
While if you want to use CIFAR-10 dataset, please set --dataset=CIFAR-10 and --model_num=0.

To run SIREN under single-target targeted model poisoning attack with Fashion-MNIST dataset, use

```
python main.py --dataset=fMNIST --k=10 --C=1.0 --E=5 --T=40 --B=64 --train --model_num=1 --mal --mal_obj=single --gar=siren --attack_type=targeted_model_poisoning
```

After running the code, please check ```/output``` directory for the results (please manually create the ```output``` directory before the execution of the codes).

To cite our paper, please use the following BibTex:
```
@inproceedings{guo2021siren,
  title={Siren: Byzantine-robust Federated Learning via Proactive Alarming},
  author={Guo, Hanxi and Wang, Hao and Song, Tao and Hua, Yang and Lv, Zhangcheng and Jin, Xiulang and Xue, Zhengui and Ma, Ruhui and Guan, Haibing},
  booktitle={ACM Symposium on Cloud Computing (SoCC)},
  year={2021}
}
```
