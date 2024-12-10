# HAM
This is the official code repository for the paper "Hard Adversarial Example Mining for Improving Robust Fairness". This paper has been accepted by IEEE TIFS 2024.

Adversarial training (AT) is widely considered the
state-of-the-art technique for improving the robustness of deep
neural networks (DNNs) against adversarial examples (AEs).
Nevertheless, recent studies have revealed that adversarially
trained models are prone to unfairness problems. Recent works
in this field usually apply class-wise regularization methods to
enhance the fairness of AT. However, this paper discovers that
these paradigms can be sub-optimal in improving robust fairness.
Specifically, we empirically observe that the AEs that are already
robust (referred to as “easy AEs” in this paper) are useless
and even harmful in improving robust fairness. To this end, we
propose the hard adversarial example mining (HAM) technique
which concentrates on mining hard AEs while discarding the easy
AEs in AT. Specifically, HAM identifies the easy AEs and hard
AEs with a fast adversarial attack method. By discarding the
easy AEs and reweighting the hard AEs, the robust fairness of
the model can be efficiently and effectively improved. Extensive
experimental results on four image classification datasets demonstrate the improvement of HAM in robust fairness and training
efficiency compared to several state-of-the-art fair adversarial
training methods

![fig13_1](https://github.com/user-attachments/assets/b42c42ee-1db4-4773-a568-1d7afd21d808)

## Requirements

+ Python >= 3.8.0
+ torch >= 1.10.2
+ torchvision >= 0.11.3
+ numpy >= 1.20.3
+ pandas >= 1.4.1
+ PIL >= 9.3.0
+ robustbench
+ tqdm >= 4.62.3



## CIFAR-10

### Prepare the data and models

Please download the [pretrained models](https://drive.google.com/file/d/1jfgJvq-kuu2f-XB-3Z5ml98ngNJX7zUe/view?usp=drive_link) and place them under ./results/515_6_wrong_early3_epo50. The CIFAR-10 dataset will be downloaded automatically when running the code. The directory structure should be like:

```
results
+-- 515_6_wrong_early3_epo50
    +-- checkpoints_cifar10
```

### Running the HAM method on CIFAR-10

you can run the following script:
```
bash ./515_6_wrong_early3_epo50.sh
```

## CIFAR-100

### Prepare the data and models

Please download the [pretrained models](https://drive.google.com/file/d/1AhvShkc799QpT4I3LnINirieHudHWzrU/view?usp=drive_link) and place them under ./results/0726_2_cifar100_ham_keep_clean0d3_60epo. The CIFAR-100 dataset will be downloaded automatically when running the code. The directory structure should be like:

```
results
+-- 0726_2_cifar100_ham_keep_clean0d3_60epo
    +-- checkpoints_cifar100
```

### Running the HAM method on CIFAR-100

you can run the following script:
```
bash ./0726_2_cifar100_ham_keep_clean0d3_60epo.sh
```

## SVHN

### Prepare the data and models

Please download the [pretrained models](https://drive.google.com/file/d/1EyLnIk-UIPVVSsLz3B1jwxJxXFTji-Li/view?usp=drive_link) and place them under ./results/718_7_svhn_wrong_early5_epo50. The SVHN dataset will be downloaded automatically when running the code. The directory structure should be like:

```
results
+-- 718_7_svhn_wrong_early5_epo50
    +-- checkpoints_svhn
```

### Running the HAM method on SVHN

you can run the following script:
```
bash ./718_7_svhn_wrong_early5_epo50.sh
```


## ImageNette

### Prepare the data and models

Please download the [pretrained models](https://drive.google.com/file/d/1WqAzuXXU363-H2PS0ByycypvcbASMUp7/view?usp=drive_link) and place them under ./results/0301_1_aham_netee_p18_nonorm. The ImageNette dataset could be downloaded from [data](https://drive.google.com/file/d/1nyYlZFvpSRl_ogmaO0cm8-hMT4YMYe9D/view?usp=drive_link). The directory structure should be like:

```
imagenette2-160
+-- train
+-- val
```

```
results
+-- 0301_1_aham_netee_p18_nonorm
    +-- checkpoints_imagenette
```

### Running the HAM method on ImageNette

you can run the following script:
```
bash ./0301_1_aham_netee_p18_nonorm.sh
```


## About us
We are in XJTU-AISEC lab led by [Prof. Chao Shen](https://gr.xjtu.edu.cn/en/web/cshen/home), [Prof. Chenhao Lin](https://gr.xjtu.edu.cn/en/web/linchenhao), [Prof. Zhengyu Zhao](https://zhengyuzhao.github.io/), Prof. Qian Li, and etc. in the School of Cyber Science and Engineering, Xi'an Jiaotong University.

Please contact Yulong Yang, Xiang Ji and Ruiqi Cao at xjtu2018yyl0808@stu.xjtu.edu.cn, xiangji@stu.xjtu.edu.cn and crq2002@stu.xjtu.edu.cn if you have any question on the codes. If you find this repository useful, please consider giving ⭐.
