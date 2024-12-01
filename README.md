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


## ImageNet

### Prepare the data and models

Please download the [data](https://drive.google.com/drive/folders/1CfobY6i8BfqfWPHL31FKFDipNjqWwAhS) and [pretrained models](https://drive.google.com/file/d/1Oj4-IrZMPDhY-tXmDfmdegBFJPbzplCr/view?usp=drive_link), unzip the file and place the data and pretrained models under ./ImageNet/. The directory should be like:

```
ImageNet
+-- val_rs
+-- val_rs.csv
+-- checkpoints_ImageNet
```

### Running the attack

Enter the ImageNet directory:

```
cd ./ImageNet
```

Train the QAA substitute model:
```
python -m torch.distributed.launch --nproc_per_node=6 train.py --data /data/yyl/data/ImageNet/ --stochastic 1 --ckpt_id 120603
```

Generate adversarial examples with the QAA substitute model and the MI-FGSM attack:
```
python attack.py --data_source /data/yyl/data/ImageNet/val_rs --arch resnet34 --attack admix --w_bit 2 --a_bit 2 --quantize_method apot --stochastic 1 --ckpt_id 120603 --device 0
```

Evaluate the attack success rates on Inception_v3 target model:
```
python evaluate.py --arch inception_v3 --output_dir ./adv_imgs/apot/resnet34_w2a2_stochastic/admix --device 0
```


## CIFAR-10

### Prepare the data and models

Please download the [pretrained models](https://drive.google.com/file/d/1Oj4-IrZMPDhY-tXmDfmdegBFJPbzplCr/view?usp=drive_link) and place them under ./ImageNet/, respectively. The CIFAR-10 dataset will be downloaded automatically when running the code. The directory structure should be like:

```
CIFAR-10
+-- checkpoints_CIFAR-10
```

### Running the attack

Enter the CIFAR-10 directory:

```
cd ./CIFAR-10
```

Train the QAA substitute model:
```
python train.py --stochastic 1 --ckpt_id 120603 --device 0 
```

Generate adversarial examples with the QAA substitute model and the MI-FGSM attack:
```
python attack.py --attack pgd --arch resnet56 --w_bit 2 --a_bit 2 --quantize_method apot --ckpt_name 120603 --stochastic 1 --device 0
```

Or you can run the following script to evaluate on multiple target models:
```
bash ./fp_benchmark.sh
```

## Acknowledgments
Our code refers to:

[VT](https://github.com/JHL-HUST/VT)

[QDrop](https://github.com/wimh966/QDrop)

[APoT_Quantization](https://github.com/yhhhli/APoT_Quantization)

[PyTorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10)


## About us
We are in XJTU-AISEC lab led by [Prof. Chao Shen](https://gr.xjtu.edu.cn/en/web/cshen/home), [Prof. Chenhao Lin](https://gr.xjtu.edu.cn/en/web/linchenhao), [Prof. Zhengyu Zhao](https://zhengyuzhao.github.io/), Prof. Qian Li, and etc. in the School of Cyber Science and Engineering, Xi'an Jiaotong University.

Please contact Yulong Yang at xjtu2018yyl0808@stu.xjtu.edu.cn if you have any question on the codes. If you find this repository useful, please consider giving ⭐.
