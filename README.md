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
