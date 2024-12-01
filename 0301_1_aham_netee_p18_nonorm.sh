# CUDA_VISIBLE_DEVICES='1' python GAIRAT_logits.py --net resnet18 --out-dir ./results/415_1/ --begin-epoch-2 70 --begin-epoch-3 80 --begin-epoch-4 90 --w-logits-base1 1.0 --w-logits-base2 0.8 --w-logits-base3 0.6 --w-logits-base4 0.4 --resume ./results/412_1/58.pth.tar

# CUDA_VISIBLE_DEVICES='1' python GAIRAT_logits.py --net resnet18 --out-dir ./results/415_2/ --begin-epoch-2 80 --begin-epoch-3 100 --begin-epoch-4 120 --w-logits-base1 1.0 --w-logits-base2 0.8 --w-logits-base3 0.6 --w-logits-base4 0.4 --resume ./results/412_1/58.pth.tar

# CUDA_VISIBLE_DEVICES='5' python GAIRAT_margin1.py --net resnet18 --out-dir ./results/417_1/ --begin-epoch-2 75 --begin-epoch-3 90 --begin-epoch-4 105 --w-logits-base1 0.5 --w-logits-base2 0.4 --w-logits-base3 0.3 --w-logits-base4 0.2 --norm

# CUDA_VISIBLE_DEVICES='5' python GAIRAT_margin1.py --net resnet18 --out-dir ./results/417_2/ --begin-epoch-2 75 --begin-epoch-3 90 --begin-epoch-4 105 --w-logits-base1 0.25 --w-logits-base2 0.2 --w-logits-base3 0.15 --w-logits-base4 0.1 --norm
CUDA_VISIBLE_DEVICES="2" python GAIRAT_wrong_early_time_copy.py \
--dataset "netee" \
--net "preactresnet18" \
--out-dir "./results/0301_1_aham_netee_p18_nonorm/" \
--begin_epoch "50" \
--classify_step "3"

# CUDA_VISIBLE_DEVICES='5' python GAIRAT_logits.py --net resnet18 --out-dir ./results/415_5/ --begin-epoch-2 75 --begin-epoch-3 90 --begin-epoch-4 105 --w-logits-base1 0.75 --w-logits-base2 0.6 --w-logits-base3 0.45 --w-logits-base4 0.3 --resume ./results/412_1/58.pth.tar

# CUDA_VISIBLE_DEVICES='1' python GAIRAT.py --net resnet18 --out-dir ./results/gairat_res18_norm/

# CUDA_VISIBLE_DEVICES='0' python GAIRAT.py --net resnet18 --out-dir ./results/at_res18_norm/ --Lambda 'inf' --norm 