CUDA_VISIBLE_DEVICES="3" python GAIRAT_wrong_early_time_copy.py \
--dataset "cifar100" \
--net "preactresnet18" \
--out-dir "./results/0726_2_cifar100_ham_keep_clean0d3_60epo/" \
--epochs "120" \
--begin_epoch "60" \
--classify_step "3" \
--keep_bili_high_conf "0.3" \
