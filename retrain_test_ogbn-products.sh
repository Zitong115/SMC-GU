# python main.py --unlearn_task edge --unlearn_ratio 0.01 --dataset cora --target_model GCN --find_k_hops False

#!/bin/bash

for a in ogbn-products 
do

    for b in edge
    do

        for c in GCN GIN GAT 
        do

            for d in 0.01 0.05 0.1 0.15 
            do

                # Function to get GPU utilization for a given GPU ID
                get_gpu_load() {
                    gpu_id=$1
                    load=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $gpu_id)
                    echo "$load"
                }

                # Function to choose the GPU with the least load
                choose_gpu_with_least_load() {
                    gpu_count=$(nvidia-smi --list-gpus | wc -l)
                    if [ $gpu_count -eq 0 ]; then
                        echo "No GPUs available."
                        exit 1
                    fi

                    # Initialize variables
                    min_load=100
                    chosen_gpu=""

                    # Loop through available GPUs
                    for ((gpu_id = 0; gpu_id < $gpu_count; gpu_id++)); do
                        load=$(get_gpu_load $gpu_id)
                        if [ -z "$load" ]; then
                            continue
                        fi

                        if ((load < min_load)); then
                            min_load=$load
                            chosen_gpu=$gpu_id
                        fi
                    done

                    echo "$chosen_gpu"
                }

                # Choose GPU with the least load
                chosen_gpu=$(choose_gpu_with_least_load)

                if [ -z "$chosen_gpu" ]; then
                    echo "No available GPUs or unable to determine GPU load."
                    exit 1
                fi

                echo "Selected GPU: $chosen_gpu"

                # Set the CUDA_VISIBLE_DEVICES environment variable to restrict execution to the chosen GPU
                export CUDA_VISIBLE_DEVICES=$chosen_gpu


                info="Dataset = ${a} task unlearn = ${b} model = ${c} unlearn ratio = ${d}"

                echo "Start ${info}"
                output_file="./output/test_retrain_20250506.txt"

                nohup python main.py --dataset_name $a \
                    --data_sampler neighbor \
                    --num_epochs 3 \
                    --train_lr 0.0001 \
                    --target_model $c \
                    --retrain_model_for_cmp False \
                    --unlearn_method retrain \
                    --unlearn_task $b \
                    --unlearn_ratio $d \
                    --is_train_target_model True \
                    --is_use_node_feature True \
                    --num_runs 1 \
                    --file_name ./output/retrain_test_20250423 \
                    --scale 50000 \
                    --iteration 10 \
                    --batch_size 512\
                    --test_ratio 0.1 \
                    --gif_sampling_dataset -1.0 >> $output_file 2>&1 &

                pid=$!
                echo "Python program running with PID: $pid"
                wait $pid

            done
        done
    done
done