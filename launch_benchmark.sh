#!/bin/bash
set -xe

# transformer-lt
function main {
    # prepare workload
    workload_dir="${PWD}"
    # set common info
    source oob-common/common.sh
    init_params $@
    fetch_device_info
    set_environment

    pip install -r ${workload_dir}/requirements.txt
    # pip install --no-deps torchvision -f https://download.pytorch.org/whl/torch_stable.html
    git clone https://github.com/pytorch/fairseq
    cd fairseq
    pip install --editable .
    #pip uninstall -y fairseq fastBPE cython
    #python setup.py clean
    #pip install -e .

    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        # pre run
        fairseq-generate ./data-bin/iwslt14.tokenized.de-en --path ${DATASET_DIR}/checkpoint_best.pt --beam 5 \
            --remove-bpe --cpu --max_iters 2 --warmup_iters 1 --batch-size 1 \
            --precision ${precision} --channels_last ${channels_last} || true
        #
        for batch_size in ${batch_size_list[@]}
        do
            # clean workspace
            logs_path_clean
            # generate launch script for multiple instance
            if [ "${OOB_USE_LAUNCHER}" == "1" ] && [ "${device}" != "cuda" ];then
                generate_core_launcher
            else
                generate_core
            fi
            # launch
            echo -e "\n\n\n\n Running..."
            cat ${excute_cmd_file} |column -t > ${excute_cmd_file}.tmp
            mv ${excute_cmd_file}.tmp ${excute_cmd_file}
            source ${excute_cmd_file}
            echo -e "Finished.\n\n\n\n"
            # collect launch result
            collect_perf_logs
        done
    done
}

function generate_core {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${cpu_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        if [ "${device}" != "cuda" ];then
            OOB_EXEC_HEADER=" numactl -m $(echo ${device_array[i]} |awk -F ';' '{print $2}') "
            OOB_EXEC_HEADER+=" -C $(echo ${device_array[i]} |awk -F ';' '{print $1}') "
        else
            OOB_EXEC_HEADER=" CUDA_VISIBLE_DEVICES=${device_array[i]} "
        fi
        printf " ${OOB_EXEC_HEADER} \
            python fairseq-generate ${DATASET_DIR}/iwslt14.tokenized.de-en --path ${DATASET_DIR}/checkpoint_best.pt \
                --beam 5 --remove-bpe --cpu \
                --max_iters ${num_iter} --warmup_iters ${num_warmup} \
                --batch-size ${batch_size} \
                --precision ${precision} \
                --channels_last ${channels_last} \
                ${addtion_options} \
        > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
        if [ "${numa_nodes_use}" == "0" ];then
            break
        fi
    done
    echo -e "\n wait" >> ${excute_cmd_file}
}

# run
function generate_core_launcher {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${cpu_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        printf "python -m oob-common.launch --enable_jemalloc \
                    --core_list $(echo ${device_array[@]} |sed 's/;.//g') \
                    --no_python \
                    --log_file_prefix rcpi${real_cores_per_instance} \
                    --log_path ${log_dir} \
                    --ninstances ${#cpu_array[@]} \
                    --ncore_per_instance ${real_cores_per_instance} \
            fairseq-generate ${DATASET_DIR}/iwslt14.tokenized.de-en --path ${DATASET_DIR}/checkpoint_best.pt \
                --beam 5 --remove-bpe --cpu \
                --max_iters ${num_iter} --warmup_iters ${num_warmup} \
                --batch-size ${batch_size} \
                --precision ${precision} \
                --channels_last ${channels_last} \
                ${addtion_options} \
        > /dev/null 2>&1 &  \n" |tee -a ${excute_cmd_file}
        break
    done
    echo -e "\n wait" >> ${excute_cmd_file}
}

# download common files
rm -rf oob-common && git clone https://github.com/intel-sandbox/oob-common.git

# Start
main "$@"
