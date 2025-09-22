#!/bin/bash

echo "Profiling the Gaussian splatting kernel"

# sudo env "PATH=$PATH" /opt/nvidia/nsight-compute/2025.1.0/ncu \
#     --launch-skip 3838 \
#     --launch-count 28 \
#     --metric "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
#     smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
#     smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
#     sm__inst_executed.sum,\
#     sm__throughput.avg.pct_of_peak_sustained_active" \
#     -f -o /home/jiexiao/research/gaussian-splatting/data/playroom/profile/forward_profile \
#     python render_single_view.py \
#     -m /home/jiexiao/research/gaussian-splatting/data/playroom/model \
#     --view_index 23 \
#     --skip_test

# sudo env "PATH=$PATH" /opt/nvidia/nsight-compute/2025.1.0/ncu \
#     --launch-skip 3838 \
#     --launch-count 28 \
#     --metric "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second,sm__throughput.avg.pct_of_peak_sustained_active,dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,lts__throughput.avg.pct_of_peak_sustained_elapsed" \
#     -f -o /home/jiexiao/research/gaussian-splatting/data/playroom/profile/single_view \
#     python render_single_view.py \
#     -m /home/jiexiao/research/gaussian-splatting/data/playroom/model \
#     --view_index 23 \
#     --skip_test
python render_single_view.py \
    -m /home/jiexiao/research/skm-gs/data/playroom/model \
    -s /home/jiexiao/research/skm-gs/data/playroom/gen_data \
    --view_index 25 \
    --skip_test

# sudo env "PATH=$PATH" /opt/nvidia/nsight-compute/2025.1.0/ncu \
#     --launch-skip 3863 \
#     --launch-count 5 \
#     --metric "sm__sass_thread_inst_executed_op_ffma_pred_on.avg,sm__sass_thread_inst_executed_op_fmul_pred_on.avg,sm__sass_thread_inst_executed_op_fadd_pred_on.avg,sm__cycles_elapsed.avg.per_second" \
#     -f -o /home/jiexiao/research/gaussian-splatting/data/playroom/profile/avg_perf \
#     python render_single_view.py \
#     -m /home/jiexiao/research/gaussian-splatting/data/playroom/model \
#     --view_index 23 \
#     --skip_test

# ncu --import /home/jiexiao/research/gaussian-splatting/data/playroom/profile/avg_perf.ncu-rep --csv > avg_perf.csv

# sudo env "PATH=$PATH" nsys profile -o /home/jiexiao/research/gaussian-splatting/data/playroom/profile/full_profile python train.py -s /home/jiexiao/research/gaussian-splatting/data/playroom/gen_data --port 8081 --eval

# sudo env "PATH=$PATH" nsys profile -o /home/jiexiao/research/gaussian-splatting/data/playroom/profile/single_view_report python render_single_view.py -m /home/jiexiao/research/gaussian-splatting/data/playroom/model --view_index 23 --skip_test

# sudo env "PATH=$PATH" /opt/nvidia/nsight-compute/2025.1.0/ncu \
#     --metric "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second, \
# l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second, \
# dram__bytes_read.sum.per_second, \
# dram__bytes_write.sum.per_second, \
#                dram__throughput.avg.pct_of_peak_sustained_elapsed, \
#                lts__throughput.avg.pct_of_peak_sustained_elapsed, \
#                smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
#                smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
#                smsp__sass_thread_inst_executed_op_ffma_pred_on.sum, \
#                sm__inst_executed.sum,\
#                sm__throughput.avg.pct_of_peak_sustained_active" \
#     -f -o /home/jiexiao/research/gaussian-splatting/data/playroom/profile/test_profile \
#     python render_single_view.py \
#     -m /home/jiexiao/research/gaussian-splatting/data/playroom/model \
#     --view_index 50 \
#     --skip_test