#!/bin/bash
set -e

# ===================================
# Configurations
# ===================================
WORKLOAD=(
#  DLRM_HybridParallel.txt
  MLP_HybridParallel_Data_Model.txt
  MLP_ModelParallel.txt
  Resnet50_DataParallel.txt
  Transformer_HybridParallel.txt
  microAllReduce.txt
)
COMM_SCALE=(
  100
  150
  200
)
COMPUTE_SCALE=(
  0.26
)
NUM_PASSES=1
SYSTEM=(
  2d_switch_themis_scf.txt
  3d_switch_homo_themis_scf.txt
  3d_switch_hetero_themis_scf.txt
  3d_fc_ring_switch_themis_scf.txt
  4d_ring_switch_switch_switch_themis_scf.txt
  4d_ring_fc_ring_switch_themis_scf.txt
)
NETWORK=(
  2d_switch.json
  3d_switch_homo.json
  3d_switch_hetero.json
  3d_fc_ring_switch.json
  4d_ring_switch_switch_switch.json
  4d_ring_fc_ring_switch.json
)
UNITS_COUNT=(
  "16 64"
  "16 8 8"
  "16 8 8"
  "8 16 8"
  "4 4 8 8"
  "4 8 4 8"
)
LINKS_COUNT=(
  "6 1"
  "4 4 1"
  "8 4 1"
  "7 4 1"
  "2 8 4 1"
  "2 7 6 1"
)
LINK_LATENCY=(
  "1 1"
  "1 1 1"
  "1 1 1"
  "1 1 1"
  "1 1 1 1"
  "1 1 1 1"
)
LINK_BANDWIDTH=(
  "25 100"
  "25 25 100"
  "25 25 50"
  "25 25 50"
  "125 25 25 50"
  "187.5 25 25 100"
)
CONFIG_NAME="DDL_themis"
# ===================================
# ===================================


# Absolue path to this script
SCRIPT_DIR=$(dirname "$(realpath $0)")

# Absolute paths to useful directories
BINARY="${SCRIPT_DIR:?}"/../build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra
PROJECT_DIR="${SCRIPT_DIR:?}"/../
INPUT_DIR=${PROJECT_DIR}/themis_inputs/inputs
COMPILE_SCRIPT="${SCRIPT_DIR:?}"/../build.sh
STATS="${SCRIPT_DIR:?}"/../results/${CONFIG_NAME}

# create result directory
rm -rf "${STATS}"
mkdir -p "${STATS}"

# compile will be done by the main script
#echo "[SCRIPT] Compiling AnalyticalAstra"
#"${COMPILE_SCRIPT}" -c


# run test
current_row=-1
tot_stat_row=$((${#WORKLOAD[@]} * ${#SYSTEM[@]} * ${#COMM_SCALE[@]}))

for workload in "${WORKLOAD[@]}"; do
  for comm_scale in "${COMM_SCALE[@]}"; do
    for comp_scale in "${COMPUTE_SCALE[@]}"; do
      for i in "${!SYSTEM[@]}"; do
        current_row=$(($current_row + 1))
	echo "================================="
	echo ${UNITS_COUNT[${i}]}
	echo "=========$workload"
        filename="${CONFIG_NAME}-${workload}-${NETWORK[${i}]}-${SYSTEM[${i}]}-${comm_scale}-${UNITS_COUNT[${i}]}-${LINKS_COUNT[${i}]}-${LINK_BANDWIDTH[${i}]}-${NUM_PASSES}"

        echo "[SCRIPT] Initiate ${filename}"

        "${BINARY}" \
          --network-configuration="${INPUT_DIR}/network/analytical/${NETWORK[${i}]}" \
          --system-configuration="${INPUT_DIR}/system/${SYSTEM[${i}]}" \
          --workload-configuration="${INPUT_DIR}/workload/${workload}" \
          --path="${STATS}/" \
          --units-count ${UNITS_COUNT[${i}]} \
          --links-count ${LINKS_COUNT[${i}]} \
          --link-bandwidth ${LINK_BANDWIDTH[${i}]} \
          --link-latency ${LINK_LATENCY[${i}]} \
          --num-passes ${NUM_PASSES} \
          --num-queues-per-dim 1 \
          --comm-scale ${comm_scale} \
          --compute-scale ${comp_scale} \
          --injection-scale 1 \
          --rendezvous-protocol false \
          --total-stat-rows "${tot_stat_row}" \
          --stat-row "${current_row}" \
          --run-name "${filename}" >> "${STATS}"/"${filename}".txt &
      done
      wait $!
    done
  done
done
