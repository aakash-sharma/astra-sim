#! /bin/bash -v

# Absolue path to this script
SCRIPT_DIR=$(dirname "$(realpath $0)")

# Absolute paths to useful directories
BINARY="${SCRIPT_DIR:?}"/../build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra
NETWORK="${SCRIPT_DIR:?}"/../inputs/network/analytical/sample_Torus3D.json
SYSTEM="${SCRIPT_DIR:?}"/../inputs/system/sample_torus_sys.txt
#WORKLOAD="${SCRIPT_DIR:?}"/../inputs/workload/microAllReduce.txt
WORKLOAD="${SCRIPT_DIR:?}"/../inputs/workload/Resnet50_DataParallel.txt
STATS="${SCRIPT_DIR:?}"/results/run_allreduce_analytical
COMPILE_SCRIPT="${SCRIPT_DIR:?}"/../build.sh
"${COMPILE_SCRIPT}" -cd

rm -rf "${STATS}"
mkdir "${STATS}"
echo $BINARY

lldb -- "${BINARY}" \
--network-configuration="${NETWORK}" \
--system-configuration="${SYSTEM}" \
--workload-configuration="${WORKLOAD}" \
--path="${STATS}/" \
--run-name="sample_all_reduce" \
--num-passes=5 \
--comm-scale=50 \
--total-stat-rows=1 \
--stat-row=0 

