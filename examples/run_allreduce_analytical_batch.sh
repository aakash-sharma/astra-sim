#! /bin/bash 

# Absolue path to this script
SCRIPT_DIR=$(dirname "$(realpath $0)")

# Absolute paths to useful directories
BINARY="${SCRIPT_DIR:?}"/../build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra
#NETWORK="${SCRIPT_DIR:?}"/../themis_inputs/inputs/network/analytical/4d_ring_fc_ring_switch.json
SYSTEM="${SCRIPT_DIR:?}"/../inputs/system/sample_torus_sys.txt
#WORKLOAD="${SCRIPT_DIR:?}"/../inputs/workload/microAllReduce.txt
WORKLOAD="${SCRIPT_DIR:?}"/../inputs/workload/Resnet50_DataParallel.txt

for topology in 'Torus3D' 'Torus2D' 'Switch' 'Ring_FullyConnected_Ring' 'Ring_FullyConnected_Switch'; do

	NETWORK="${SCRIPT_DIR:?}"/../inputs/network/analytical/sample_${topology}.json
	STATS="${SCRIPT_DIR:?}"/results/run_allreduce_analytical/${topology}

	rm -rf "${STATS}"
	mkdir "${STATS}"

	"${BINARY}" \
	--network-configuration="${NETWORK}" \
	--system-configuration="${SYSTEM}" \
	--workload-configuration="${WORKLOAD}" \
	--path="${STATS}/" \
	--run-name="${topology}" \
	--num-passes=5 \
	--comm-scale=50 \
	--total-stat-rows=1 \
	--stat-row=0 
done


for topology in '2d_switch' '3d_switch_hetero' '4d_ring_fc_ring_switch' '3d_fc_ring_switch' '3d_switch_homo' '4d_ring_switch_switch_switch'; do

	NETWORK="${SCRIPT_DIR:?}"/../themis_inputs/inputs/network/analytical/${topology}.json
	STATS="${SCRIPT_DIR:?}"/results/run_allreduce_analytical/${topology}

	rm -rf "${STATS}"
	mkdir "${STATS}"

	"${BINARY}" \
	--network-configuration="${NETWORK}" \
	--system-configuration="${SYSTEM}" \
	--workload-configuration="${WORKLOAD}" \
	--path="${STATS}/" \
	--run-name="${topology}" \
	--num-passes=5 \
	--comm-scale=50 \
	--total-stat-rows=1 \
	--stat-row=0 
done
