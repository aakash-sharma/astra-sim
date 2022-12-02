#! /bin/bash 

# Absolue path to this script
SCRIPT_DIR=$(dirname "$(realpath $0)")

# Absolute paths to useful directories
BINARY="${SCRIPT_DIR:?}"/../build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra
SYSTEM="${SCRIPT_DIR:?}"/../inputs/system/sample_torus_sys.txt

run() {
	WORKLOAD_DIR=$1
	WORKLOAD=$2

	echo $WORKLOAD_DIR
	echo $WORKLOAD

	#for topology in 'Torus5D' 'Torus2D' 'Switch' 'Ring_FullyConnected_Ring' 'Ring_FullyConnected_Switch'; do
	for topology in 'Torus2D'; do

		NETWORK="${SCRIPT_DIR:?}"/../inputs/network/analytical/sample_${topology}.json
		STATS="${SCRIPT_DIR:?}"/results/run_allreduce_analytical/${WORKLOAD_DIR}/${topology}

		if [ -d ${STATS} ]; then
			continue
		fi

		mkdir -p "${STATS}"

		"${BINARY}" \
		--network-configuration="${NETWORK}" \
		--system-configuration="${SYSTEM}" \
		--workload-configuration="${WORKLOAD}" \
		--path="${STATS}/" \
		--run-name="${topology}" \
		--num-passes=5 \
		--comm-scale=50 \
		--total-stat-rows=1 \
		--stat-row=0  >> out &
	done
	wait $!
}

run_themis() {
	WORKLOAD_DIR=$1
	WORKLOAD=$2

	for topology in '2d_switch' '3d_switch_hetero' '4d_ring_fc_ring_switch' '3d_fc_ring_switch' '3d_switch_homo' '4d_ring_switch_switch_switch'; do

		NETWORK="${SCRIPT_DIR:?}"/../themis_inputs/inputs/network/analytical/${topology}.json
		STATS="${SCRIPT_DIR:?}"/results/run_allreduce_analytical/${WORKLOAD_DIR}/${topology}

		if [ -d ${STATS} ]; then
                        continue
                fi

		mkdir -p "${STATS}"
		
		"${BINARY}" \
		--network-configuration="${NETWORK}" \
		--system-configuration="${SYSTEM}" \
		--workload-configuration="${WORKLOAD}" \
		--path="${STATS}/" \
		--run-name="${topology}" \
		--num-passes=5 \
		--comm-scale=50 \
		--total-stat-rows=1 \
		--stat-row=0 >> out &
	done
	wait $!
}

WORKLOAD_DIR="DLRM_HybridParallel"
WORKLOAD="${SCRIPT_DIR:?}"/../inputs/workload/DLRM_HybridParallel.txt

run $WORKLOAD_DIR $WORKLOAD
run_themis $WORKLOAD_DIR $WORKLOAD

WORKLOAD_DIR="MLP_HybridParallel_Data_Model"
WORKLOAD="${SCRIPT_DIR:?}"/../inputs/workload/MLP_HybridParallel_Data_Model.txt

run $WORKLOAD_DIR $WORKLOAD
run_themis $WORKLOAD_DIR $WORKLOAD

WORKLOAD_DIR="MLP_ModelParallel.txt"
WORKLOAD="${SCRIPT_DIR:?}"/../inputs/workload/MLP_ModelParallel.txt

run $WORKLOAD_DIR $WORKLOAD
run_themis $WORKLOAD_DIR $WORKLOAD

WORKLOAD="${SCRIPT_DIR:?}"/../inputs/workload/Resnet50_DataParallel.txt
WORKLOAD_DIR="Resnet50_DataParallel"

run $WORKLOAD_DIR $WORKLOAD
run_themis $WORKLOAD_DIR $WORKLOAD

WORKLOAD_DIR="Transformer_HybridParallel"
WORKLOAD="${SCRIPT_DIR:?}"/../inputs/workload/Transformer_HybridParallel.txt

run $WORKLOAD_DIR $WORKLOAD
run_themis $WORKLOAD_DIR $WORKLOAD

WORKLOAD_DIR="microAllReduce"
WORKLOAD="${SCRIPT_DIR:?}"/../inputs/workload/microAllReduce.txt

run $WORKLOAD_DIR $WORKLOAD
run_themis $WORKLOAD_DIR $WORKLOAD


