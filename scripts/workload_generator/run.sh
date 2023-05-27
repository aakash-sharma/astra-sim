python3 gen_astrasim_workload_input.py   --datatype_size=2  --topology=topology_inputs/resnet50.csv  --num_npus=16 --num_packages=2 --output_file=outputs/resnet50.txt --parallel=DATA --run_name=resnet50  --scalesim_config=../../extern/compute/SCALE-Sim/configs/google.cfg  --scalesim_path=../../extern/compute/SCALE-Sim

#python3 gen_astrasim_workload_input.py   --datatype_size=2  --topology=topology_inputs/yolo-tiny.csv  --num_npus=16 --num_packages=2 --output_file=outputs/yolo-tiny.txt --parallel=DATA --run_name=yolo-tiny --scalesim_config=../../extern/compute/SCALE-Sim/configs/google.cfg  --scalesim_path=../../extern/compute/SCALE-Sim

#python3 gen_astrasim_workload_input.py   --datatype_size=2  --topology=topology_inputs/alexnet.csv  --num_npus=16 --num_packages=2 --output_file=outputs/alexnet.txt --parallel=DATA --run_name=alexnet  --scalesim_config=../../extern/compute/SCALE-Sim/configs/google.cfg  --scalesim_path=../../extern/compute/SCALE-Sim
