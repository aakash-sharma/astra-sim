import torch
from torch import nn
import time
import csv
import sys
import pynvml
from torch.profiler import profile, record_function, ProfilerActivity
from pynvml.smi import nvidia_smi
from threading import Thread

# all parameters
FOLDER = sys.argv[1] if len(sys.argv) > 1 else "./"
B = int(sys.argv[2]) if len(sys.argv) > 2 else 32 
device = sys.argv[3] if len(sys.argv) > 3 else 'cuda'
TIME_SCALE = 1 #int(sys.argv[5]) if len(sys.argv) > 5 else 1000000
model_name = 'vgg11'
model_name = 'resnet50'
#model_name = 'alexnet'
#model_name = 'resnet18'
#model_name = 'mobilenet_v2'
lr = 0.01
print_short = 1 # print short form or long form
run_thread = True
gpu_usage = 0
cuda_time = 0

if model_name == 'vgg11':
    from torchvision.models import vgg11
    #net = vgg11(weights=None).to(device)
    net = vgg11().to(device)
elif model_name == 'resnet50':
    from torchvision.models import resnet50, ResNet50_Weights
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
elif model_name == 'alexnet':
    from torchvision.models import alexnet
    net = alexnet().to(device)
elif model_name == 'resnet18':
    from torchvision.models import resnet18
    net = resnet18().to(device)
elif model_name == 'mobilenet_v3':
    from torchvision.models import mobilenet_v3_small
    net = mobilenet_v3_small().to(device)
elif model_name == 'mobilenet_v2':
    from torchvision.models import mobilenet_v2
    net = mobilenet_v2().to(device)
elif model_name == 'efficientnet':
    from torchvision.models import efficientnet_b7
    net = efficientnet_b7().to(device)


pynvml.nvmlInit()
print(net)

def record_gpu_usage(layer: str) -> None:
    global gpu_usage
    i = 0
    while run_thread:
        #print(nvidia_smi.getInstance().DeviceQuery('utilization.gpu')['gpu'][0]['utilization']['gpu_util'])
        gpu_usage += float(nvidia_smi.getInstance().DeviceQuery('utilization.gpu')['gpu'][0]['utilization']['gpu_util'])
        i += 1
        #print(f"GPU util for {layer} - {nvidia_smi.getInstance().DeviceQuery('utilization.gpu')}")
    
    gpu_usage /= i


def trace_handler(p):
    global cuda_time
    output = p.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=10)
#    print(output)
    output_lines = output.splitlines()

    cuda_line = output_lines[-1]
    print(cuda_line)

    if "Self CUDA time total" not in cuda_line:
        cuda_time = -1
        return

    cuda_time = cuda_line.split()[4]

    
    """
    if "Self CUDA" not in output_lines[1]:
        return

    cuda_time = output_lines[3].split()[6]
    """

    if "ms" in cuda_time:
        cuda_time = float(cuda_time.split("ms")[0]) * 1000000

    elif "us" in cuda_time:
        cuda_time = float(cuda_time.split("us")[0]) * 1000

    cuda_time /= 3

    #print(output)


"""
X = torch.rand((B, 3, 224, 224), device=device)
Y = torch.rand((B, 3, 224, 224), device=device, requires_grad=True)

for i in range(60):
    X = torch.rand((B, 3, 224, 224), device=device)
    Y = torch.rand((B, 3, 224, 224), device=device, requires_grad=True)

    for n, m in net.named_modules():

        if 'layer' in n and 'conv1' in n and '0' in n:
            X_pll = X.clone()
            Y_pll = Y.clone()

        if isinstance(m, (nn.Conv2d, nn.Linear)):

            if isinstance(m, nn.Linear) and len(X.size()) > 2:
                X = torch.reshape(X, (X.size(0), -1))
                Y = torch.reshape(Y, (Y.size(0), -1))

            if 'downsample' in n:
                X = X_pll
                Y = Y_pll

            layer_out = m(X) # FP
            layer_loss = torch.sum(layer_out)
            layer_loss.backward()
            layer_out_y = m(Y) # FP with IG
            layer_loss_y = torch.sum(layer_out_y)
            layer_loss_y.backward()
            m.weight.data = m.weight.data - lr * m.weight.grad

        elif isinstance(m, (nn.ReLU, nn.AvgPool2d, nn.BatchNorm2d, nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.Dropout)):
            with torch.no_grad():
                layer_out = m(X)
                layer_out_y = m(Y)

        else:
            continue

        X = layer_out.detach()
        Y = layer_out_y.detach()
        Y.requires_grad = True
"""

headers = ['#', 'name', 'FP', 'WG', 'IG', 'wts', 'WU']
if not print_short:
    headers = headers + ['ifm', 'ofm', 'stride', 'pad']

X = torch.rand((B, 3, 224, 224), device=device)
Y = torch.rand((B, 3, 224, 224), device=device, requires_grad=True)
rows = []
l_idx = 0
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# iterate over all layers and calculate FP, IG and WG times

for n, m in net.named_modules():

    if 'layer' in n and 'conv1' in n and '0' in n:
        X_pll = X.clone()
        Y_pll = Y.clone()

    if isinstance(m, (nn.Conv2d, nn.Linear)):
        # csv row entry
        row_entry = {}

        if isinstance(m, nn.Linear) and len(X.size()) > 2:
            X = torch.reshape(X, (X.size(0), -1))
            Y = torch.reshape(Y, (Y.size(0), -1))

        if 'downsample' in n:
            X = X_pll
            Y = Y_pll

        thread = Thread(target = record_gpu_usage, args=(str(l_idx),))

        layer_out_list = []
        with profile(activities=[ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                skip_first=10,
                wait=5,
                warmup=2,
                active=3),
            on_trace_ready=trace_handler
                ) as prof:
            # foward pass time
                thread.start()
                for idx in range(20):
                    layer_out = m(X) # FP
                    layer_out_list.append(layer_out)
                    #X_ = X.detach().clone()
                    prof.step()
               
                run_thread = False
        thread.join()
        run_thread = True

        print("gpu usage: ", gpu_usage)
        print("cuda_time: ", cuda_time)
        if cuda_time == -1:
            sys.exit(-1)
        fp_cycles = int(cuda_time / 3 * gpu_usage)
        print("fp_cycles: ",  fp_cycles)
        gpu_usage = 0
        cuda_time = 0
        torch.cuda.synchronize()

        # weight gradient time
        layer_loss_list = []
        for i in range(len(layer_out_list)):
            layer_loss_list.append(torch.sum(layer_out_list[i]))
        layer_loss = torch.sum(layer_out)
        thread = Thread(target = record_gpu_usage, args=(str(l_idx),))
        
        with profile(activities=[ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                skip_first=10,
                wait=5,
                warmup=2,
                active=3),
            on_trace_ready=trace_handler
                ) as prof:
            # wt gradient time
                thread.start()
                for idx in range(20):
                    layer_loss_list[idx].backward()
                    prof.step()

                run_thread = False
        thread.join()
        run_thread = True

        print("gpu usage: ", gpu_usage)
        print("cuda_time: ", cuda_time)
        if cuda_time == -1:
            sys.exit(-1)
        wg_cycles = int(cuda_time * gpu_usage / 100)
        print("wg_cycles: ", wg_cycles)
        gpu_usage = 0
        cuda_time = 0
        torch.cuda.synchronize()

        # weight + input gradient time
        layer_loss_y_list = [] 
        for i in range(len(layer_out_list)):
            layer_out_y = m(Y) # FP with IG
            layer_loss_y_list.append(torch.sum(layer_out_y))
        thread = Thread(target = record_gpu_usage, args=(str(l_idx),))

        with profile(activities=[ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                skip_first=10,
                wait=5,
                warmup=2,
                active=3),
            on_trace_ready=trace_handler
                ) as prof:
            # wg + ig  time
                thread.start()
                for idx in range(20):
                    layer_loss_y_list[idx].backward()
                    #layer_loss_y_.backward()
                    #layer_loss_y_ = layer_loss_y.detach().clone()
                    prof.step()

                run_thread = False
        thread.join()
        run_thread = True

        print("gpu usage: ", gpu_usage)
        print("cuda_time: ", cuda_time)
        if cuda_time == -1:
            sys.exit(-1)
        wg_ig_cycles = int(cuda_time * gpu_usage / 100)
        print("wg_ig_cycles: ", wg_ig_cycles)
        gpu_usage = 0
        cuda_time = 0
        torch.cuda.synchronize()

        ig_cycles = wg_ig_cycles - wg_cycles
        #assert ig_cycles > 0, "Input gradient compute time {} CANNOT be negative!!".format(ig_cycles)
        assert wg_ig_cycles  > fp_cycles, "Input + weight gradient compute time {} CANNOT be less than fowd pass cycles{}".format(wg_ig_cycles, fp_cycles)

        # weight update time
        start.record()
        m.weight.data = m.weight.data - lr * m.weight.grad
        end.record()
        torch.cuda.synchronize()
        wu_time = int(start.elapsed_time(end))

        row_entry['#'] = l_idx
        row_entry['name'] = n
        row_entry['FP'] = round(fp_cycles, 3)
        row_entry['WG'] = round(wg_cycles, 3)
        row_entry['IG'] = round(ig_cycles, 3)
        row_entry['WU'] = round(wu_time, 3)

        wt_size = 1
        for wt in list(m.weight.size()):
            wt_size *= wt

        row_entry['wts'] = wt_size * 4

        if not print_short:
            row_entry['ifm'] = X.size()
            #row_entry['wts'] = m.weight.size()
            if isinstance(m, nn.Conv2d):
                row_entry['ofm'] = layer_out.size()
                row_entry['stride'] = m.stride
                row_entry['pad'] = m.padding
            else:
                row_entry['stride'] = '-'
                row_entry['pad'] = '-'

        rows.append(row_entry)
        print(row_entry)

        l_idx += 1

    elif isinstance(m, (nn.ReLU, nn.AvgPool2d, nn.BatchNorm2d, nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.Dropout)):
        with torch.no_grad():
            layer_out = m(X)
            layer_out_y = m(Y)

    else:
        continue

    X = layer_out.detach()
    Y = layer_out_y.detach()
    Y.requires_grad = True

    # CAUTION!!: adding the foll. 2 lines results in negative IG time. REFRAIN!!
    #X = X.to(device)
    #Y = X.to(device)

file_name = FOLDER + "/" + model_name + '_B{}.csv'.format(B)
print('headers ', headers)
with open(file_name, 'w') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)
f.close()


file_name = FOLDER + model_name + '_B{}_astra-sim2.csv'.format(B)
print('headers ', headers)

output = []
for row in rows:

    if row["IG"] <= 0:
        continue
    line = []
    line.append(row["name"])
    line.append("-1")
    line.append(row["FP"] * TIME_SCALE)
    line.append("NONE")
    line.append("0")
    line.append(row["IG"] * TIME_SCALE)
    line.append("NONE")
    line.append("0")
    line.append(row["WG"] * TIME_SCALE)
    line.append("ALLREDUCE")
    line.append(row["wts"])
    line.append(row["WU"] * 1000000)

    line = map(lambda x: str(x), line)
    line_string = "\t".join(line)

    output.append(line_string)

with open(file_name, 'w') as f:
    f.write("DATA\n")
    f.write(str(len(rows)) + "\n")

    for line in output:
        f.write(line)
        f.write("\n")

f.close()

