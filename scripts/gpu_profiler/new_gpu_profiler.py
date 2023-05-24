import torch
from torch import nn
import time
import csv
import sys
from torch.profiler import profile, record_function, ProfilerActivity


# all parameters
WARMPUP_ITS = sys.argv[1] if len(sys.argv) > 1 else 0
FOLDER = sys.argv[2] if len(sys.argv) > 2 else "./"
B = int(sys.argv[3]) if len(sys.argv) > 3 else 32 
device = sys.argv[4] if len(sys.argv) > 4 else 'cuda'
TIME_SCALE = int(sys.argv[5]) if len(sys.argv) > 5 else 1000000
model_name = 'vgg11'
#model_name = 'resnet50' # failing as of now
#model_name = 'resnet50'
lr = 0.01
print_short = 1 # print short form or long form

if model_name == 'vgg11':
    from torchvision.models import vgg11
    #net = vgg11(weights=None).to(device)
    net = vgg11().to(device)
elif model_name == 'resnet50':
    from torchvision.models import resnet50
    net = resnet50().to(device)
elif model_name == 'alexnet':
    from torchvision.models import alexnet
    #net = alexnet(weights=None).to(device)
    net = alexnet().to(device)
print(net)

X = torch.rand((B, 3, 224, 224), device=device)
Y = torch.rand((B, 3, 224, 224), device=device, requires_grad=True)


headers = ['#', 'name', 'FP', 'WG', 'IG', 'wts', 'WU']
if not print_short:
    headers = headers + ['ifm', 'ofm', 'stride', 'pad']

X = torch.rand((B, 3, 224, 224), device=device)
Y = torch.rand((B, 3, 224, 224), device=device, requires_grad=True)
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
output=net(X)
end.record()
torch.cuda.synchronize()
print("fwd pass time: " , start.elapsed_time(end))
rows = []
l_idx = 0
torch.cuda.synchronize()
# iterate over all layers and calculate FP, IG and WG times
for n, m in net.named_modules():

    if isinstance(m, (nn.Conv2d, nn.Linear)):
        # csv row entry
        row_entry = {}

        if isinstance(m, nn.Linear) and len(X.size()) > 2:
            X = torch.reshape(X, (X.size(0), -1))
            Y = torch.reshape(Y, (Y.size(0), -1))

        #for i in range(int(WARMPUP_ITS)):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("fwd pass"):
            # foward pass time
            #start.record()
                layer_out = m(X) # FP
            #end.record()
            #torch.cuda.synchronize()
            #fp_time = start.elapsed_time(end) 
       
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        layer_loss = torch.sum(layer_out)

        # weight gradient time
        start.record()
        layer_loss.backward()
        end.record()
        torch.cuda.synchronize()
        wg_time = start.elapsed_time(end) 

        # weight + input gradient time
        start.record()
        layer_out_y = m(Y) # FP with IG
        end.record()
        torch.cuda.synchronize()
        layer_loss_y = torch.sum(layer_out_y)

        start.record()
        layer_loss_y.backward()
        end.record()
        torch.cuda.synchronize()
        wg_ig_time = start.elapsed_time(end) 

        # input gradient time
        ig_time = wg_ig_time - wg_time
        assert ig_time > 0, "Input gradient compute time {} CANNOT be negative!!".format(ig_time)

        # weight update time
        start.record()
        m.weight.data = m.weight.data - lr * m.weight.grad
        end.record()
        torch.cuda.synchronize()
        wu_time = start.elapsed_time(end) 

        row_entry['#'] = l_idx
        row_entry['name'] = n
        row_entry['FP'] = round(fp_time, 3)
        row_entry['WG'] = round(wg_time, 3)
        row_entry['IG'] = round(ig_time, 3)
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

    #print("in: {}, out: {}".format(X.size(), layer_out.size()))
    X = layer_out.detach()
    Y = layer_out_y.detach()
    Y.requires_grad = True

    # CAUTION!!: adding the foll. 2 lines results in negative IG time. REFRAIN!!
    #X = X.to(device)
    #Y = X.to(device)

file_name = FOLDER + model_name + '_B{}.csv'.format(B)
print('headers ', headers)
with open(file_name, 'w') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)
f.close()


file_name = FOLDER + model_name + '_B{}_astra-sim.csv'.format(B)
print('headers ', headers)

output = []
for row in rows:
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
    line.append(row["WU"] * TIME_SCALE)

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

