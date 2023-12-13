from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import pyarrow.dataset as ds
from torch.utils.data import Dataset
import pyarrow as pa
import time
import numpy as np

'''
This cript implement batch fetch shuffle data
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class CustomDataset(Dataset):
    def __init__(self, pyarrow_dataset):
        self.record_count = 0
        self. batch_count = []
        for _ in pyarrow_dataset.to_batches():
            self.record_count += len(_)  
        self.dataset = pyarrow_dataset# Count the records in the current batch

    def __len__(self):
        return self.record_count

    def __getitems__(self, idx):
        # sample = self.dataset.to_batches().take([idx])  # Use to_batches to get the record
        self.comsume_time = 0
        
        
        arrow_array = pa.array(idx,type=pa.int64())
        start_time = time.time()
        # print(dataset.take(arrow_array).to_pandas())
        sample = self.dataset.take(arrow_array)  # Use to_batches to get the record
        end_time1 = time.time()

        # labels = sample['label'].to_pandas()
        labels = sample['label'].to_pandas()

        # image = torch.FloatTensor(sample['image'].to_pandas()).view((28,28)).unsqueeze(0)
        images = torch.FloatTensor(sample['image'].to_pandas()).view(32,1,28,28)
        
        comsume_time = (end_time1-start_time)
        with open('/home/yue21/mlndp/ML_example/result/take_time.txt', 'a') as take_time_save:
            take_time_save.write(f"take_operation_time: {comsume_time} second\n")
        # image = torch.tensor(sample[0]['image'][0])

        # batch_data = [{'data': images[i], 'target': int(labels[i])} for i in range(len(idx))]
        batch_data = [{'data': images[i], 'target': int(labels[i])} for i in range(len(idx))]


        return batch_data


# class CustomDataset(Dataset):
#     def __init__(self, pyarrow_dataset, batch_size==32):
#         self.record_batches = list(pyarrow_dataset.to_batches())
#         self.batch_size = batch_size
#         self.cache = {}  # Cache to store entire batches

#     def __len__(self):
#         return len(self.record_batches)

#     def __getitem__(self, idx):
#         if idx not in self.cache:
#             # If data is not in the cache, compute and save the entire batch
#             start_idx = idx * self.batch_size
#             end_idx = (idx + 1) * self.batch_size

#             image_data = self.record_batches[0]['image'].slice(start_idx, end_idx).to_pandas()
#             # image_tensor = torch.FloatTensor(image_data).view(-1, 28, 28)
#             image_tensor = torch.FloatTensor(image_data)

#             label_data = self.record_batches[0]['label'].slice(start_idx, end_idx).to_pandas()

#             # Save entire batch to cache
#             self.cache[idx] = {'images': image_tensor, 'labels': label_data}

#         # Return a single sample from the cached batch
#         sample_idx = idx % self.batch_size
#         reshaped_image = self.cache[idx]['images'][sample_idx].view(1, 28, 28)
#         return reshaped_image, int(self.cache[idx]['labels'][sample_idx])

# class CustomDataset(Dataset):
#     def __init__(self, pyarrow_dataset):
#         self.record_count = 0
#         for _ in pyarrow_dataset.to_batches():
#             self.record_count += len(_)  
#         self.dataset = pyarrow_dataset# Count the records in the current batch

#     def __len__(self):
#         return self.record_count

#     def __getitem__(self, idx):
#         # sample = self.dataset.to_batches().take([idx])  # Use to_batches to get the record

#         arrow_array = pa.array([idx])
#         # print(dataset.take(arrow_array).to_pandas())
#         sample = self.dataset.take(arrow_array)  # Use to_batches to get the record

#         label = int(sample['label'].to_pandas())
#         # image = torch.FloatTensor(sample['image'].to_pandas()).view((28,28)).unsqueeze(0)
#         image = torch.FloatTensor(sample['image'].to_pandas()).view((1,28,28))
#         # image = torch.tensor(sample[0]['image'][0])
#         return image, label


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    # statixtic calculate
    total_data_processed = 0

        # Open the files in append mode
    with open('/home/yue21/mlndp/ML_example/result/iops.txt', 'a') as iops_file, \
         open('/home/yue21/mlndp/ML_example/result/throughput.txt', 'a') as throughput_file, \
         open('/home/yue21/mlndp/ML_example/result/latency.txt', 'a') as latency_file, \
         open('/home/yue21/mlndp/ML_example/result/time_consume_with_ML.txt', 'a') as time_consume_with_ML:

        batch_total_time = 0
        batches_num = 0
        # time stample, to do , one epoch. 
        # for batch_idx, (data, target) in enumerate(train_loader):
        for batch_idx, batch in enumerate(train_loader):
            start_time = time.time()

            #==================================for Pure ML =====================================
            # image_data = batch[0]
            # # Get label value for the current index
            # label_value = batch[1]
            # data, target = image_data.to(device), label_value.to(device)
            #===================================================================================

            #===================================For near data and non near data================================
            batches_num = train_loader.dataset.record_count/train_loader.batch_size
            data = batch['data']
            label = batch['target']
            data, target = data.to(device), label.to(device)
            #====================================================================================================
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            total_data_processed += len(data)
            elapsed_time = time.time() - start_time
            iops = (batch_idx + 1) / elapsed_time # batch per second
            throughput = total_data_processed / elapsed_time # data per second
            latency = elapsed_time #time for processing for each batch

            batch_total_time+=elapsed_time

            # Write the metrics to the files
            iops_file.write(f'IOPS: {iops:.2f} batch/s\n')
            throughput_file.write(f'Throughput: {throughput:.2f} data/s\n')
            latency_file.write(f'Batch_Latency: {latency:.6f} second\n')
            time_consume_with_ML.write(f'train_time_consume: {latency:.6f} second\n')


            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

                print(f'IOPS: {iops:.2f} batch/s')
                print(f'Throughput: {throughput:.2f} data/s')
                print(f'Batch_Latency: {latency:.6f} second')

                if args.dry_run:
                    break
        # ========================for near data and non near data ====================================
        ave_batch_time_consume = batch_total_time/batches_num
        print(f"ave_batch_time_consume:{ave_batch_time_consume}")
        #=============================================================================================


# def test(model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         # for data, target in test_loader:
#         for batch in enumerate(test_loader):
#             data = batch['data']
#             target = batch['target']
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)

#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

#===============================Skyhook=========================================
    train_dataset_path = "/mnt/cephfs/raw_minist_dataset/train"
    test_dataset_path = "/mnt/cephfs/raw_minist_dataset/test"

    format = ds.SkyhookFileFormat("parquet", "/etc/ceph/ceph.conf")
    train_dataset,metadata = ds.dataset(train_dataset_path, format=format)
    test_dataset,metadata = ds.dataset(test_dataset_path, format=format)
    train_custom_dataset = CustomDataset(train_dataset)
    test_custom_dataset = CustomDataset(test_dataset)
    train_loader = torch.utils.data.DataLoader(train_custom_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_custom_dataset, batch_size=32, shuffle=True)

# #===============================No Skyhook=========================================
    # train_dataset_path = "/mnt/cephfs/raw_minist_dataset/train"
    # test_dataset_path = "/mnt/cephfs/raw_minist_dataset/test"

    # train_dataset,metadata = ds.dataset(train_dataset_path, format="parquet")
    # test_dataset,metadata = ds.dataset(test_dataset_path, format="parquet")
    # train_custom_dataset = CustomDataset(train_dataset)
    # test_custom_dataset = CustomDataset(test_dataset)
    # train_loader = torch.utils.data.DataLoader(train_custom_dataset, batch_size=32, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_custom_dataset, batch_size=32, shuffle=True)


#=============================Pure ML=================================================
    # dataset1 = datasets.MNIST('/mnt/cephfs/petastorm_dataset/pure_mnist_dataset', train=True, download=True,
    #                    transform=transform)
    # dataset2 = datasets.MNIST('/mnt/cephfs/petastorm_dataset/pure_mnist_dataset', train=False,
    #                    transform=transform)

    # train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    # test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
#=======================================================================================


    model = Net().to(device)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train(args, model, device, train_loader , optimizer, epoch)
        end = time.time()
        epoch_time_comsume = end- start
        print(f"epoch {epoch}: {epoch_time_comsume:.6f} second\n")
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()