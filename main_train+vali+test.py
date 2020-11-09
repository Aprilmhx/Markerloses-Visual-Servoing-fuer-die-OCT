import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn, optim
import torchvision.models as models
from neu_data_generator import OCTDataset
import argparse
from matplotlib import pyplot as plt
import numpy as np
from hexapod_control import control,control_valid

device = torch.device('cuda')
model = models.resnet18(pretrained=True)
fc_features = model.fc.in_features
model.fc = nn.Linear(fc_features, 3)
criteon = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(batch_size, epochs):

    dataset_train = OCTDataset('/home/mohanxu/Desktop/ophonlas-oct/oct_network_app/train')
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataset_vali = OCTDataset('/home/mohanxu/Desktop/ophonlas-oct/oct_network_app/validation')
    dataloader_vali= DataLoader(dataset_vali, batch_size=batch_size, shuffle=True)
    dataloaders = {'train': dataloader_train,'vali': dataloader_vali}

    #x, label = iter(dataloader_train).next()
    #print('x:', x.shape, 'label:', label.shape)


    #model = model.to(device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, eps=1e-10)
    #print(model)
    loss_list_train = []
    loss_list_vali = []
    for epoch in range(epochs):
        print("lr =", optimizer.param_groups[0]['lr'])
        #batches = tqdm.tqdm(dataloader_train)
        loss_train = []
        loss_vali = []
        for phase in ['train','vali']:
            if phase == 'train':
                model.train(True)
                print('train')
                for batchidx, (data, pos) in enumerate(dataloaders[phase]):
                    data_z = torch.zeros(32, 1, 224, 224)
                    data_e = data.view(16, 2, 224, 224)
                    padding = torch.zeros(16, 1, 224, 224)
                    data_e = torch.cat((data_e, padding), 1)
                    for j in range(32):
                        if j % 2 == 0:
                            data_z[j] = data[j + 1]
                        else:
                            data_z[j] = data[j - 1]
                    data_z = data_z.view(16, 2, 224, 224)
                    data_z = torch.cat((data_z, padding), 1)

                    for i in range(16):
                        trans = transforms.Compose([
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            # pretraied model use this
                        ])
                        data_e[i] = trans(data_e[i])
                        data_z[i] = trans(data_z[i])

                    pos = torch.cat((pos[:, 0].view(32, 1), pos[:, 1].view(32, 1), pos[:, 2].view(32, 1)), 1)  # (32,3)
                    i = []
                    j = []
                    for k in range(32):
                        if k % 2 == 0:
                            i.append(k)
                        else:
                            j.append(k)
                    a = torch.LongTensor(i)
                    pos_a = torch.index_select(pos, 0, a)
                    b = torch.LongTensor(j)
                    pos_b = torch.index_select(pos, 0, b)
                    pos_batchsz = pos_b - pos_a
                    pos_batchsz = pos_batchsz.type('torch.FloatTensor')

                    logits_e = model(data_e)
                    # print(logits_e)
                    logits_z = model(data_z)
                    logits_z = logits_z * (-1)
                    loss1 = criteon(logits_e, logits_z)

                    loss2 = criteon(logits_e, pos_batchsz)
                    loss = loss1 + loss2

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_train.append(loss.item())
                print(epoch, 'train loss:', round(np.mean(loss_train).item(), 8))
                loss_list_train.append(round(np.mean(loss_train), 8))
                scheduler.step(loss_list_train[-1])


            else:
                model.train(False)
                print('validation')
                for batchidx, (data, pos) in enumerate(dataloaders[phase]):
                    data_z = torch.zeros(32, 1, 224, 224)
                    data_e = data.view(16, 2, 224, 224)
                    padding = torch.zeros(16, 1, 224, 224)
                    data_e = torch.cat((data_e, padding), 1)
                    for j in range(32):
                        if j % 2 == 0:
                            data_z[j] = data[j + 1]
                        else:
                            data_z[j] = data[j - 1]
                    data_z = data_z.view(16, 2, 224, 224)
                    data_z = torch.cat((data_z, padding), 1)

                    for i in range(16):
                        trans = transforms.Compose([
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            # pretraied model use this
                        ])
                        data_e[i] = trans(data_e[i])
                        data_z[i] = trans(data_z[i])

                    pos = torch.cat((pos[:, 0].view(32, 1), pos[:, 1].view(32, 1), pos[:, 2].view(32, 1)), 1)  # (32,3)
                    i = []
                    j = []
                    for k in range(32):
                        if k % 2 == 0:
                            i.append(k)
                        else:
                            j.append(k)
                    a = torch.LongTensor(i)
                    pos_a = torch.index_select(pos, 0, a)
                    b = torch.LongTensor(j)
                    pos_b = torch.index_select(pos, 0, b)
                    pos_batchsz = pos_b - pos_a
                    pos_batchsz = pos_batchsz.type('torch.FloatTensor')

                    logits_e = model(data_e)
                    # print(logits_e)
                    logits_z = model(data_z)
                    logits_z = logits_z * (-1)
                    loss1 = criteon(logits_e, logits_z)

                    loss2 = criteon(logits_e, pos_batchsz)
                    loss = loss1 + loss2

                    loss_vali.append(loss.item())
                print(epoch, 'validation loss:', round(np.mean(loss_vali).item(), 8))
                loss_list_vali.append(round(np.mean(loss_vali), 8))

                if loss_list_vali[-1] == min(loss_list_vali):
                    print('epoch:',epoch)
                    torch.save({
                        'epoch': epoch,
                        'loss': loss_list_train,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'state_dict': model.state_dict()
                    }, '/home/mohanxu/Desktop/code_mohan2/paras1.pth.tar')
                else:
                    print("don't save net")

    x = np.array(range(0,350))
    plt.plot(x, loss_list_train, color='green', label='training loss')
    plt.plot(x, loss_list_vali, color='red', label='validation loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('MSE-Loss')
    plt.savefig('/home/mohanxu/Desktop/code_mohan2/MSE-train+vali1.png')
    plt.show()
    a = loss_list_train
    for i in range(350):
        a[i] = round(a[i],3)
    plt.hist(a, 350, density = True, histtype = 'bar', facecolor = 'blue', edgecolor = 'black',alpha = 0.5)
    plt.ylabel('Probability')
    plt.savefig('/home/mohanxu/Desktop/code_mohan2/Probability1.png')
    plt.show()
    print('training finish')


def probe(batch_size):
    dataset_test = OCTDataset('/home/mohanxu/Desktop/ophonlas-oct/oct_network_app/test')   #Test
    #dataset_test = OCTDataset('/home/mohanxu/Desktop/code_mohan2/Data_VS/Test2')   #visual servo Dataset
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    checkpoint = torch.load('/home/mohanxu/Desktop/code_mohan2/paras1.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()  # train model to test model
    with torch.no_grad():
            # test
            epoch_losses = []
            for batchidx, (data,pos) in enumerate(dataloader_test):
                data = data.view(1, 2, 224, 224)
                padding = torch.zeros(1, 1, 224, 224)
                data = torch.cat((data, padding), 1)
                for i in range(1):
                    trans = transforms.Compose([
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    data[i] = trans(data[i])

                pos = torch.cat((pos[:, 0].view(2, 1), pos[:, 1].view(2, 1), pos[:, 2].view(2, 1)), 1)  # (2,3)
                a = torch.LongTensor([0])
                pos_a = torch.index_select(pos, 0, a)
                b = torch.LongTensor([1])
                pos_b = torch.index_select(pos, 0, b)
                pos_batchsz = pos_b - pos_a   #Test
                #pos_batchsz = pos_a - pos_b   #visual servo Loss
                pos_batchsz = pos_batchsz.type('torch.FloatTensor')

                # [b, 2]
                logits = model(data)
                # [b]
                loss = criteon(logits, pos_batchsz)
                print(loss)
                epoch_losses.append(loss.item())

                epoch_losses_n = np.mean(epoch_losses)
            print('loss:', epoch_losses)
    x = np.array(range(0, 736))
    #print(x)
    y = epoch_losses
    print('average value:', epoch_losses_n)
    plt.plot(x, y, '-')
    plt.xlabel('batch_size')
    plt.ylabel('MSE-Loss')
    plt.savefig('/home/mohanxu/Desktop/code_mohan2/MSE-Loss-test1.png')
    plt.show()

    print('testing finish')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn both from labeled and unlabeled data.')
    parser.add_argument('--bs', metavar='N', type=int, help='Specify the batch size N', default=32,
                        choices=list(range(10, 500)))
    parser.add_argument('--epochs', metavar='E', type=int, help='Specify the number of epochs E', default=350,
                        choices=list(range(1, 501)))
    args = parser.parse_args()

    #train(batch_size=args.bs, epochs=args.epochs)

    probe(batch_size=2)
