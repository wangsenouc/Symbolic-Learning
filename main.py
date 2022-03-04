import torch.nn as nn
import torch
import numpy as np
import argparse


def gen_data(num=128):
    x = torch.randint(1, 99, (num, 2))
    y = x[:, 1]
    return x.float(), y.float()


def gen_nm_data(min_, max_, size=1000):
    train_y = []
    train_x = torch.randint(min_, max_, (size, 2))
    for d in train_x:
        train_y.append([i for i in range(d[0]+1, d[0]+d[1]+1)])

    train_x = train_x.float()
    return train_x, train_y


def train_length_net(model, train_x, train_y, num_epoch=80):
    batch_size = 1
    learning_rate = 0.00001
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(num_epoch):
        losses = []
        for idx in range(0, len(train_y), batch_size):
            if idx + batch_size < len(train_y):
                x = train_x[idx: idx + batch_size]
                y = train_y[idx: idx + batch_size]
            else:
                x = train_x[idx:]
                y = train_y[idx:]
            outputs = model(x)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if (epoch + 1) % 5 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epoch, np.average(losses)))
            losses.clear()
    print(model.weight)
    print(model.bias)
    return model


def train(train_x, train_y):
    transformer_model = nn.Transformer(nhead=1, num_encoder_layers=12, d_model=1, batch_first=True)
    transformer_model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(transformer_model.parameters(), lr=1.0)
    losses = []
    epochs = 10
    for epoch in range(epochs):
        for src, label in zip(train_x, train_y):
            src = src.view(1, 2, 1)
            label = torch.tensor(label, dtype=torch.float)
            label = label.view(1, -1, 1)
            out = transformer_model(src, label)
            loss = criterion(label, out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if (len(losses) + 1) % 100 == 0:
                print(np.average(losses))
                losses.clear()
        print(epoch)

    return transformer_model


def test(transformer_model, test_x, test_y, length_net):
    transformer_model.eval()
    criterion = nn.MSELoss()
    length_net.eval()
    losses = []
    for src, label in zip(test_x, test_y):
        out_len = length_net(src).item()
        src = src.view(1, 2, 1)
        label = torch.tensor(label, dtype=torch.float)
        label = label.view(1, -1, 1)
        tgt = torch.ones((1, round(out_len), 1))
        out = transformer_model(src, tgt)
        loss = criterion(label, out)
        losses.append(loss.item())
    print(np.average(losses))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--minmn', type=int, default=1)
    parser.add_argument('--maxmn', type=int, default=100)
    parser.add_argument('--data-num', type=int, default=1000)
    args = parser.parse_args()
    return args


def main():
    x, y = gen_data(10000)
    length_net = nn.Linear(2, 1)
    length_net = train_length_net(length_net, x, y)
    args = get_args()
    min_ = args.minmn
    max_ = args.maxmn
    data_num = args.data_num
    train_x, train_y = gen_nm_data(min_, max_, data_num)
    transformer_model = train(train_x, train_y)
    test_x, test_y = gen_nm_data(min_, max_, 100)
    test(transformer_model, test_x, test_y, length_net)


if __name__ == '__main__':
    main()
