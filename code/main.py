from param import parameter_parser
from MMGCN import MMGCN
from dataprocessing import data_pro
import torch


def train(model, train_data, optimizer, opt):
    model.train()
    lossRes=[]
    for epoch in range(0, opt.epoch):
        model.zero_grad()
        score = model(train_data)
        loss = torch.nn.MSELoss(reduction='mean')
        loss = loss(score, train_data['md_p'].cpu())
        loss.backward()
        optimizer.step()
        lossRes.append(loss.item())
        print(loss.item())

    # 写入文件
    with open('loss.txt', 'w') as f:
        for i in range(len(lossRes)):
            f.write(str(lossRes[i]) + '\n')
    score = score.detach().cpu().numpy()
    scoremin, scoremax = score.min(), score.max()
    score = (score - scoremin) / (scoremax - scoremin)
    return score

def main():
    args = parameter_parser()
    dataset = data_pro(args)
    train_data = dataset
    model = MMGCN(args)
    model.cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    score = train(model, train_data, optimizer, args)

if __name__ == "__main__":
    main()
