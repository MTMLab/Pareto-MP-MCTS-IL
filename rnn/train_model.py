import torch
import torch.utils.data as Data

seq_len=65
onehot_len=19
embed_size=256
hidden_size=512
rnn_layers_num=2
dropout=0
batch_size=128
learning_rate=3e-4



class net(torch.nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.embed=torch.nn.Embedding(onehot_len,embed_size)
        self.rnn=torch.nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers_num,
            bidirectional=False,
            dropout=dropout,
            batch_first=True,
            bias=True
        )
        self.linear=torch.nn.Linear(hidden_size,onehot_len)

    def forward(self,data):
        input=self.embed(data[:,:-1])
        self.rnn.flatten_parameters()
        # input_onehot=torch.nn.functional.one_hot(data,onehot_len).float()
        output_onehot=self.linear(self.rnn(input)[0]).permute(dims=[0,2,1])
        target=data[:,1:]
        return output_onehot,target


def train(epoch_num=5):
    dataset = Data.TensorDataset(torch.load("dataset3").long())
    train_set = Data.DataLoader(dataset, batch_size=batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=net().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    criteria=torch.nn.CrossEntropyLoss()
    # criteria=torch.nn.NLLLoss()
    epoch_id=0
    loss_history = []
    try:
        model_parameters=torch.load("rnn_model_parameters")
        model.load_state_dict(model_parameters["model"])
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
        optimizer.load_state_dict(model_parameters["optimizer"])
        epoch_id=model_parameters["epoch_id"]
        loss_history=model_parameters["loss_history"]
    except:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
        pass

    for epoch in range(epoch_num):
        for batch_id, data in enumerate(train_set):
            input=data[0].to(device)
            output_onehot, target=model(input)
            loss=criteria(output_onehot,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_=loss.detach_().mean().item()
            print("epoch id:",epoch_id,"batch id:",batch_id,"loss:",loss_)
            loss_history.append(loss_)
        epoch_id += 1

    model_parameters={
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch_id":epoch_id,
        "loss_history":loss_history,
    }
    torch.save(model_parameters,"rnn_model_parameters")


if __name__=="__main__":
    for i in range(20):
        train(5)
#
# a=torch.randint(0,26,[100,65])
# print(torch.nn.Embedding(27,100)(a).shape)



