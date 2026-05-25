import torch
import random
from train_model import net

seq_len=65
onehot_len=19

model=net()
parameters=torch.load("../model/rnn_model_parameters",map_location='cpu')["model"]
model.load_state_dict(parameters)
model.eval()


symbols=["^",'#', '(', ')', '1', '2', '=', '@', 'C', 'F', 'H', 'N', 'O', 'S', '[', ']', 'Cl', 'Br', "$"]
symbol2label={i:id for id,i in enumerate(symbols)}
label2symbol={id:i for id,i in enumerate(symbols)}


def list2string(l):
    return ("".join(label2symbol[i] for i in l)).strip("^").strip("$")


def onehot_j_top3(o,j): # attention (batch onehot seqlen)
    _,index=o.topk(3,dim=1)

    return index[0,:,j].tolist()



def next_labels(l):

    list_len = len(l)
    try:
        seq_len=65
        assert list_len <= seq_len+1
        ll = torch.tensor([l + [onehot_len-1 for i in range(seq_len+1 - list_len)]])
        predict_onehot, target = model(ll)
        return onehot_j_top3(predict_onehot,list_len-1)
    except:
        seq_len=200
        assert list_len <= seq_len+1
        ll = torch.tensor([l + [onehot_len-1 for i in range(seq_len+1 - list_len)]])
        predict_onehot, target = model(ll)
        return onehot_j_top3(predict_onehot,list_len-1)



def next_lists(l):
    labels = next_labels(l)
    ls=[0 for i in range(3)]
    for i in range(3):
        ls[i]=l+[labels[i]]
    return ls

def next_list(l,rand=True):
    labels=next_labels(l)
    if rand:
        return l+[labels[random.randint(0,2)]]
    else:
        return l+[labels[0]]

def final_list(l):
    tmp=next_list(l)
    while tmp[-1]!=onehot_len-1:
        tmp=next_list(tmp)
    return tmp

if __name__=="__main__":
    print(symbols)
    print(next_labels([0]))
    print(next_lists([0]))
    print(list2string(final_list([0,11])))

    for i in range(20):
        print(final_list([0,i]))






