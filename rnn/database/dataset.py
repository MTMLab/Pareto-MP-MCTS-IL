import rdkit
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Draw
import pandas as pd
import re
import csv

f = open("dataset_v1.csv", "r")
csvreader = csv.reader(f)
final_list = list(csvreader)[1:]
#
#
def B_not_in_smile(smile):
     if "B" not in smile:
         return True
     else:
         sit = 0
         for i in range(len(smile) - 1):
             if smile[i] == "B" and smile[i + 1] != "r":
                 sit += 1
         if sit > 0:
             return False
         else:
             return True
#
#
def process1(inte):
     if "." not in inte and "se" not in inte and "Na" not in inte and "Se" not in inte and "te" not in inte and \
             "As" not in inte and "p" not in inte and "P" not in inte and "Xe" not in inte and "I" not in inte and \
             "Si" not in inte and "[N+]" not in inte and "V" not in inte and "Al" not in inte and "Zn" not in inte and \
             "[O-]" not in inte and "b" not in inte and "As" not in inte and "+" not in inte and "Ca" not in inte and \
             "Cu" not in inte and "Mg" not in inte and "Li" not in inte and "-" not in inte and "Ni" not in inte and \
             "C1" not in inte and "C2" not in inte and "C3" not in inte and "C4" not in inte and \
             "c1" not in inte and "." not in inte and "c2" not in inte and "c3" not in inte and "c4" not in inte and "-" not in inte and\
             "He" not in inte and B_not_in_smile(inte):
         return True
     else:
         return False

#####################################

#f=open("dataset1.txt","w")
#for i in final_list:
#    inte=i[0]
#    if process1(inte):
#        try:
#            mol=Chem.MolFromSmiles(inte)
#            if mol!=None and sum(atom.GetFormalCharge() for atom in mol.GetAtoms())==0:
#                smi=Chem.MolToSmiles(mol)
#                print(smi,file=f)
#        except:
#            pass
#f.close()

####################################
# # symbols
f=open("dataset1.txt","r")
symbols=set()
for id,i in enumerate(f):
    if id %10000==0:
        print(id)
    symbols=symbols|set(i[:-1].replace("Cl","").replace("Br",""))
#
symbols=list(symbols)
symbols.sort()
symbols+=["Cl","Br"]
print(symbols)
print(symbols,file=open("symbols.txt",'w'))
f.close()

##################################
## write dataset2 and get onehot num, data shape
#
symbols=["^"]+eval(open("symbols.txt","r").read())+["$"]
symbol2label={i:id for id,i in enumerate(symbols)}
label2symbol={id:i for id,i in enumerate(symbols)}
onehot=len(symbols)
def smile_split(smile):
     base1=symbols[:-3]+["$"]
     base2=["Cl","Br"]
     smile_s_l = []
     i=0
     while i < len(smile):
         if smile[i] in base1 and smile[i:i+2] not in base2:
             smile_s_l.append(smile[i])
         elif smile[i:i+2] in base2:
             smile_s_l.append(smile[i:i+2])
             i+=1
         elif smile[i]=="[":
             for j in range(i+2,len(smile)):
                 if smile[j]=="]":
                     smile_s_l.append(smile[i:j+1])
                     i = j
                     break
         else:
             smile_s_l.append(smile[i])
         i+=1
     return smile_s_l
#
f1=open("dataset3.txt","r")
f2=open("dataset3label.txt","w")
for id,i in enumerate(f1):
     if id%10000==0:
         print(id)
     print([symbol2label[j] for j in smile_split(i[:-1])],file=f2)
f1.close()
f2.close()
f1=open("dataset3label.txt","r")
maxnum=max(len(eval(i)) for i in f1)
print(maxnum)

############################################

import torch
symbols=["^"]+eval(open("symbols.txt","r").read())+["$"]
symbol2label={i:id for id,i in enumerate(symbols)}
label2symbol={id:i for id,i in enumerate(symbols)}
seqlen=65
onehot_num=len(symbols)
print(onehot_num)
f=open("dataset3label.txt","r").readlines()
samples=len(f)
data=torch.zeros([samples,seqlen+1])
#
for id, i in enumerate(f):
     if id %10000==0:
         print(id)
     l=eval(i)
     tmp=l+[onehot_num-1 for i in range(seqlen-len(l))]
     data[id]=torch.tensor([0]+tmp)
data=data.byte()
print(data[0])
torch.save(data,"dataset3")


import torch

a = torch.tensor([0, 2, 3.])
print(a / a.sum(dim=-1))
#
data=torch.load("dataset3")
print(data.shape)
print(torch.nn.functional.one_hot(data.long(),23).shape)

