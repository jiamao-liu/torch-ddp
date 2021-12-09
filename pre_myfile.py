import tqdm
import pickle

def main_function(path,new_path):
    lines = open(path, "r", encoding="utf-8").readlines()
    new_str=[]
    for n in range(0,len(lines),3):
        ccc=lines[n].split("$T$")
        if len(ccc)==2:
            new_str.append(lines[n])
            new_str.append(lines[n+1])
            new_str.append(lines[n+2])
    new_file=open(new_path,"w",encoding="utf-8")
    for n in new_str:
        new_file.write(n)
    new_file.close()


train_path="./data/train.raw"
test_path="./data/test.raw"
new_train_path="./data/new_train.raw"
new_test_path="./data/new_test.raw"
main_function(train_path,new_train_path)
main_function(test_path,new_test_path)

