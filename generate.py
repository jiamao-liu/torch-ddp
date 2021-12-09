# -*- coding: utf-8 -*-

from transformers import BertTokenizer
import numpy as np
import spacy
import pickle
import tqdm
import re
nlp = spacy.load('en_core_web_sm')
#########################tool

def fenci(text):
    document=nlp(re.sub(r' {2,}',' ',text.strip()))
    return [token.text for token in document]

def concat(texts,aspect):
    source=""
    split_num=0
    for i,text in enumerate(texts):
        source +=text
        split_num+=len(fenci(text))
        if i<len(texts)-1:
            source = source+' ' + aspect + ' '
            split_num += len(fenci(aspect))
    if split_num!=len(fenci(source.strip())):
        print("貌似出问题了！！！")
        input("111111111111111111111")
    return re.sub(r' {2,}',' ',source.strip())

def fenju(line):
    text = []
    for s in line.split("$T$"):
        text.append(s.lower().strip())
    return text

#########################func
def update_edge(text,vocab):
    doc=nlp(text)
    for word_nlp in doc:
        if word_nlp.dep_ not in vocab:
            vocab[word_nlp.dep_]=len(vocab)
    return 0


def dependency_adj_matrix(text):
    document=nlp(text.strip())
    seq_len=len(fenci(text))
    adj_matrix=np.zeros((seq_len,seq_len)).astype("int32")
    undir_adj_matrix=np.zeros((seq_len,seq_len)).astype("int32")
    if len(document)!=seq_len:
        print("111111111111111111111111111111")
        return 0
    for word in document:
        if word.i>=seq_len:
            print("11111111111111111111111111111111111")
            return 0
        else:
            undir_adj_matrix[word.i][word.i]=1
            adj_matrix[word.i][word.i]=1
            for child in word.children:
                if child.i <seq_len:
                    adj_matrix[word.i][child.i]=1
                    undir_adj_matrix[word.i][child.i] = 1
                    undir_adj_matrix[child.i][word.i] = 1
    return adj_matrix,undir_adj_matrix

def generate_edge(lines_train,lines_test,edge_vocab):
    for i in tqdm.tqdm(range(0,len(lines_train),3)):
        text=fenju(lines_train[i])
        aspect=lines_train[i+1].lower().strip()
        update_edge(concat(text,aspect),edge_vocab)
    for i in tqdm.tqdm(range(0,len(lines_test),3)):
        text=fenju(lines_test[i])
        aspect=lines_test[i+1].lower().strip()
        update_edge(concat(text,aspect),edge_vocab)
    return 0

def generate_graph(lines,fout_graph,undir_fout_graph):
    idx_to_undir_graph={}
    idx_to_graph={}
    for i in tqdm.tqdm(range(0,len(lines),3)):
        text=fenju(lines[i])
        aspect = lines[i + 1].lower().strip()
        adj_matrix,undir_adj_matrix=dependency_adj_matrix(concat(text,aspect))
        idx_to_graph[i]=adj_matrix
        idx_to_undir_graph[i]=undir_adj_matrix
    pickle.dump(idx_to_graph,fout_graph)
    pickle.dump(idx_to_undir_graph,undir_fout_graph)
    fout_graph.close()
    undir_fout_graph.close()
    return 0

#########################main
def main_function(file_path_train,file_path_test):
    #edge_vocab={"<pad>":0,"<unk>":1}
    file_train=open(file_path_train,"r",encoding="utf-8",newline="\n",errors="ignore")
    file_test=open(file_path_test,"r",encoding="utf-8",newline="\n",errors="ignore")
    lines_train=file_train.readlines()
    lines_test=file_test.readlines()
    file_test.close()
    file_train.close()
    fout_graph_train=open(file_path_train+"_graph","wb")
    fout_graph_test=open(file_path_test+"_graph","wb")
    undir_fout_graph_train=open(file_path_train+"_undir_graph","wb")
    undir_fout_graph_test=open(file_path_test+"_undir_graph","wb")
    #####生成边词典
    #generate_edge(lines_train,lines_test,edge_vocab)
    #####生成依赖关系图并且保存
    generate_graph(lines_train,fout_graph_train,undir_fout_graph_train)
    generate_graph(lines_test,fout_graph_test,undir_fout_graph_test)
    return 0

if __name__=="__main__":
    main_function(file_path_train="./data/new_train.raw",file_path_test="./data/new_test.raw")
