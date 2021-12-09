from transformers import BertTokenizer
from generate import fenci,concat,fenju
import pickle
import numpy as np
import tqdm
import os



def load_word_vector(path,vocab,embed_dim):
    lines=open(path,"r",encoding="utf-8",newline="\n",errors="ignore").readlines()
    word_vec = {}
    for n in tqdm.tqdm(range(len(lines))):
        tokens = lines[n].rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        # ：-300 代表去掉最后三百个 剩下的    -300: 代表取最后三百个    所以结果就是 word和 vec
        if word in vocab.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec

def build_embedding_matrix(diy_vocab,embed_dim,project):
    embedding_matrix_file_name = '{0}_embedding_matrix.pkl'.format(str(project[0]))
    if os.path.exists(embedding_matrix_file_name):
        print(' embedding_matrix已经存在，直接加载:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('加载glove ...')
        embedding_matrix=np.zeros((len(diy_vocab),embed_dim))
        embedding_matrix[1, :] = np.random.uniform(-1 / np.sqrt(embed_dim), 1 / np.sqrt(embed_dim), (1, embed_dim))
        word_vec=load_word_vector('./glove/glove.840B.300d.txt',vocab=diy_vocab,embed_dim=embed_dim)
        print('建立embedding_matrix:', embedding_matrix_file_name)
        for word,index in diy_vocab.items():  # word vector 是 word 和向量对应  vocab是word 和id 对应
            vec=word_vec.get(word)
            #print("+++", index, "+++", word,"+++",vec)
            if vec is not None:
                embedding_matrix[index]=vec
    pickle.dump(embedding_matrix,open(embedding_matrix_file_name,"wb"))
    return embedding_matrix

class Diy_tokenizer(object):
    def __init__(self,word_to_index=None,):
        if word_to_index is None:
            self.word_to_index={"<pad>":0, '<unk>': 1}
            self.index_to_word={0: '<pad>', 1: '<unk>'}
            self.index = 2
        else:
            self.word_to_index=word_to_index
            self.index_to_word = {v:k for k,v in word_to_index.items()}
        #self.bert = BertTokenizer.from_pretrained("bert-base-uncased")
        #self.bert.do_basic_tokenize = False  ################buzhidao you shenme yohng

    def fit_on_text(self,text):
        text=text.lower().strip()
        words=fenci(text)
        for word in words:
            if word not in self.word_to_index:
                self.word_to_index[word]=self.index
                self.index_to_word[self.index]=word
                self.index+=1

    def text_to_sequence(self,text,is_aspect=False):
        text=text.lower().strip()
        words=fenci(text)
        unknownidx = 1
        sequence = [self.word_to_index[w] if w in self.word_to_index else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence

def span(texts,aspect):
    aspect_len=len(fenci(aspect))
    spans=[]
    move_point=0
    for index,text in enumerate(texts):
        text_len=len(fenci(text))
        move_point+=text_len
        text_len=move_point
        if index < len(texts)-1:
            move_point+=aspect_len
            spans.append([text_len,move_point])
    return spans

class ABSADatesetReader(object):
    def __init__(self,project,embed_dim=300):
        self.diy_tokenizer=Diy_tokenizer()
        text = ABSADatesetReader.__read_text__([project[1], project[2]])
        self.diy_tokenizer.fit_on_text(text)
        c=self.diy_tokenizer.word_to_index["i"]
        print(c)
        self.embedding_matrix= build_embedding_matrix(self.diy_tokenizer.word_to_index, embed_dim=embed_dim,project=project)
        self.train_data=ABSADatesetReader.__read_data__(project[1],self.diy_tokenizer)
        self.test_data=ABSADatesetReader.__read_data__(project[2],self.diy_tokenizer)

    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "
        return text

    @staticmethod
    def __read_data__(data_type,diy_tokenizer):
        fin=open(data_type,"r",encoding="utf-8",newline="\n",errors="ignore")
        lines=fin.readlines()
        fin.close()
        index_to_graph=pickle.load(open(data_type+'_graph', 'rb'))
        index_to_undir_graph=pickle.load(open(data_type+'_undir_graph', 'rb'))
        all_data=[]
        for i in tqdm.tqdm(range(0,len(lines),3)):
            text=fenju(lines[i])
            aspect=lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            concats=concat(text,aspect)
            text_indices = diy_tokenizer.text_to_sequence(text[0]+" "+text[1])
            cattext_indices = diy_tokenizer.text_to_sequence(concats)
            left_indices= diy_tokenizer.text_to_sequence(text[0])
            right_indices= diy_tokenizer.text_to_sequence(text[1])
            aspect_indices= diy_tokenizer.text_to_sequence(aspect,is_aspect=True)
            span_out=span(text,aspect)
            polarity = int(polarity)+1
            dependency_graph = index_to_graph[i]
            undir_dependency_graph = index_to_undir_graph[i]
            data = {
                'text': fenci(concats.lower().strip()),
                'aspect': fenci(aspect),
                'text_indices': text_indices,   ###  原文
                'cattext_indices': cattext_indices,    ####   cat之后的结果
                'left_indices': left_indices,
                "right_indices": right_indices,
                'aspect_indices': aspect_indices,
                'span_out': span_out,     ### 如果第一句话有十个字  asp的长度是3  那么结果就是  10， 13
                'polarity': polarity,
                'dependency_graph': dependency_graph,
                "undir_dependency_graph":undir_dependency_graph,
            }
            all_data.append(data)
        return all_data

class ABSADataset(object):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    path=["jiamao","./data/new_train.raw","./data/new_test.raw",]
    a=ABSADatesetReader(project=path)
    with open("./data/project.pkl","wb") as f:
        pickle.dump(a,f)