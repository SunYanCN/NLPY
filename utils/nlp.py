from nltk.corpus import stopwords
import numpy as np
from tqdm import tqdm
import unicodedata
import re
import operator
from sklearn.metrics import f1_score

stop = set(stopwords.words('english'))
stop_dict = dict([[word,word]for word in stop])


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def remove_stopwords(data:list):
    after_remove = list()
    data_num = len(data)
    for i in range(data_num):
        clean_sentence = list()
        for word in data[i].split():
            if word not in stop_dict:
                clean_sentence.append(word)
        after_remove.append(" ".join(clean_sentence))
    return after_remove

def text_length_distribution(data:list):
    lengths = [len(i.split()) for i in data]
    max_length = np.max(lengths)
    print("Max length:", max_length)
    min_length = np.min(lengths)
    print("Min length:", min_length)
    ave_length = np.mean(lengths)
    print("Average length:", ave_length)

def word_dict(word_index):
    word2idx = word_index
    idx2word = {idx: word for word, idx in zip(word2idx.keys(), word2idx.values())}
    return word2idx,idx2word


def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile, 'r')
    model = {}
    for line in tqdm(f):
        splitLine = line.split(" ")
        word = splitLine[0]
        embedding = np.asarray(splitLine[1:], dtype='float32')
        model[word] = embedding
    f.close()
    print("Done.", len(model), " words loaded!")
    return model


def split_train_val(data,labels,val_size,seed=1234):
    np.random.seed(seed)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(val_size * data.shape[0])
    X_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    X_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]

    return (X_train, y_train, X_val, y_val)

def cnen_text(text:str):
    text = re.compile(u'[\u4E00-\u9FA5|\s\w]').findall(text)
    text = ''.join(text)
    return text

def build_vocab(texts):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

def check_coverage(vocab, embeddings_index):
    known_words = {}
    unknown_words = {}
    nb_known_words = 0
    nb_unknown_words = 0
    for word in vocab.keys():
        try:
            known_words[word] = embeddings_index[word]
            nb_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]
            pass
    print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]
    return unknown_words

def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in [i * 0.01 for i in range(10,70)]:
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result

if __name__ == '__main__':
    data = ["he is a boy","she is a gril"]
    data = remove_stopwords(data)
    print(data)
    ave_length = cala_al(data)
    text = cnen_text("我爱……×）absc")
    print(text)
