import os
import numpy as np
import torch.nn as nn
import torch
from model import EncoderRNN


class embeddingsLoad:
    '''
    Loads the embeddings.
    '''
    def __init__(self, file, langDict):
        self.file = file
        self.embeddings_index = dict()
        self.word2index = langDict.word2index
        self.vocab_size = len(langDict.word2index) + 2
        self.emb_dim = 0

    def txt2dict(self):
        '''
        Reads the glove word embeddings and creates a dictionary.
        :return: Dict with words as keys and embeddings as items.
        '''
        f = open(self.file, 'r', encoding='utf-8')
        for idx, line in enumerate(f):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
            if idx == 0:
                self.emb_dim = coefs.shape[0]
                print('Embedding has a shape of {}'.format(self.emb_dim))

    def c_embedding_matrix(self):
        '''
        For word in lang dict stacks the pertaining word embedding.
        :return: Embedding matrix
        '''
        # +2 for the EOS tokens
        embedding_matrix = np.zeros((self.vocab_size + 2, self.emb_dim))
        for word, idx in self.word2index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[idx] = embedding_vector
        return embedding_matrix

    def fillZeros(self):
        '''
        :return: Embedding np.array with the not occurring words, i.e. all values in row = 0,
                 such as the EOS tokens filled with random values
        '''
        emb_mat = self.c_embedding_matrix()
        zeroidx = list()
        for idx, row in enumerate(emb_mat):
            if not np.any(row):
                emb_mat[idx] = np.random.rand(self.emb_dim)
                zeroidx.append(idx)
        return emb_mat

    def embLayer(self, non_trainable=False):
        '''
        Transforms the numpy array to a pytorch embedding.
        :return: Pytorch Emebdding
        '''
        emb_matrix = self.fillZeros()
        if isinstance(emb_matrix, np.ndarray):
            emb_matrix = torch.from_numpy(emb_matrix)
        emb_layer = nn.Embedding(self.vocab_size + 2, self.emb_dim)
        emb_layer.load_state_dict({'weight': emb_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False
        return emb_layer


class LangDict:
    '''
    Class for creating word2index, index2word dictionaries.
    '''

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.word_count = {}
        self.n_word = 2

    def read_sentence(self, sentence):
        '''
        Wrapper for read_word.
        '''
        for word in sentence.split(" "):
            self.read_word(word)

    def read_word(self, word):
        '''
        Adds word to index.
        '''
        if word not in self.word2index:
            self.word2index[word] = self.n_word
            self.index2word[self.n_word] = word
            self.word_count[word] = 1
            self.n_word += 1
        else:
            self.word_count[word] += 1


def word2tensor(sentence, lookup):
    '''
    Transforms sentence to pytorch tensor.
    :sentence: Sentence in words to be transformed.
    :lookup: Lookup class with the specific language.
    :return: Pytorch tensor
    '''
    EOS_token = 1
    index = [lookup.word2index[word] for word in sentence.split(' ')]
    index.append(EOS_token)
    tensor = torch.tensor(index, dtype=torch.long).view(-1, 1)
    return tensor


def tensorsFromPair(sentence_pair, lookup_inp, lookup_out):
    '''
    Wrapper for word2tensor for each pair.
    :sentence_pair: Nested list of sentence pairs.
    :lookup_inp, lookup_out: Lookup class with the specific language.
    :return: Input & target sentences as tensor.
    '''
    inp_tensor = word2tensor(sentence_pair[0], lookup=lookup_inp)
    tar_tensor = word2tensor(sentence_pair[1], lookup=lookup_out)
    return inp_tensor, tar_tensor


def tensor2word(tensor, lookup):
    '''
    Transforms a tensor back to a sentence.
    :tensor: Pytorch tensor.
    :lookup: Lookup class with the specific language.
    :return: Sentence.
    '''
    if not isinstance(tensor, list):
        tensor2list = tensor.squeeze().tolist()
    else:
        tensor2list = tensor
    sentence = [lookup.index2word[i] for i in tensor2list]
    return sentence





# if __name__ == '__main__':
#     training_data = 'translate/data/train_ue.txt'
#     with open(training_data, 'r', encoding='utf-8') as f:
#         sentences = f.read().strip().split('\n')
#     f.close()
#
#     sentences_list = [i.split('\t') for i in sentences]
#     # engDict = LangDict('eng')
#     spaDict = LangDict('spa')
#     for pair in sentences_list:
#        # engDict.read_sentence(pair[0])
#         spaDict.read_sentence((pair[1]))
#     # print('Length of training sentences is {}'.format(len(sentences_list)))
#     # print('Dictionary as {} word keys'.format(len(list(engDict.index2word.keys()))))
#
#
#
#     # Load english embeddings
#     # english_embeddings = 'translate/data/EnglishEmbeddings_c.txt'
#     # embClass = embeddingsLoad(file=english_embeddings, langDict=engDict)
#     # embClass.txt2dict()
#     # layer = embClass.embLayer()
#     # print('Embedding dimension is {}'.format(embClass.__dict__.get('emb_dim')))
#     # print('Vocabulary dimension is {}'.format(embClass.__dict__.get('vocab_size')))
#
#
#     # Load spanish embeddings
#     spanish_embeddings = 'translate/data/SpanishEmbeddings_c.txt'
#     embClass = embeddingsLoad(file=spanish_embeddings, langDict=spaDict)
#     embClass.txt2dict()
#     layer = embClass.embLayer()
#     print('Embedding dimension is {}'.format(embClass.__dict__.get('emb_dim')))
#     print('Vocabulary dimension is {}'.format(embClass.__dict__.get('vocab_size')))
#
#
#
#     mbMatrix = embClass.fillZeros()
#     # not_occuring = []
#     # for idx, row in enumerate(embMatrix):
#     #     if not np.any(row):
#     #         not_occuring.append(idx)
#     # print(len(not_occuring))


