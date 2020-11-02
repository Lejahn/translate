import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import time
import matplotlib.pyplot as plt


class EncoderRNN(nn.Module):
    def __init__(self, emb_class):
        super(EncoderRNN, self).__init__()

        # From class embClass
        self.vocab_size = emb_class.__dict__.get('vocab_size')
        self.emb_dim = emb_class.__dict__.get('emb_dim')
        self.embLayer = emb_class.embLayer()

        self.gru = nn.GRU(self.emb_dim, self.emb_dim)


    def forward(self, input, hidden):
        embedded = self.embLayer(input).view(1, 1, -1)
        output = embedded
        # print(output.size())
        output, hidden = self.gru(output, hidden)
        # print(output.size(), hidden.size())
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.emb_dim)


class DecoderRNN(nn.Module):
    def __init__(self, emb_class):
        super(DecoderRNN, self).__init__()

        # From class embClass
        self.vocab_size = emb_class.__dict__.get('vocab_size')
        self.emb_dim = emb_class.__dict__.get('emb_dim')

        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.gru = nn.GRU(self.emb_dim, self.emb_dim)
        self.out = nn.Linear(self.emb_dim, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


def training(input_tensor, target_tensor,
             encoder, decoder,
             encoder_optimizer, decoder_optimizer,
             criterion,
             max_length=100):

    SOS_token = 0
    EOS_token = 1
    # Initialize Hidden Layer
    encoder_hidden = encoder.initHidden()

    # Re-set optimizer
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()


    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # Saves the output for each word
    # encoder_outputs = torch.zeros(max_length, encoder.emb_dim)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        # encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]])

    decoder_hidden = encoder_hidden # Last hidden state

    # plt.plot(decoder_hidden.view(-1))


    # Decoder
    '''
    Decoder outputs prob over vocabulary. The method topk(1) gets the max prob with the pertaining index. 
    The index is then used as input again. 
    '''
    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input

        loss += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == EOS_token:
            break


    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length



def evaluate(encoder, decoder,
             training_pair):
    SOS_token = 0
    EOS_token = 1

    with torch.no_grad():
        input_tensor, target_tensor = training_pair
        encoder_hidden = encoder.initHidden()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            # encoder_outputs = encoder_output[0, 0]

        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor([[SOS_token]])

        output_ind = list()
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            if decoder_input.item() == EOS_token:
                break
            else:
                output_ind.append(topi.item())

    return output_ind


            # print(topi)
            # print(decoder_output.data.topk(1))

            # decoder_output is the prob over the vocabulary!









