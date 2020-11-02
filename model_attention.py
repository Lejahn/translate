import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F



class EncoderRNN(nn.Module):
    def __init__(self, emb_class):
        super(EncoderRNN, self).__init__()

        self.vocab_size = emb_class.__dict__.get('vocab_size')
        self.emb_dim = emb_class.__dict__.get('emb_dim')
        self.embLayer = emb_class.embLayer()
        self.gru = nn.GRU(self.emb_dim, self.emb_dim)


    def forward(self, input, hidden):

        embedded = self.embLayer(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.emb_dim)


class Attention(nn.Module):
    def __init__(self, emb_dim):
        super(Attention, self).__init__()

        self.emb_dim = emb_dim
        self.v = nn.Parameter(torch.rand(self.emb_dim))  # [emb_dim]
        self.att_para = nn.Linear(self.emb_dim * 2, self.emb_dim)

    def forward(self, hidden, encoder_outputs):
        '''
        params:
              hidden: [b, n_layer=1, emb_dim]
              encoder_outputs: [seq_len, emb_dim]
        :return: att_weights: [b, seq_len]
        '''
        seq_len = encoder_outputs.size()[0]
        hidden = hidden.repeat(1, seq_len, 1)  # [b, seq_len, emb]
        alpha = torch.tanh(self.att_para(torch.cat([hidden, encoder_outputs.unsqueeze(0)], dim=2)))  # [b, seq_len, emb]
        alpha = alpha.permute(0, 2, 1)  # [b, emb, seq_len]

        v = self.v.repeat(1, 1).unsqueeze(1)  # [b, 1, emb]
        e = torch.bmm(v, alpha).squeeze(1)  # [b, seq_len]
        attn_weights = F.softmax(e, dim=1)  # [b, seq_len]
        return attn_weights


class DecoderRNN(nn.Module):
    def __init__(self, emb_class):  # , encoder_outputs):   # Problem input encoder_outputs. Need for seq_len
        super(DecoderRNN, self).__init__()

        self.vocab_size = emb_class.__dict__.get('vocab_size')
        self.emb_dim = emb_class.__dict__.get('emb_dim')
        self.dropout = nn.Dropout(0.1)

        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim)
        self.att_comb = nn.Linear(self.emb_dim * 2, self.emb_dim)

        self.gru = nn.GRU(self.emb_dim, self.emb_dim)
        self.out = nn.Linear(self.emb_dim, self.vocab_size)  # <- error here/ Must be
        self.att_weights = Attention(emb_dim=self.emb_dim)  # , seq_len=self.seq_len)
        self.attn_combine = nn.Linear(self.emb_dim * 2, self.emb_dim)

    def forward(self, input, hidden, encoder_outputs):
        '''
        PARAMS:
            input: [idx]
            hidden: [b, n_layer=1, emb_dim]
            encoder_outputs: [seq_len, emb_dim]
        '''
        # Embeddings & Dropout
        embeddings = self.embeddings(input).view(1, 1, -1)  # [b, seq_len_de, emb_dim]
        embeddings = self.dropout(embeddings)

        # Attention Weights
        att_weights = self.att_weights(hidden, encoder_outputs)  # [b, seq_len]
        att_dp = torch.bmm(att_weights.unsqueeze(1), encoder_outputs.unsqueeze(0))  # [b, seq_len_de, emb_dim]

        input = torch.cat([embeddings, att_dp], dim=2)
        input = self.attn_combine(input)

        input = F.relu(input)
        output, hidden = self.gru(input, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden


def training(input_tensor, target_tensor,
             encoder, decoder,
             encoder_optimizer, decoder_optimizer,
             criterion,
             max_length=250):
    '''
    Training loop.
    '''

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
    encoder_outputs = torch.zeros(max_length, encoder.emb_dim)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]])
    decoder_hidden = encoder_hidden # Last hidden state
    # print('Encoder_outputs have a shape of {}'.format(encoder_outputs.size()))
    # # print('Decoder_hidden have a shape of {}'.format(decoder_hidden.size()))


    # Decoder
    '''
    Decoder outputs prob over vocabulary. The method topk(1) gets the max prob with the pertaining index. 
    The index is then used as input again. 
    '''
    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

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
             training_pair,
             max_length=250):
    '''
    Evaluates the model.
    '''
    SOS_token = 0
    EOS_token = 1

    with torch.no_grad():
        input_tensor, target_tensor = training_pair

        encoder_hidden = encoder.initHidden()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        encoder_outputs = torch.zeros(max_length, encoder.emb_dim)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor([[SOS_token]])

        output_ind = list()
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            if decoder_input.item() == EOS_token:
                break
            else:
                output_ind.append(topi.item())

    return output_ind