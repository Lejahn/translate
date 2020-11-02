import numpy as np
import os
import torch.nn as nn
from auxil import LangDict, embeddingsLoad, tensorsFromPair, tensor2word
from model_attention import EncoderRNN, DecoderRNN, training, evaluate
from torch import optim
import random


# Define files
spanish_embeddings = 'translate/data/SpanishEmbeddings_c.txt'
english_embeddings = 'translate/data/EnglishEmbeddings100_c.txt'
training_data = 'translate/data/train_ua.txt'


if __name__ == '__main__':

    # Read sentences
    with open(os.path.join(training_data), 'r', encoding='utf-8') as f:
        sentences = f.read().strip().split('\n')
    f.close()

    sentences_list = [i.split('\t') for i in sentences]

    # Instantiate LangDict
    lang_eng = LangDict('eng')
    lang_spa = LangDict('spa')
    for pair in sentences_list:
        lang_eng.read_sentence(pair[0])
        lang_spa.read_sentence(pair[1])


    # Create Embeddings
    # English
    embClass_eng = embeddingsLoad(file=english_embeddings, langDict=lang_eng)
    embClass_eng.txt2dict()
    # Spanish
    embClass_spa = embeddingsLoad(file=spanish_embeddings, langDict=lang_spa)
    embClass_spa.txt2dict()

    # Load Model & Embeddings
    encoder = EncoderRNN(emb_class=embClass_eng)
    decoder = DecoderRNN(emb_class=embClass_spa)

    # Set parameters
    learning_rate = 0.001
    n_iters = 1

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    # Create training pair
    training_pairs = [tensorsFromPair(random.choice(sentences_list)) for i in range(n_iters)]

    for iters in range(1, n_iters + 1):
        training_pair = training_pairs[iters - 1]
        input_tensor, target_tensor = training_pair

        loss = training(input_tensor=input_tensor,
                        target_tensor=target_tensor,
                        encoder=encoder,
                        decoder=decoder,
                        encoder_optimizer=decoder_optimizer,
                        decoder_optimizer=decoder_optimizer,
                        criterion=criterion)
        print(loss)

    # Evaluation
    eval_pairs = [tensorsFromPair(random.choice(sentences_list)) for i in range(1)]

    output_ind = evaluate(encoder=encoder,
                          decoder=decoder,
                          training_pair=eval_pairs[0])

    print(tensor2word(eval_pairs[0][0], lang_eng))
    print(tensor2word(output_ind, lang_spa))






















