import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from network.english import EnglishLSTM, EnglishCNN
from network.en_rythm import EnglishRythmLSTM
from network.japanese import JapaneseLSTM
from network.ja_rythm import JapaneseRythmLSTM
from util.data import DataManager
import sys

from chainer import Variable
from chainer import optimizers
from chainer import serializers
import chainer.functions as F


###
# English -> LSTM
###
def train(params):
    en_model = EnglishLSTM(len(params['en_list']))
    en_rythm_model = EnglishRythmLSTM(len(params['en_rythm_list']))
    ja_model = JapaneseLSTM(len(params['ja_list']))
    ja_rythm_model = JapaneseRythmLSTM(len(params['ja_rythm_list']))
    data = {
        # 'english': en_model.get_train_data(params['english'], params['batch_size']), #並列
        # 'en_rythm': en_model.get_train_data(params['en_rythm'], params['batch_size'])
        'english' : params['english'],
        'en_rythm': params['en_rythm'],
        'japanese' : params['japanese'],
        'ja_rythm': params['ja_rythm'],
    }

    # 最適化アルゴリズムにAdamを採用
    optimizer = [
        optimizers.Adam().setup(en_model),
        optimizers.Adam().setup(en_rythm_model),
        optimizers.Adam().setup(ja_model),
        optimizers.Adam().setup(ja_rythm_model),
    ]

    loss_list = []
    step = []
    for epoch in range(params['epoch_num']):
        print("epoch: %d" % (epoch+1))
        loss = 0.0
        # 英語歌詞の学習
        en_model.reset()
        for index, (en_phrase, en_rythm_phrase, ja_phrase, ja_rythm_phrase) in enumerate(zip(data['english'], data['en_rythm'], data['japanese'], data['ja_rythm'])):
            # 曲が違う場合は状態をリセット
            if len(en_phrase) == 0:
                en_model.reset()
                en_rythm_model.reset()
                continue
            # if len(en_rythm_phrase) == 0:
            #     en_rythm_model.reset()
            #     continue
            # if len(ja_phrase) == 0:
            #     ja_model.reset()
            #     continue
            # 英語の歌詞
            for word in en_phrase:
                y_en = en_model.forward(word, params['en_list'])
            # 英語のリズム
            for rythm in en_rythm_phrase:
                y_en_rythm = en_rythm_model.forward(rythm, params['en_rythm_list'])

            # 出力を足し合わせる
            h = y_en + y_en_rythm

            # hから日本語の1単語目を推測
            tx = Variable(np.array([params['ja_list'][ja_phrase[0]]], dtype=np.int32))
            loss += F.softmax_cross_entropy(ja_model.predict(h), tx)
            # 足し合わせた出力から日本語を出力
            for index, word in enumerate(ja_phrase):
                y_ja = ja_model.forward(word, params['ja_list'])
                if word != '<eos>':
                    tx = Variable(np.array([params['ja_list'][ja_phrase[index+1]]], dtype=np.int32))
                    # print(y_ja, tx)
                    loss += F.softmax_cross_entropy(y_ja, tx)

            # hから日本語の1つ目のリズムを推測
            tx = Variable(np.array([params['ja_rythm_list'][ja_rythm_phrase[0]]], dtype=np.int32))
            loss += F.softmax_cross_entropy(ja_rythm_model.predict(h), tx)
            # 足し合わせた出力から日本語のリズムを出力
            for index, rythm in enumerate(ja_rythm_phrase):
                y_ja_rythm = ja_rythm_model.forward(rythm, params['ja_rythm_list'])
                if rythm != '<eos>':
                    tx = Variable(np.array([params['ja_rythm_list'][ja_rythm_phrase[index+1]]], dtype=np.int32))
                    # print(y_ja, tx)
                    loss += F.softmax_cross_entropy(y_ja_rythm, tx)
            # print(ja_model.l1.upward.W.grad)
            en_model.cleargrads()
            en_rythm_model.cleargrads()
            ja_model.cleargrads()
            ja_rythm_model.cleargrads()

            loss.backward()
            loss.unchain_backward()
            ja_model.reset()
            ja_rythm_model.reset()
            for opt in optimizer:
                opt.update()

        # lossの可視化
        step.append(epoch+1)
        loss_list.append(loss.data)

        print(loss)
    # モデルとして保存
    serializers.save_hdf5('models/en_model_' + str(params['epoch_num']), en_model)
    serializers.save_hdf5('models/en_rythm_model_' + str(params['epoch_num']), en_rythm_model)
    serializers.save_hdf5('models/ja_model_' + str(params['epoch_num']), ja_model)
    serializers.save_hdf5('models/ja_rythm_model_' + str(params['epoch_num']), ja_rythm_model)

    # 学習過程のlossグラフ
    plt.plot(step, loss_list)
    plt.title("Training Data")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.grid(True)
    plt.show()

def predict(params, filename):
    en_model = EnglishLSTM(len(params['en_list']))
    en_rythm_model = EnglishRythmLSTM(len(params['en_rythm_list']))
    ja_model = JapaneseLSTM(len(params['ja_list']))
    ja_rythm_model = JapaneseRythmLSTM(len(params['ja_rythm_list']))

    serializers.load_hdf5('models/en_model_' + str(params['epoch_num']), en_model)
    serializers.load_hdf5('models/en_rythm_model_' + str(params['epoch_num']), en_rythm_model)
    serializers.load_hdf5('models/ja_model_' + str(params['epoch_num']), ja_model)
    serializers.load_hdf5('models/ja_rythm_model_' + str(params['epoch_num']), ja_rythm_model)
    x1 = [
        'are',
        'you',
        'going',
        'to',
        'scarborough',
        'fair',
        '?',
        '<eos>'
    ]
    x2 = [
        '48',
        '24',
        '24',
        '24',
        '24',
        '36',
        '12',
        '24',
        '72',
        '<eos>',
    ]
    arr = [k for k in params['ja_list']]
    arr2 = [k for k in params['ja_rythm_list']]
    ja_y = ""
    ja_rythm_y = ""
    while((ja_y != '<eos>') and (ja_rythm_y != '<eos>')):
        for x in x1:
            y1 = en_model.forward(x, params['en_list'])
        for x in x2:
            y2 = en_rythm_model.forward(x, params['en_rythm_list'])

        h = y1 + y2

        # hから1つ目の単語を推測
        y3 = ja_model.predict(h)

        prob = F.softmax(y3.data).data
        prob = prob.argmax(axis=1)
        prob = int(prob)
        ja_y = arr[prob]
        # ja_y = str(np.random.choice(arr, p = prob[0]))
        print(ja_y)

        while(ja_y != '<eos>'):
            y3 = ja_model.forward(ja_y, params['ja_list'])
            prob = F.softmax(y3.data).data
            prob = prob.argmax(axis=1)
            prob = int(prob)
            ja_y = arr[prob]
            # ja_y = str(np.random.choice(arr, p = prob[0]))
            print(ja_y)


        # hから1つ目のリズムを推測
        y4 = ja_rythm_model.predict(h)

        prob = F.softmax(y4.data).data
        ja_rythm_y = str(np.random.choice(arr2, p = prob[0]))
        print(ja_rythm_y)

        while(ja_rythm_y != '<eos>'):
            y4 = ja_rythm_model.forward(ja_rythm_y, params['ja_rythm_list'])
            prob = F.softmax(y4.data).data
            ja_rythm_y = str(np.random.choice(arr2, p = prob[0]))
            print(ja_rythm_y)

def get_data_arr(filename):
    file = open(filename)
    line = file.read()
    line = line.strip()
    file.close()
    return line.split("\n")

if __name__=="__main__":
    data_manager = DataManager()
    epoch_num = 350
    # read data
    en_data = get_data_arr("./data/english.txt")
    en_rythm = get_data_arr("./data/en_rythm.txt")
    ja_data = get_data_arr("./data/japanese.txt")
    ja_rythm = get_data_arr("./data/ja_rythm.txt")

    params = {
        'epoch_num': epoch_num,
        'english': data_manager.get_splite_list(en_data[:]),
        'en_rythm': data_manager.get_splite_list(en_rythm[:]),
        'japanese': data_manager.get_splite_list(ja_data[:]),
        'ja_rythm': data_manager.get_splite_list(ja_rythm[:]),
        'batch_size': 5,
        'en_list': data_manager.get_word_list(en_data[:]),
        'en_rythm_list': data_manager.get_word_list(en_rythm[:]),
        'ja_list': data_manager.get_word_list(ja_data[:]),
        'ja_rythm_list': data_manager.get_word_list(ja_rythm[:]),
    }

    # train(params)
    predict(params, 'models/model_3.npz')
