import chainer.links as L
import chainer.functions as F
from chainer import Chain, Variable
import numpy as np

from util.data import DataManager

class EnglishRythmLSTM(Chain):
    def __init__(self, vocab_size):
        super(EnglishRythmLSTM, self).__init__(
            embed = L.EmbedID(vocab_size, 8),
            l1 = L.LSTM(None, 16),
            l2 = L.Linear(None, 32),
            l3 = L.Linear(None, 16)
        )
        with self.init_scope():
            self.data_manager = DataManager()

    def get_train_data(self, data, num):
        return self.data_manager.get_train_batch(data, num)

    def reset(self):
        self.l1.reset_state()

    def forward(self, rythm, en_list):
        rid = en_list[rythm]
        emb_y = self.embed(Variable(np.array([rid], dtype=np.int32)))
        return self.predict(emb_y)

    def forward_parallel(self, tune, en_list):
        # フレーズ、単語数の0ベクトルの作成
        x = np.zeros((len(tune), len(en_list)), np.float32)
        for i, word in enumerate(tune):
            if word is not None: x[i, en_list[word]] = 1
        x = Variable(x)
        y = self.predict(x)
        return y

    def predict(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
