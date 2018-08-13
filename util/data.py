import math

class DataManager():
    # 重複しない単語リストを作成
    def get_word_list(self, data):
        result = {}
        for i in range(len(data)):
            lt = data[i].split()
            data[i] = lt
            for w in lt:
                if w not in result:
                    result[w] = len(result)
        result['<eos>'] = len(result)
        return result

    def get_splite_list(self, data):
        result = []
        for i in range(len(data)):
            lt = data[i].split()
            if(len(lt) > 0):
                lt.append('<eos>')
            result.append(lt)
        return result

    # 学習できるデータ形式に変更
    def get_train_batch(self, data, batch_size):
        result = []
        for index in range(0, math.ceil(len(data) / batch_size)):
            tune = []
            batch = data[index*batch_size:min([index*batch_size+batch_size, len(data)])]
            max_length = max([len(st) for st in batch])
            tune_flg = False
            for j in range(max_length):
                phrase = []
                for s in batch:
                    # 曲の区切りには空の配列を挿入
                    if len(s) == 0:
                        tune_flg = True
                        continue
                    if j < len(s):
                        phrase.append(s[j])
                    else:
                        phrase.append(None)
                tune.append(phrase)
                # tune.append([s[j] if j < len(s) else None for s in batch])
            if tune_flg:
                result.append([])
            result.append(tune)
        return result
