import numpy as np
import os
import re
import sys
import MeCab
import xml.etree.ElementTree as ET


def get_re_delimiter(delimiter):
    re_delimiter = ".*("
    for num in range(len(delimiter)-1):
        re_delimiter += "\\" + delimiter[num] + "|"
    re_delimiter += "\\" + delimiter[len(delimiter)-1] + ")"

    return re_delimiter

# フレーズから、歌詞とモーラ数を抽出する
def read_musicxml(filename, delimiter, is_en=True):
    # 正規表現の作成
    re_delimiter = get_re_delimiter(delimiter)
    mecab = MeCab.Tagger("-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")

    file = open(filename, 'rb')
    elem = ET.fromstring(file.read())
    divisions = elem.find('.//divisions').text
    # multiply = 24 / int(divisions)

    tune = []
    phrase = {'Lyrics': '', 'Mora': ''}
    mora = 0
    if not is_en:
        ja_ly_mora_dict = {'Lyric':[], 'Mora':[]}
        ja_current_num = 0
    # tie_durations = 0
    for note in elem.findall('.//note'):
        # 歌詞がない音符は休符の時のみアスタリスク
        # lyric_text = "* " if note.find('rest') is not None else ""
        lyric_text = ''
        mora_text = ''
        duration = note.find('duration').text
        mora = 0 if note.find('rest') is not None else mora + 1
        lyric = note.find('lyric')
        notations = note.find('notations')
        syllabic = ""
        if lyric is not None:
            lyric_text = lyric.find('text').text.lower()
            syllabic = lyric.find('syllabic').text

        # tie: 途中で歌詞が変わることは想定していない
        if notations is not None:
            type = ""
            if notations.find('tied') is not None:
                type = notations.find('tied').get('type')
            if type == 'start':
                tie_lyrics = lyric_text
                # tie_durations += int(duration)
                if ((syllabic == 'single') or (syllabic == 'end')) and re.match(r'[a-zA-z]', lyric_text):
                    tie_lyrics += ' '
                continue
            elif type == 'stop':
                # duration = tie_durations + int(duration)
                lyric_text = tie_lyrics
                # tie_durations = 0
                tie_lyrics = ""
                mora = mora - 1

        # judge phrase
        if re.match(re_delimiter, lyric_text):  # end of phrase
            for delm in delimiter:  # 句読点抜き
                lyric_text = lyric_text.replace(delm, '')
            if is_en:
                # phrase['Mora'] += str(int(duration) * int(multiply))
                phrase['Mora'] += str(mora)
                phrase['Lyrics'] += lyric_text
                # print(phrase)
                tune.append(phrase)
            else:
                phrase['Lyrics'] += lyric_text
                # それぞれの形態素が何個モーラを持っているか ja_ly_mora_dict を突き合わせて調べる
                print(mecab.parse(phrase['Lyrics']))
                # print(phrase['Lyrics'])

            phrase = {'Lyrics': "", 'Mora': ""}
            mora = 0
        else:
            # 歌詞より前に休符がある場合は無視
            # if len(phrase['lyrics']) == 0 and "*" in lyric_text:
                # continue
            # phrase['Mora'] += str(int(duration) * int(multiply)) + ' '
            if is_en:
                phrase['Lyrics'] += lyric_text
                if ((syllabic == 'single') or (syllabic == 'end')) and re.match(r'[a-zA-z]|\'', lyric_text):
                    phrase['Lyrics'] += ' '
                    phrase['Mora'] += str(mora) + ' '
                    mora = 0
            # 日本語の場合、各単語へのモーラを保存しておかなければならない
            else:
                if lyric_text != '':
                    phrase['Lyrics'] += lyric_text
                    ja_ly_mora_dict['Lyric'].append(lyric_text)
                    ja_ly_mora_dict['Mora'].append(mora)
                    ja_current_num += 1
                else:
                    if mora > 0:
                        ja_ly_mora_dict['Mora'][ja_current_num-1] += mora
                mora = 0
    return tune

# フレーズから、歌詞と音符の長さを抽出する
def extract_lyrics_duration(filename, delimiter):
    # 正規表現の作成
    re_delimiter = get_re_delimiter(delimiter)

    file = open(filename, 'rb')
    elem = ET.fromstring(file.read())
    divisions = elem.find('.//divisions').text
    multiply = 24 / int(divisions)

    tune = []
    phrase = {'lyrics': "", 'durations': ""}
    tie_durations = 0
    for note in elem.findall('.//note'):
        # 歌詞がない音符は休符の時のみアスタリスク
        lyric_text = "* " if note.find('rest') is not None else ""
        duration = note.find('duration').text
        lyric = note.find('lyric')
        notations = note.find('notations')
        syllabic = ""
        if lyric is not None:
            lyric_text = lyric.find('text').text.lower()
            syllabic = lyric.find('syllabic').text

        # tie: 途中で歌詞が変わることは想定していない
        if notations is not None:
            type = ""
            if notations.find('tied') is not None:
                type = notations.find('tied').get('type')
            if type == 'start':
                tie_lyrics = lyric_text
                tie_durations += int(duration)
                if ((syllabic == 'single') or (syllabic == 'end')) and re.match(r'[a-zA-z]', lyric_text):
                    tie_lyrics += ' '
                continue
            elif type == 'stop':
                duration = tie_durations + int(duration)
                lyric_text = tie_lyrics
                tie_durations = 0
                tie_lyrics = ""

        # judge phrase
        if re.match(re_delimiter, lyric_text):  # end of phrase
            for delm in delimiter:  # 句読点抜き
                lyric_text = lyric_text.replace(delm, '')
                # lyric_text = lyric_text.replace(delm, ' '+delm)
            phrase['durations'] += str(int(duration) * int(multiply))
            phrase['lyrics'] += lyric_text
            # print(phrase)
            tune.append(phrase)
            phrase = {'lyrics': "", 'durations': ""}
        else:
            # 歌詞より前に休符がある場合は無視
            if len(phrase['lyrics']) == 0 and "*" in lyric_text:
                continue
            phrase['durations'] += str(int(duration) * int(multiply)) + ' '
            phrase['lyrics'] += lyric_text
            if ((syllabic == 'single') or (syllabic == 'end')) and re.match(r'[a-zA-z]|\'', lyric_text):
                phrase['lyrics'] += ' '
    return tune

if __name__=="__main__":
    file_num = 5  # number of reading file
    en_path = './data/score/en'
    ja_path = './data/score/ja'
    en_delimiter = [',', '.' , '?', '!', ':']
    ja_delimiter = ['、', '。', '？', '！']

    en_file_list = os.listdir(en_path)
    ja_file_list = os.listdir(ja_path)

    files = [
        open('./data/english.txt', 'w'),
        open('./data/en_rythm.txt', 'w'),
        open('./data/japanese.txt', 'w'),
        open('./data/ja_rythm.txt', 'w')
    ]

    # en_file_list = ['05.musicxml']
    # ja_file_list = ['05.musicxml']

    for num in range(file_num):
        print("tune: %d" % (num+1))
        try:
            en_data = read_musicxml(en_path +'/'+ en_file_list[num], en_delimiter)
            ja_data = read_musicxml(ja_path +'/'+ ja_file_list[num], ja_delimiter, False)
        except:
            print('Error: ' + en_file_list[num])
            import traceback
            traceback.print_exc()

        for en, ja in zip(en_data, ja_data):
            files[0].write(en['Lyrics']+'\n')
            files[1].write(en['Mora']+'\n')
            files[2].write(ja['Lyrics']+'\n')
            files[3].write(ja['Mora']+'\n')
        for file in files:
            file.write('\n')
        # durationのとき
        # for en, ja in zip(en_data, ja_data):
        #     files[0].write(en['lyrics']+'\n')
        #     files[1].write(en['durations']+'\n')
        #     files[2].write(ja['lyrics']+'\n')
        #     files[3].write(ja['durations']+'\n')
        # for file in files:
        #     file.write('\n')
    for file in files:
        file.close()
