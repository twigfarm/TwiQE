import pandas as pd
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from konlpy.tag import Kkma
import nltk
import spacy
from koalanlp.Util import initialize, finalize
from koalanlp import API
from koalanlp.proc import Parser
import time


def make_sentLen(raw_sentence, save_dir):
    # 0: 1~4 어절, 1: 5~8 어절, 2: 9~12어절, 3: 13~16어절, 4: 17~20어절, 5: 21~25어절, 6: 26~28어절
    len_class = [[0, 1, 4], [1, 5, 8], [2, 9, 12], [3, 13, 16], [4, 17, 20], [5, 21, 25], [6, 26, 28]]
    sent, label = [], []

    for i in tqdm(range(len(raw_sentence))):
        for st in len_class:
            if len(raw_sentence[i].split()) >= st[1] and len(raw_sentence[i].split()) <= st[2]:
                label.append(st[0])
                sent.append(raw_sentence[i])

    df = pd.DataFrame({'sentence': sent, 'label': label})

    df.to_csv(save_dir + f'/sentLen_{len(df)}_{time.strftime("%Y%m%d-%H%M%S")}.tsv', encoding='utf-8', index=False,
              sep='\t')


def make_wc(raw_sentence, save_dir):
    words = []

    # 어절 빈도수 확인을 위해 특수문자들 제거
    table = str.maketrans(',.\'"><?!“”‘’', '            ')
    pp_sentence = list(map(lambda x: x.translate(table), raw_sentence))

    # 어절 단위로 잘라서 저장, 영어는 전부 소문자로 처리
    for i in range(len(pp_sentence)):
        words.extend(pp_sentence[i].lower().split())

    counter = Counter(words)
    counter = list(counter.items())

    # 빈도수에 따라 나열
    counter = sorted(counter, key=lambda x: x[1], reverse=True)

    # 어휘를 빈도 수에 따라 나열했을 때, 몇 위부터 몇 위까지를 중빈도 어휘로 뽑을 것인지
    words = sorted(list(map(lambda x: x[0], counter[6000:7000])))

    sent, label = [], []

    # 각 문장에서 라벨로 선정한 어휘가 유일하게 하나 포함되는 문장 선별
    for i in tqdm(range(len(pp_sentence))):
        cnt = {}
        for w in words:
            if w in pp_sentence[i].lower().split():
                cnt[w] = pp_sentence[i].lower().split().count(w)
        if len(cnt.items()) == 1 and list(cnt.items())[0][1] == 1:
            sent.append(raw_sentence[i])
            label.append(words.index(list(cnt.items())[0][0]))

    df = pd.DataFrame({'sentence': sent, 'label': label})

    df.to_csv(save_dir + f'/wc_{len(df)}_{time.strftime("%Y%m%d-%H%M%S")}.tsv', encoding='utf-8', index=False, sep='\t')


def make_tense_and_negation_ko(raw_sentence, save_dir):
    ko_tags_past = ('었', 'EPT'), ('았', 'EPT'), ('였', 'EPT')
    ko_tags_not = ('안', 'MAG'), ('못', 'MAG'), ('않', 'VXV'), ('말', 'VXV'), ('못하', 'VX'), ('아니', 'VV'), ('없', 'VA'), (
    '모르', 'VV'), ('안되', 'VV')

    kkma = Kkma()

    sentence_past, sentence_not, label_past, label_not = [], [], [], []

    for i in tqdm(range(len(raw_sentence))):
        ko_past = False
        try:
            ko_sent = kkma.pos(raw_sentence[i])
        except:
            raw_sentence[i] = '<문장에 처리할 수 없는 문자가 있습니다.>'
            ko_sent = kkma.pos(raw_sentence[i])
        ko_independent_clause = len(ko_sent) // 2 + 1

        ko_not = False
        ko_not_tag = 0

        # 뒤에서부터 확인
        for j in range(len(ko_sent) - 1, -1, -1):
            if ko_sent[j][1].startswith('V'):
                ko_independent_clause = max(ko_independent_clause, j)
                break

        for tag in ko_sent[ko_independent_clause:]:
            if tag in ko_tags_past:
                ko_past = True
            if tag in ko_tags_not:
                ko_not_tag += 1

        if ko_not_tag == 1:
            ko_not = True

        sentence_past.append(raw_sentence[i])
        if ko_past:
            label_past.append(1)
        else:
            label_past.append(0)

        sentence_not.append(raw_sentence[i])
        if ko_not:
            label_not.append(0)
        else:
            label_not.append(1)

    df_past = pd.DataFrame({'sentence': sentence_past, 'label': label_past})
    df_not = pd.DataFrame({'sentence': sentence_not, 'label': label_not})

    df_past.to_csv(save_dir + f'/tense_ko_{len(df_past)}_{time.strftime("%Y%m%d-%H%M%S")}.tsv', encoding='utf-8',
                   index=False, sep='\t')
    df_not.to_csv(save_dir + f'/negation_ko_{len(df_not)}_{time.strftime("%Y%m%d-%H%M%S")}.tsv', encoding='utf-8',
                  index=False, sep='\t')


def make_tense_and_negation_en(raw_sentence, save_dir):
    en_tags_past = ['VBD']  # , 'VBN'
    en_tags_not = ("n't", 'ADV'), ('not', 'ADV'), ('Nothing', 'VERB')

    nlp = spacy.load("en_core_web_sm")

    sentence_past, sentence_not, label_past, label_not = [], [], [], []

    for i in tqdm(range(len(raw_sentence))):
        en_past = False

        en_sent = nlp(raw_sentence[i])
        en_independent_clause = 0

        en_not = False
        en_not_tag = 0

        for token in en_sent:
            if token.dep_ == 'ROOT':
                root = token.head.text
                if token.head.tag_ == 'VBD':
                    en_past = True
                elif token.head.tag_ in ['VB', 'VBN', 'VBG']:
                    for token_ in en_sent:
                        if token_.dep_.startswith('aux') and token_.head.text == root:
                            if token_.tag_ == 'VBD' or token_.text in ['\'ve', 'have', 'has', 'had']:
                                en_past = True

        for token in en_sent:
            if token.dep_ == 'neg' and token.head.text == root:
                en_not = True

        sentence_past.append(raw_sentence[i])
        if en_past:
            label_past.append(1)
        else:
            label_past.append(0)

        sentence_not.append(raw_sentence[i])
        if en_not:
            label_not.append(0)
        else:
            label_not.append(1)

    df_past = pd.DataFrame({'sentence': sentence_past, 'label': label_past})
    df_not = pd.DataFrame({'sentence': sentence_not, 'label': label_not})

    df_past.to_csv(save_dir + f'/tense_en_{len(df_past)}_{time.strftime("%Y%m%d-%H%M%S")}.tsv', encoding='utf-8',
                   index=False, sep='\t')
    df_not.to_csv(save_dir + f'/negation_en_{len(df_not)}_{time.strftime("%Y%m%d-%H%M%S")}.tsv', encoding='utf-8',
                  index=False, sep='\t')


def make_treeDepth_and_topConst_ko(raw_sentence, save_dir):
    result = []

    def dfs(graph, r, depth):
        for n in graph[r]:
            if graph[n] != []:
                dfs(graph, n, depth + 1)
            else:
                result.append(depth + 1)

    initialize(java_options="-Xmx4g", KKMA="2.0.4", ETRI="2.0.4")
    parser = Parser(API.KKMA)

    sentence_depth, sentence_const, label_depth, label_const = [], [], [], []

    kkma = Kkma()

    for j in tqdm(range(0, len(raw_sentence), 1)):
        # print(ko[j])
        try:
            ko_parsed = parser(raw_sentence[j:min(j + 1, len(raw_sentence))])
        except:
            raw_sentence[j] = '의존 구조 분석이 불가능한 문장입니다.'
            ko_parsed = parser(raw_sentence[j:min(j + 1, len(raw_sentence))])

        if len(ko_parsed) > 1:
            raw_sentence[j] = '의존 구조 분석이 불가능한 문장입니다.'
            ko_parsed = parser(raw_sentence[j:min(j + 1, len(raw_sentence))])

        if '+' in raw_sentence[j]:
            raw_sentence[j] = '의존 구조 분석이 불가능한 문장입니다.'
            ko_parsed = parser(raw_sentence[j:min(j + 1, len(raw_sentence))])

        for i in range(len(ko_parsed)):
            graph = [[] for _ in range(len(ko_parsed[i]))]
            root = []
            top_const = []

            for dep in ko_parsed[i].getDependencies():
                if dep.src == None:
                    root.append(str(ko_parsed[i]).split().index(str(dep.dest).split()[0]))
                    continue

            if len(root) != 1:
                sentence_depth.append(raw_sentence[j])
                sentence_const.append(raw_sentence[j])
                label_depth.append('-')
                label_const.append('-')
                continue

            for dep in ko_parsed[i].getDependencies():
                if dep.src == None: continue

                src_idx = str(ko_parsed[i]).split().index(str(dep.src).split()[0])
                dest_idx = str(ko_parsed[i]).split().index(str(dep.dest).split()[0])

                while dep not in ko_parsed[i][src_idx].getDependentEdges() or dep != ko_parsed[i][
                    dest_idx].getGovernorEdge():
                    if dep not in ko_parsed[i][src_idx].getDependentEdges():
                        src_idx += str(ko_parsed[i]).split()[src_idx + 1:].index(str(dep.src).split()[0]) + 1
                    if dep != ko_parsed[i][dest_idx].getGovernorEdge():
                        dest_idx += str(ko_parsed[i]).split()[dest_idx + 1:].index(str(dep.dest).split()[0]) + 1

                graph[src_idx].append(dest_idx)

            for dep in ko_parsed[i].getDependencies():
                if str(dep.src).split()[0] == str(ko_parsed[i]).split()[root[0]]:
                    top_const.append(str(dep.dest).split()[2].split('+')[0].split('/')[1])

            top_const = sorted(list(set(top_const)))

            sentence_const.append(raw_sentence[j])
            label_const.append('_'.join(top_const))
            if label_const[-1] == '':
                label_const[-1] = '-'

            sentence_depth.append(raw_sentence[j])

            result = []
            try:
                dfs(graph, root[0], 0)
            except:
                label_depth.append('-')
                continue

            try:
                label_depth.append(max(result))
            except:
                label_depth.append('-')

    finalize()

    df_depth = pd.DataFrame({'sentence': sentence_depth, 'label': label_depth})
    df_const = pd.DataFrame({'sentence': sentence_const, 'label': label_const})

    idx = df_depth[df_depth['label'] == '-'].index
    df_depth.drop(idx, inplace=True)
    idx = df_depth[df_depth['sentence'] == '의존 구조 분석이 불가능한 문장입니다.'].index
    df_depth.drop(idx, inplace=True)

    idx = df_const[df_const['label'] == '-'].index
    df_const.drop(idx, inplace=True)
    idx = df_const[df_const['sentence'] == '의존 구조 분석이 불가능한 문장입니다.'].index
    df_const.drop(idx, inplace=True)

    df_depth.to_csv(save_dir + f'/treeDepth_ko_{len(df_depth)}_{time.strftime("%Y%m%d-%H%M%S")}.tsv', encoding='utf-8',
                    index=False, sep='\t')
    df_const.to_csv(save_dir + f'/topConst_ko_{len(df_depth)}_{time.strftime("%Y%m%d-%H%M%S")}.tsv', encoding='utf-8',
                    index=False, sep='\t')


def make_treeDepth_and_topConst_en(raw_sentence, save_dir):
    result = []

    def dfs(graph, r, depth):
        for n in graph[r]:
            if graph[n] != []:
                dfs(graph, n, depth + 1)
            else:
                result.append(depth + 1)

    nlp = spacy.load("en_core_web_sm")

    sentence_depth, sentence_const, label_depth, label_const = [], [], [], []

    for i in tqdm(range(len(raw_sentence))):
        top_const = []
        doc = nlp(raw_sentence[i])
        graph = [[] for _ in range(len(doc))]

        for token in doc:
            graph[token.head.i].append(token.i)
            if token.dep_ == 'ROOT':
                root = token.i

        for token in doc:
            if token.head.i == root and token.i != root and token.tag_ not in [',', '.']:
                top_const.append(token.tag_)

        top_const = sorted(list(set(top_const)))

        sentence_const.append(raw_sentence[i])
        label_const.append('_'.join(top_const))
        if label_const[-1] == '':
            label_const[-1] = '-'

        sentence_depth.append(raw_sentence[i])

        graph[root].remove(root)
        result = []
        try:
            dfs(graph, root, 0)
        except:
            label_depth.append('-')
            continue

        try:
            label_depth.append(max(result))
        except:
            label_depth.append('-')

    finalize()

    df_depth = pd.DataFrame({'sentence': sentence_depth, 'label': label_depth})
    df_const = pd.DataFrame({'sentence': sentence_const, 'label': label_const})

    df_depth.to_csv(save_dir + f'/treeDepth_en_{len(df_depth)}_{time.strftime("%Y%m%d-%H%M%S")}.tsv', encoding='utf-8',
                    index=False, sep='\t')
    df_const.to_csv(save_dir + f'/topConst_en_{len(df_depth)}_{time.strftime("%Y%m%d-%H%M%S")}.tsv', encoding='utf-8',
                    index=False, sep='\t')


def postprocess_wc(data, save_dir):
    '''
    이상치 제거
    라벨별 일정 개수
    라벨 0부터 1씩 증가
    '''
    label_series = data['label'].value_counts()
    label_dict = label_series.to_dict().items()

    iqr = label_series.quantile(0.75) - label_series.quantile(0.25)
    upper_outlier = label_series.quantile(0.75) + (iqr * 1.5)
    lower_outlier = label_series.quantile(0.25) - (iqr * 1.5)

    label_dict = list(filter(lambda x: x[1] <= upper_outlier and x[1] >= lower_outlier, label_dict))

    label_num = input(f'이상치를 제거한 결과, 라벨 중 최다 빈도 횟수는 {label_dict[0][1]} 회 이고, 최소 빈도 횟수는 {label_dict[-1][1]} 입니다.\n\
                    현재 총 라벨 개수는 {len(label_dict)} 개 입니다. 라벨별 개수를 몇 개로 맞추시겠습니까? (정수 입력): ')

    label_dict = list(filter(lambda x: x[1] >= int(label_num), label_dict))
    key_list = list(map(lambda x: x[0], label_dict))

    for label in key_list:
        data_sample = data[data['label'] == label].sample(n=int(label_num))
        try:
            processed_data = pd.concat([processed_data, data_sample])
        except:
            processed_data = data_sample.copy()

    for src, tgt in zip(sorted(data['label'].unique().tolist()), list(range(len(label_dict)))):
        processed_data['label'] = processed_data['label'].apply(lambda x: str(x).replace(str(src), str(tgt)))

    processed_data = processed_data.sample(frac=1).reset_index(drop=True)

    processed_data.to_csv(save_dir + f'/wc_postprocessed_{len(processed_data)}_{time.strftime("%Y%m%d-%H%M%S")}.tsv',
                          encoding='utf-8', index=False, sep='\t')


def postprocess_treeDepth(data, save_dir, language_code):
    '''
    한, 영에 따라 범위 설정
    라벨 0부터 1씩 증가
    '''
    if language_code == 'ko':
        data = data[(data['label'] >= 4) & (data['label'] <= 9)]
        data['label'] = data['label'].apply(lambda x: x - 4)
    elif language_code == 'en':
        data = data[(data['label'] >= 5) & (data['label'] <= 12)]
        data['label'] = data['label'].apply(lambda x: x - 5)

    data.to_csv(save_dir + f'/treeDepth_postprocessed_{language_code}_{len(data)}_{time.strftime("%Y%m%d-%H%M%S")}.tsv',
                encoding='utf-8', index=False, sep='\t')


def postprocess_topConst(data, save_dir, language_code):
    '''
    상위 19개 + 기타 1개 라벨
    기타는 전체 5% 이내
    '''
    labels = list(map(lambda x: x[0], list(data['label'].value_counts().to_dict().items())[:20]))

    for i in range(len(data)):
        try:
            label = labels.index(data.loc[i]['label'])
            data.loc[i, 'label'] = label
        except:
            data.loc[i, 'label'] = 19

    if len(data[data['label'] == 19]) // len(data) > 0.05:
        others = data[data['label'] == 19].sample(n=len(data[data['label'] == 19]) // len(data))
        data = pd.concat([data[data['label'] != 19], others])

    data.to_csv(save_dir + f'/topConst_postprocessed_{language_code}_{len(data)}_{time.strftime("%Y%m%d-%H%M%S")}.tsv',
                encoding='utf-8', index=False, sep='\t')


def make_dataset(raw_sentence, task_name, language_code, save_dir):
    '''
    raw_sentence (list): 데이터로 만들 문장들로 이루어진 리스트
    task_name (str): 프로빙 태스크 이름 (sentLen, wc, tense, negation, treeDepth, topConst)
    language_code (str): 언어 코드 (ko, en)
    save_dir (str): 결과 데이터셋을 저장할 경로
    '''
    task = {'sentLen': [make_sentLen, make_sentLen], 'wc': [make_wc, make_wc],
            'tense': [make_tense_and_negation_ko, make_tense_and_negation_en],
            'negation': [make_tense_and_negation_ko, make_tense_and_negation_en],
            'treeDepth': [make_treeDepth_and_topConst_ko, make_treeDepth_and_topConst_en],
            'topConst': [make_treeDepth_and_topConst_ko, make_treeDepth_and_topConst_en]}

    lan = {'ko': 0, 'en': 1}

    task[task_name][lan[language_code]](raw_sentence, save_dir)
