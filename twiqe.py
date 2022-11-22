import argparse
from tqdm import trange
import logging
import os
from datetime import datetime

import torch

# randomness control
import random
import numpy as np

# twiqe model import
import xgboost
import joblib
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

from utils.util import sbert_cossim, huggingface_cossim, sbert_model_selector, huggingface_model_selector
from utils.grid_search import SearchBase


class TwiQE:
    def __init__(self,
                 source_language_code,
                 target_language_code,
                 score_model_path=None,
                 randomness_control=True,
                 logging_level='warning'):
        if randomness_control:
            self._randomness_control()
        set_logging(logging_level)

        self.source_language_code = source_language_code
        self.target_language_code = target_language_code
        self.score_model_path = score_model_path
        if score_model_path:
            assert os.path.isfile(score_model_path), "Check the path!"
        logging.warning("Loading PLMs...")
        self.multilingual_model = sbert_model_selector('mpnet')
        if self.source_language_code == 'ko':
            self.monolingual_model = huggingface_model_selector("kobert_multi")
        elif self.source_language_code == 'en':
            self.monolingual_model = huggingface_model_selector("enroberta")
        else:
            assert self.source_language_code in ['ko', 'en'], 'Not supported language!'

        self.score_model = joblib.load(self.score_model_path) if self.score_model_path else None
        logging.warning("...DONE!")

    def __call__(self, sentences, df_col_match={}):
        score = self.predict(sentences, df_col_match=df_col_match)
        return score

    def __str__(self):
        return f"TwiQE [{self.source_language_code}] -> [{self.target_language_code}]"

    def _randomness_control(self):
        random_seed = 42
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        return

    def cossim_from_df(self, df, batch_size=32):
        src_tgt_cossim, src_rtt_cossim = np.array([]), np.array([])
        for i in trange(len(df) // batch_size + 1):
            minibatch = df[i * batch_size: (i + 1) * batch_size]
            if minibatch.empty: continue
            src = minibatch['source_sentence'].values.tolist()
            tgt = minibatch['target_sentence'].values.tolist()
            rtt = minibatch['rtt_sentence'].values.tolist()

            part_of_src_tgt_cossim = sbert_cossim(self.multilingual_model, src, tgt)
            part_of_src_rtt_cossim = huggingface_cossim(self.monolingual_model, src, rtt)

            src_tgt_cossim = np.append(src_tgt_cossim, part_of_src_tgt_cossim)
            src_rtt_cossim = np.append(src_rtt_cossim, part_of_src_rtt_cossim)

        df['src_tgt_cossim'] = src_tgt_cossim
        df['src_rtt_cossim'] = src_rtt_cossim

        if 'src_n_tokens' not in df:
            df['src_n_tokens'] = df['source_sentence'].map(lambda x: len(x.split()))
        if 'rtt_n_tokens' not in df:
            df['rtt_n_tokens'] = df['rtt_sentence'].map(lambda x: len(x.split()))
        if 'diff' not in df:
            df['diff'] = abs(df['src_n_tokens'] - df['rtt_n_tokens'])

        return df

    def _extract_features(self, sentences, batch_size):
        self.src_tgt_cossim, self.src_rtt_cossim = np.array([]), np.array([])
        self.src_n_tokens, self.rtt_n_tokens, self.diff = np.array([]), np.array([]), np.array([])

        for i in trange(len(sentences)//batch_size+1):
            minibatch = sentences[i*batch_size: (i+1)*batch_size]
            if minibatch == []: continue

            src, tgt, rtt = zip(*(minibatch))
            src, tgt, rtt = list(src), list(tgt), list(rtt)


            part_of_src_tgt_cossim = sbert_cossim(self.multilingual_model, src, tgt)
            part_of_src_rtt_cossim = huggingface_cossim(self.monolingual_model, src, rtt)
            part_of_src_n_tokens = np.array([len(i.split(' ')) for i in src])
            part_of_rtt_n_tokens = np.array([len(i.split(' ')) for i in rtt])
            part_of_diff = abs(part_of_src_n_tokens - part_of_rtt_n_tokens)

            self.src_tgt_cossim = np.append(self.src_tgt_cossim, part_of_src_tgt_cossim)
            self.src_rtt_cossim = np.append(self.src_rtt_cossim, part_of_src_rtt_cossim)
            self.src_n_tokens = np.append(self.src_n_tokens, part_of_src_n_tokens)
            self.rtt_n_tokens = np.append(self.rtt_n_tokens, part_of_rtt_n_tokens)
            self.diff = np.append(self.diff, part_of_diff)



    def _extract_features_from_already_done(self, df, df_col_match, batch_size=32):
        """
        df_col_match={"src_tgt_cossim": "src_tgt_cossim",
                      "src_rtt_cossim": "src_rtt_cossim",
                      "src_n_tokens": "src_n_tokens",
                      "rtt_n_tokens": "rtt_n_tokens",
                      "diff": "diff"}
        :param df:
        :param df_col_match:
        :return:
        """
        for col_name in df_col_match.values():
            assert col_name in df.keys(), 'df 컬럼 이름과 실제 이름을 매치해주세요!'

        if 'src_tgt_cossim' not in df_col_match or 'src_rtt_cossim' not in df_col_match:
            df = self.cossim_from_df(df, batch_size)

        self.src_tgt_cossim = df['src_tgt_cossim']
        self.src_rtt_cossim = df['src_rtt_cossim']
        self.src_n_tokens = df['src_n_tokens']
        self.rtt_n_tokens = df['rtt_n_tokens']
        self.diff = df['diff']

        return df

    def train(self, sentences, labels, batch_size=16, split_ratio=None, save_dir='./score_model', do_grid_search=False):
        '''score model 훈련
        batch_size 만큼 나눠서 rtt 생성, 코사인 유사도 계산, 문장 길이와 차이 계산
        준비된 데이터로 score model 훈련
        '''

        self._extract_features(sentences, batch_size=batch_size)
        features = np.array([self.src_tgt_cossim, self.src_rtt_cossim,
                             self.src_n_tokens, self.rtt_n_tokens, self.diff]).transpose()
        targets = np.array(labels)

        if do_grid_search:
            split_ratio = 0.1

        if split_ratio:
            logging.warning('split dataset')
            features, X_test, targets, y_test = train_test_split(features, targets, stratify=targets,
                                                                 train_size=split_ratio, random_state=42)

        if not self.score_model:
            self.score_model = xgboost.XGBRegressor(random_state=42)

        if do_grid_search:
            grid_search = SearchBase(features, targets, X_test, y_test)
            grid_search('xgb')
            self.score_model = grid_search.best_estimator


        self.score_model.fit(features, targets)

        if split_ratio:
            model_y_hat = self.score_model.predict(X_test)
            model_r2 = r2_score(y_test, model_y_hat)
            scores = cross_val_score(self.score_model, features, targets, scoring="neg_mean_squared_error", cv=10)
            score = -1 * np.mean(scores)
            logging.warning(f"R2 Score: {model_r2 * 100:.5f}")
            logging.warning(f"MSE: {score:.5f}")

        if save_dir:
            if self.score_model_path is None:
                now = datetime.now().strftime('%y%m%d_%H%M')
                self.score_model_path = pathlib.Path(
                    save_dir) / f'twiqe_{self.source_language_code}2{self.target_language_code}_{now}.pkl'
            joblib.dump(self.score_model, self.score_model_path)

    def predict(self, sentences, batch_size=16, df_col_match={"src_tgt_cossim": "src_tgt_cossim",
                                                              "src_rtt_cossim": "src_rtt_cossim",
                                                              "src_n_tokens": "src_n_tokens",
                                                              "rtt_n_tokens": "rtt_n_tokens",
                                                              "diff": "diff"}):
        assert self.score_model_path is not None, "Train TwiQE first or Put `score_model_path` into the model instance!"
        self._extract_features_from_already_done(sentences, df_col_match)

        features_for_predict = np.array([self.src_tgt_cossim, self.src_rtt_cossim,
                                         self.src_n_tokens, self.rtt_n_tokens, self.diff]).transpose()

        score = self.score_model.predict(features_for_predict).tolist()
        return score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--logging_level', default='warning')
    args = parser.parse_args()
    return args


def set_logging(logging_level):
    pid = os.getpid()
    log_format = f"[{pid}] " + '%(levelname)s - %(asctime)s :: %(message)s'
    LOG_LEVEL = {
        'critical': logging.CRITICAL,
        'error': logging.ERROR,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG,
    }

    logging_level = LOG_LEVEL[logging_level]

    logging_style = [logging.StreamHandler()]
    logging.basicConfig(
        handlers=logging_style,
        format=log_format,
        level=logging_level
    )
    return logging

