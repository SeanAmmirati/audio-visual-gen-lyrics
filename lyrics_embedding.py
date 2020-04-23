import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import gensim.downloader as api
import nltk
import gensim
from wikipedia2vec import Wikipedia2Vec
import yaml
import numpy as np
import os
import pickle
import sys

sys.path.append('/home/seanammirati/dev/audio_visual_gen')


def tokenize_lyrics(filename):

    with open(filename, 'r') as f:
        ret = f.read()

    return word_tokenize(ret)


def process_tokens(tokens):
    cleaned = [w.lower() for w in tokens if w.isalpha()]
    additional_stpwords = ['yeah']
    stp_word = set(stopwords.words('english') + additional_stpwords)
    important = [w for w in cleaned if w not in stp_word]
    return important


def preprocess(filename):
    tokens = tokenize_lyrics(filename)
    return process_tokens(tokens)


def lyric_cone_weight(word_list, p=1):
    n_tokens = len(word_list)

    weights = [max([(n_tokens - i)/n_tokens, (i + 1)/n_tokens])
               for i in range(n_tokens)]

    weights = [w ** p for w in weights]

    return [[i, j] for i, j in zip(word_list, weights)]


def lyric_equal_weight(word_list):
    n_tokens = len(word_list)
    if n_tokens > 0:
        weights = [1/n_tokens] * n_tokens
    else:
        weights = []
    return [[i, j] for i, j in zip(word_list, weights)]


def lyric_nth_word_weight(word_list, n):
    n_tokens = len(word_list)

    weights = [0] * n_tokens
    if not weights:
        return []
    try:
        weights[n] = 1
    except IndexError:
        print('No index for this value, going to the next smallest')
        return lyric_nth_word_weight(word_list, n - 1)
    return [[i, j] for i, j in zip(word_list, weights)]


def lyric_last_word_weight(word_list):
    return lyric_nth_word_weight(word_list, -1)


def lyric_first_word_weight(word_list):
    return lyric_nth_word_weight(word_list, 0)


def lyric_weight(word_list, ty='last', p=1):

    if ty == 'eq':
        return lyric_equal_weight(word_list)
    elif ty == 'last':
        return lyric_last_word_weight(word_list)
    elif ty == 'first':
        return lyric_first_word_weight(word_list)
    elif ty == 'cone':
        return lyric_cone_weight(word_list, p)


def lyric_importance(filename, ty='last', p=1):
    with open(filename, 'r') as f:
        file = f.read()

    return lyric_importance_str(file, ty, p)


def lyric_importance_str(string, ty='last', p=1):

    lines = string.split('\n')
    words_with_weights = [w for l in lines if word_tokenize(
        l) for w in lyric_weight(process_tokens(word_tokenize(l)), ty, p)]

    return words_with_weights


def model_gen():
    model = Wikipedia2Vec.load(
        '/home/seanammirati/dev/audio_visual_gen/embeddings/enwiki_20180420_100d.pkl')

    return model


def create_vectors(tokens, m):
    ret_dict = {}
    for i, t in enumerate(tokens):
        try:
            ret_dict[t] = m.get_word_vector(t)
        except:
            print(f'Could not find token {t} in vocabulary.')
        print(f'{i} of {len(tokens)} complete.')
    return ret_dict


def vectorize_lyrics(songname):
    if (os.path.exists(f'{songname}/token_to_vec.yml')):
        return

    if not os.path.exists(songname):
        os.mkdir(songname)
    primadonna = f'/home/seanammirati/dev/audio_visual_gen/example_lyrics/{songname}.txt'
    important = preprocess(primadonna)
    m = model_gen()
    token_to_vec = create_vectors(important, m)

    for k in token_to_vec.keys():
        token_to_vec[k] = token_to_vec[k].tolist()
    with open(f'{songname}/token_to_vec.yml', 'w') as f:
        yaml.dump(token_to_vec, f)

    with open('/home/seanammirati/dev/audio_visual_gen/image_classes.yml', 'r') as f:
        img_classes = yaml.load(f)

    flipped = {}
    for k, v in img_classes.items():
        comma_sep = v.split(',')

        wordvec_sum = None
        i = 0
        for word in comma_sep:
            i += 1
            tkns = process_tokens(word_tokenize(word))
            for vector in list(create_vectors(tkns, m).values()):
                if wordvec_sum is None:
                    wordvec_sum = vector
                else:
                    wordvec_sum += vector
        mean_wordvec = (
            wordvec_sum / i).tolist() if wordvec_sum is not None else None

        if mean_wordvec:
            flipped[v] = {'id': k, 'vec': mean_wordvec}

    with open(f'{songname}/img_cls_to_vec.yml', 'w') as f:
        yaml.dump(flipped, f)

    return flipped


def process_lrc(path, line_jump=1):
    with open(path, 'r') as f:
        lrc = f.read()

    lines = lrc.split('\n')

    lrcs = []

    for i, l in enumerate(lines):
        try:
            # Quick and dirty way of dealing with non time stamp lines in lrc file
            int(l[1])
        except:
            continue
        splt = l.split(']')
        if len(splt) == 2:
            time = splt[0][1:]
            lyrics = splt[1]
            lrcs.append({'start': time, 'lyrics': lyrics})

    for i in range(len(lrcs)):
        try:
            lrcs[i]['end'] = lrcs[i+1]['start']
        except IndexError:
            pass
    df = pd.DataFrame(lrcs)
    df['start'] = pd.to_datetime(df['start'], format='%M:%S.%f')

    df['start'] = (
        df['start'] - pd.to_datetime(df['start'].dt.date.min()))
    return df


def quick():
    primadonna = '/home/seanammirati/dev/audio_visual_gen/example_lyrics/primadonna.txt'
    important = lyric_importance(primadonna)

    print(important)

    with open('token_to_vec.yml', 'r') as f:
        token_to_vec = yaml.load(f)
        lyric_df = pd.DataFrame(token_to_vec)

    with open('img_cls_to_vec.yml', 'r') as f:
        cls_to_vec = yaml.load(f)
        class_df = pd.DataFrame(
            {k: v['vec'] for k, v in cls_to_vec.items() if isinstance(v, dict)})

    ret_list = []

    for token, importance in important:
        if token in lyric_df.columns:
            vec = lyric_df[token]
            compare_to_classes = ((class_df.T - vec) ** 2).sum(axis=1)

            min_name = compare_to_classes.idxmin()
            min_score = compare_to_classes[min_name]

            ret_list.append({min_name: min_score * (1/importance)})

    terms = pd.DataFrame(ret_list).mean().sort_values()[0:12].index.tolist()
    print('12 Highest Average Matches:')
    ret_vals = [cls_to_vec[t]['id'] for t in terms]
    print(ret_vals)
    print(terms)
    return ' '.join(str(x) for x in ret_vals)


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def neg_euclidian_distance(v1, v2):
    return - ((v1 - v2) ** 2).sum()


def weighted_mean(df, total):
    return (df.sum(axis=1) / total).sort_values(ascending=False)


def max_df(df, total):
    return df.max(axis=1).sort_values(ascending=False)


def return_scaled_matches(token_list, lyric_df, class_df, dist='cosine', agg='max'):

    if agg == 'max':
        agg_func = max_df
    elif agg == 'mean':
        agg_func = weighted_mean

    if dist == 'cosine':
        dist_func = cosine_similarity
    elif dist == 'euclid':
        dist_func = neg_euclidian_distance

    ret_df = pd.DataFrame()
    sum_importances = 0
    for token, importance in token_list:

        if not importance:
            continue
        sum_importances += importance

        if token in lyric_df.columns:
            vec = lyric_df[token]

            try:
                compare_to_classes = (class_df.apply(
                    lambda x: dist_func(x, vec)))
            except Exception as e:
                print(e)
                import pdb
                pdb.set_trace()
            ret_df[token] = compare_to_classes * importance

    # return (ret_df.sum(axis=1) / sum_importances).sort_values(ascending=False)
    return agg_func(ret_df, sum_importances)


def collapse_lines(processed_df, batches=1):

    if batches == 1:
        return processed_df
    processed_df = processed_df.copy()
    lowers = []
    token_cols = processed_df.columns[processed_df.columns.str.contains(
        'tokens')]
    for i in range(0, len(processed_df) // batches):
        range_lower = i * batches
        range_upper = (i * batches) + batches

        processed_df.loc[range_lower,
                         token_cols] = processed_df.loc[range_lower:range_upper, token_cols].sum()

        lowers.append(range_lower)

    return processed_df.loc[lowers].reset_index(drop=True)


def process_lrc_song(songname, n_classes=12, line_batch=1, batches=[1, 2, 4, 8], load=True):
    if load:
        if os.path.exists(f'{songname}/processes_lrc_{n_classes}_{batches}.pkl'):
            with open(f'{songname}/processes_lrc_{n_classes}_{batches}.pkl', 'rb') as f:
                return pickle.load(f)
        else:
            pass
    primadonna = f'/home/seanammirati/dev/audio_visual_gen/example_lyrics/{songname}.lrc'
    df = process_lrc(primadonna)
    df['tokens_eq'] = df['lyrics'].apply(
        lambda x: lyric_importance_str(x, 'eq'))
    df['tokens_cone_1'] = df['lyrics'].apply(
        lambda x: lyric_importance_str(x, 'cone', 1))
    df['tokens_cone_2'] = df['lyrics'].apply(
        lambda x: lyric_importance_str(x, 'cone', 2))
    df['tokens_last'] = df['lyrics'].apply(lambda x: lyric_importance_str(x))
    df['tokens_first'] = df['lyrics'].apply(
        lambda x: lyric_importance_str(x, 'first'))

    df_list = []
    temp = df.copy()
    args = [('cosine', 'mean', 'tokens_eq'), ('cosine', 'max', 'tokens_first'),
            ('euclid', 'max', 'tokens_eq'), ('euclid', 'max', 'tokens_eq')]
    for i in batches:

        df = collapse_lines(temp, i)
        df_list.append(df)
    with open(f'{songname}/token_to_vec.yml', 'r') as f:
        token_to_vec = yaml.load(f)
        lyric_df = pd.DataFrame(token_to_vec)

    with open(f'{songname}/img_cls_to_vec.yml', 'r') as f:
        cls_to_vec = yaml.load(f)
        class_df = pd.DataFrame(
            {k: v['vec'] for k, v in cls_to_vec.items() if isinstance(v, dict)})

    res_list = []
    universal = False
    for j, df in enumerate(df_list):
        which_args = j % 4
        dist, agg, tokens = args[which_args]
        scaled = df[tokens].apply(
            lambda x: return_scaled_matches(x, lyric_df, class_df, dist=dist, agg=agg))

        if not universal:
            universal = [cls_to_vec[name]['id']
                         for name in scaled.mean().sort_values(ascending=False).index[0:n_classes].tolist()]

        top_n = scaled.apply(
            lambda x: x.sort_values(ascending=False).index[0:n_classes].tolist(), axis=1)
        top_n_id = top_n.apply(lambda x: [cls_to_vec[n]['id'] for n in x])
        cnct = pd.concat([df, top_n_id, top_n], axis=1)
        res_list.append(cnct)
    res_list.append(universal)

    with open(f'{songname}/processes_lrc_{n_classes}_{batches}.pkl', 'wb') as f:
        pickle.dump(res_list, f)
    return res_list


if __name__ == '__main__':

    vectorize_lyrics('you_oughta_know')
    df = process_lrc_song('you_oughta_know', 3, load=False)
    print(df)
