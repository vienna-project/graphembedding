"""
Copyright 2020, All rights reserved.
Author : SangJae Kang
Mail : craftsangjae@gmail.com
"""
import pandas as pd
from tqdm import tqdm


def load_github(name='linux',
                event_types=("WatchEvent", 'PushEvent', 'IssuesEvent'),
                k_core=5):
    """knowledge graph Dataset을 불러오는 함수
    현재 3가지 github knowledge graph가 구성되어 있음
    name : (linux, tensorflow, vim)
    event_types : (CommitCommentEvent, CreateEvent, DeleteEvent,
                   ForkEvent, GollumEvent, IssueCommentEvent,
                   IssuesEvent, MemberEvent, PublicEvent,
                   PullRequestEvent, PullRequestReviewCommentEvent,
                   PushEvent, ReleaseEvent, WatchEvent)
    """
    from tensorflow.keras.utils import get_file
    fpath = get_file("github-playground.h5",
                     "https://storage.googleapis.com/github-playground/playground.h5")
    target_df = pd.read_hdf(fpath, key=name)

    type_df = pd.read_hdf(fpath, key='type')
    target_df.type = target_df.type.map(type_df.type.to_dict())

    repository_df = pd.read_hdf(fpath, key='repository')
    df = pd.merge(target_df, repository_df)

    df.rename({
        "actor_id": 'subject',
        "type": 'relation',
        "repo_name": "object"}, axis=1, inplace=True)

    # 필요한 event type만 가져오기
    df = trim_relations(df, event_types)

    # K-core Sampling 수행
    df = kcore_sampling(df, k_core)
    return df


def trim_relations(df, event_types):
    return df[df.relation.isin(event_types)]


def kcore_sampling(df, k_core=5):
    print("Start K-core Sampling", flush=True)
    pbar = tqdm()

    while True:
        prev_counts = len(df)
        if prev_counts == 0:
            raise ValueError("No data remains")

        sub_counts = df.subject.value_counts()
        obj_counts = df.object.value_counts()
        df = df[df.subject.isin(sub_counts[sub_counts >= k_core].index)
                & df.object.isin(obj_counts[obj_counts >= k_core].index)]

        if prev_counts == len(df):
            # 변화가 없으면 종료
            return df
        pbar.update(1)
    return df
