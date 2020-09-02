"""
Copyright 2020, All rights reserved.
Author : SangJae Kang
Mail : craftsangjae@gmail.com
"""
import pandas as pd


def load_github(name='linux',
                event_types=("WatchEvent", 'PushEvent', 'IssuesEvent')):
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
    target_df.type = target_df.type.map(type_df['type'].to_dict())

    repository_df = pd.read_hdf(fpath, key='repository')
    df = pd.merge(target_df, repository_df)

    df.rename({
        "actor_id": 'subject',
        "type": 'relation',
        "repo_name": "object"}, axis=1, inplace=True)

    # 필요한 event type만 가져오기
    df = trim_relations(df, event_types)
    return df


def trim_relations(df, event_types):
    return df[df.relation.isin(event_types)]
  