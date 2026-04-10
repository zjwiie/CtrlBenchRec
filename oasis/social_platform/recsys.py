# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
'''Note that you need to check if it exceeds max_rec_post_len when writing
into rec_matrix'''
import heapq
import json
import logging
import os.path
import pickle
import random
import time
from ast import literal_eval
from datetime import datetime
from math import log
from typing import Any, Dict, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tool.embedding_processor import EmbeddingProcessor
from tool.path_constants import PathConstants
from .process_recsys_posts import (generate_post_vector,
                                   generate_post_vector_openai)
from .typing import ActionType, RecsysType

from model.narm_recommender import NarmRecommender
from model.sasrec_recommender import SasRecRecommender

rec_log = logging.getLogger(name='social.rec')
rec_log.setLevel('DEBUG')

# Initially set to None, to be assigned once again in the recsys function
model = None
twhin_tokenizer = None
twhin_model = None

# Create the TF-IDF model
tfidf_vectorizer = TfidfVectorizer()
# Prepare the twhin model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# All historical tweets and the most recent tweet of each user
user_previous_post_all = {}
user_previous_post = {}
user_profiles = []
# Get the {post_id: content} dict
t_items = {}
# Get the {uid: follower_count} dict
# It's necessary to ensure that agent registration is sequential, with the
# relationship of user_id=agent_id+1; disorder in registration will cause
# issues here
u_items = {}
# Get the creation times of all tweets, assigning scores based on how recent
# they are
date_score = []


def get_twhin_tokenizer():
    global twhin_tokenizer
    if twhin_tokenizer is None:
        from transformers import AutoTokenizer
        twhin_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=PathConstants.twhin_bert_base_model_path,
            model_max_length=512)
    return twhin_tokenizer


def get_twhin_model(device):
    global twhin_model
    if twhin_model is None:
        from transformers import AutoModel
        twhin_model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=PathConstants.twhin_bert_base_model_path).to(device)
    return twhin_model

def get_bge_tokenizer():
    global bge_tokenizer
    if bge_tokenizer is None:
        from transformers import AutoTokenizer
        bge_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=PathConstants.bge_model_path,
            model_max_length=512)
    return bge_tokenizer


def get_bge_model(device):
    global bge_model
    if bge_model is None:
        from transformers import AutoModel
        bge_model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=PathConstants.bge_model_path).to(device)
    return bge_model


def load_model(model_name):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_name == 'paraphrase-MiniLM-L6-v2':
            return SentenceTransformer("../encoder/paraphrase-MiniLM-L6-v2",
                                       device=device)
        elif model_name == 'Twitter/twhin-bert-base':
            twhin_tokenizer = get_twhin_tokenizer()
            twhin_model = get_twhin_model(device)
            return twhin_tokenizer, twhin_model
        elif model_name == 'bge':
            bge_tokenizer = get_bge_tokenizer()
            bge_model = get_bge_model(device)
            return bge_tokenizer, bge_model
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    except Exception as e:
        raise Exception(f"Failed to load the model: {model_name}") from e


def get_recsys_model(recsys_type: str = None):
    if recsys_type == RecsysType.TWITTER.value:
        model = load_model('paraphrase-MiniLM-L6-v2')
        return model
    elif recsys_type == RecsysType.TWHIN.value:
        twhin_tokenizer, twhin_model = load_model("Twitter/twhin-bert-base")
        models = (twhin_tokenizer, twhin_model)
        return models
    elif (recsys_type == RecsysType.REDDIT.value
          or recsys_type == RecsysType.RANDOM.value):
        return None
    elif (recsys_type == RecsysType.BGE.value):
        bge_tokenizer,bge_model = load_model("bge")
        models = (bge_tokenizer, bge_model)
        return models
    else:
        raise ValueError(f"Unknown recsys type: {recsys_type}")


# Move model to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if model is not None:
    model.to(device)
else:
    pass


# Reset global variables
def reset_globals():
    global user_previous_post_all, user_previous_post
    global user_profiles, t_items, u_items
    global date_score
    user_previous_post_all = {}
    user_previous_post = {}
    user_profiles = []
    t_items = {}
    u_items = {}
    date_score = []


def rec_sys_random(post_table: List[Dict[str, Any]], rec_matrix: List[List],
                   max_rec_post_len: int) -> List[List]:
    """
    Randomly recommend posts to users.

    Args:
        user_table (List[Dict[str, Any]]): List of users.
        post_table (List[Dict[str, Any]]): List of posts.
        trace_table (List[Dict[str, Any]]): List of user interactions.
        rec_matrix (List[List]): Existing recommendation matrix.
        max_rec_post_len (int): Maximum number of recommended posts.

    Returns:
        List[List]: Updated recommendation matrix.
    """
    # Get all post IDs
    post_ids = [post['post_id'] for post in post_table]
    new_rec_matrix = []
    if len(post_ids) <= max_rec_post_len:
        # If the number of posts is less than or equal to the maximum number
        # of recommendations, each user gets all post IDs
        new_rec_matrix = [post_ids] * len(rec_matrix)
    else:
        # If the number of posts is greater than the maximum number of
        # recommendations, each user randomly gets a specified number of post
        # IDs
        for _ in range(len(rec_matrix)):
            new_rec_matrix.append(random.sample(post_ids, max_rec_post_len))

    return new_rec_matrix


def calculate_hot_score(num_likes: int, num_dislikes: int,
                        created_at: datetime) -> int:
    """
    Compute the hot score for a post.

    Args:
        num_likes (int): Number of likes.
        num_dislikes (int): Number of dislikes.
        created_at (datetime): Creation time of the post.

    Returns:
        int: Hot score of the post.

    Reference:
        https://medium.com/hacking-and-gonzo/how-reddit-ranking-algorithms-work-ef111e33d0d9
    """
    s = num_likes - num_dislikes
    order = log(max(abs(s), 1), 10)
    sign = 1 if s > 0 else -1 if s < 0 else 0

    # epoch_seconds
    epoch = datetime(1970, 1, 1)
    td = created_at - epoch
    epoch_seconds_result = td.days * 86400 + td.seconds + (
        float(td.microseconds) / 1e6)

    seconds = epoch_seconds_result - 1134028003
    return round(sign * order + seconds / 45000, 7)


def get_recommendations(
    user_index,
    cosine_similarities,
    items,
    score,
    top_n=100,
):
    similarities = np.array(cosine_similarities[user_index])
    similarities = similarities * score
    top_item_indices = similarities.argsort()[::-1][:top_n]
    recommended_items = [(list(items.keys())[i], similarities[i])
                         for i in top_item_indices]
    return recommended_items


def rec_sys_reddit(post_table: List[Dict[str, Any]], rec_matrix: List[List],
                   max_rec_post_len: int) -> List[List]:
    """
    Recommend posts based on Reddit-like hot score.

    Args:
        post_table (List[Dict[str, Any]]): List of posts.
        rec_matrix (List[List]): Existing recommendation matrix.
        max_rec_post_len (int): Maximum number of recommended posts.

    Returns:
        List[List]: Updated recommendation matrix.
    """
    # Get all post IDs
    post_ids = [post['post_id'] for post in post_table]

    if len(post_ids) <= max_rec_post_len:
        # If the number of posts is less than or equal to the maximum number
        # of recommendations, each user gets all post IDs
        new_rec_matrix = [post_ids] * len(rec_matrix)
    else:
        # The time complexity of this recommendation system is
        # O(post_num * log max_rec_post_len)
        all_hot_score = []
        for post in post_table:
            try:
                created_at_dt = datetime.strptime(post['created_at'],
                                                  "%Y-%m-%d %H:%M:%S.%f")
            except Exception:
                created_at_dt = datetime.strptime(post['created_at'],
                                                  "%Y-%m-%d %H:%M:%S")
            hot_score = calculate_hot_score(post['num_likes'],
                                            post['num_dislikes'],
                                            created_at_dt)
            all_hot_score.append((hot_score, post['post_id']))
        # Sort
        top_posts = heapq.nlargest(max_rec_post_len,
                                   all_hot_score,
                                   key=lambda x: x[0])
        top_post_ids = [post_id for _, post_id in top_posts]

        # If the number of posts is greater than the maximum number of
        # recommendations, each user gets a specified number of post IDs
        # randomly
        new_rec_matrix = [top_post_ids] * len(rec_matrix)

    return new_rec_matrix


def rec_sys_personalized(user_table: List[Dict[str, Any]],
                         post_table: List[Dict[str, Any]],
                         trace_table: List[Dict[str,
                                                Any]], rec_matrix: List[List],
                         max_rec_post_len: int) -> List[List]:
    """
    Recommend posts based on personalized similarity scores.

    Args:
        user_table (List[Dict[str, Any]]): List of users.
        post_table (List[Dict[str, Any]]): List of posts.
        trace_table (List[Dict[str, Any]]): List of user interactions.
        rec_matrix (List[List]): Existing recommendation matrix.
        max_rec_post_len (int): Maximum number of recommended posts.

    Returns:
        List[List]: Updated recommendation matrix.
    """
    global model
    if model is None or isinstance(model, tuple):
        model = get_recsys_model(recsys_type="twitter")

    post_ids = [post['post_id'] for post in post_table]
    print(
        f'Running personalized recommendation for {len(user_table)} users...')
    start_time = time.time()
    new_rec_matrix = []
    if len(post_ids) <= max_rec_post_len:
        # If the number of posts is less than or equal to the maximum
        # recommended length, each user gets all post IDs
        new_rec_matrix = [post_ids] * len(rec_matrix)
    else:
        # If the number of posts is greater than the maximum recommended
        # length, each user gets personalized post IDs
        user_bios = [
            user['bio'] if 'bio' in user and user['bio'] is not None else ''
            for user in user_table
        ]
        post_contents = [post['content'] for post in post_table]

        if model:
            user_embeddings = model.encode(user_bios,
                                           convert_to_tensor=True,
                                           device=device)
            post_embeddings = model.encode(post_contents,
                                           convert_to_tensor=True,
                                           device=device)

            # Compute dot product similarity
            dot_product = torch.matmul(user_embeddings, post_embeddings.T)

            # Compute norm
            user_norms = torch.norm(user_embeddings, dim=1)
            post_norms = torch.norm(post_embeddings, dim=1)

            # Compute cosine similarity
            similarities = dot_product / (user_norms[:, None] *
                                          post_norms[None, :])

        else:
            # Generate random similarities
            similarities = torch.rand(len(user_table), len(post_table))

        # Iterate through each user to generate personalized recommendations.
        for user_index, user in enumerate(user_table):
            # Filter out posts made by the current user.
            filtered_post_indices = [
                i for i, post in enumerate(post_table)
                if post['user_id'] != user['user_id']
            ]

            user_similarities = similarities[user_index, filtered_post_indices]

            # Get the corresponding post IDs for the filtered posts.
            filtered_post_ids = [
                post_table[i]['post_id'] for i in filtered_post_indices
            ]

            # Determine the top posts based on the similarities, limited by
            # max_rec_post_len.
            _, top_indices = torch.topk(user_similarities,
                                        k=min(max_rec_post_len,
                                              len(filtered_post_ids)))

            top_post_ids = [filtered_post_ids[i] for i in top_indices.tolist()]

            # Append the top post IDs to the new recommendation matrix.
            new_rec_matrix.append(top_post_ids)

    end_time = time.time()
    print(f'Personalized recommendation time: {end_time - start_time:.6f}s')
    return new_rec_matrix


def get_like_post_id(user_id, action, trace_table):
    """
    Get the post IDs that a user has liked or unliked.

    Args:
        user_id (str): ID of the user.
        action (str): Type of action (like or unlike).
        post_table (list): List of posts.
        trace_table (list): List of user interactions.

    Returns:
        list: List of post IDs.
    """
    # Get post IDs from trace table for the given user and action
    trace_post_ids = [
        literal_eval(trace['info'])["post_id"] for trace in trace_table
        if (trace['user_id'] == user_id and trace['action'] == action)
    ]
    """Only take the last 50 liked posts, if not enough, pad with the most
    recently liked post. Only take IDs, not content, because calculating
    embeddings for all posts again is very time-consuming, especially when the
    number of agents is large"""
    # if len(trace_post_ids) < 50 and len(trace_post_ids) > 0:
    #     trace_post_ids += [trace_post_ids[-1]] * (50 - len(trace_post_ids))
    #     print(f"oasis.social_platform.recsys.get_like_post_id:{trace_post_ids}")
    # elif len(trace_post_ids) > 50:
    #     trace_post_ids = trace_post_ids[-50:]
    # else:
    #     trace_post_ids = [0]

    new_trace_post_ids = [int(pid) for pid in trace_post_ids]

    return new_trace_post_ids


# Calculate the average cosine similarity between liked posts and target posts
def calculate_like_similarity(liked_vectors, target_vectors):
    # Calculate the norms of the vectors
    liked_norms = np.linalg.norm(liked_vectors, axis=1)
    target_norms = np.linalg.norm(target_vectors, axis=1)
    # Calculate dot products
    dot_products = np.dot(target_vectors, liked_vectors.T)
    # Calculate cosine similarities
    cosine_similarities = dot_products / np.outer(target_norms, liked_norms)
    # Take the average
    average_similarities = np.mean(cosine_similarities, axis=1)

    return average_similarities


def coarse_filtering(input_list, scale):
    """
    Coarse filtering posts and return selected elements with their indices.
    """
    if len(input_list) <= scale:
        # Return elements and their indices as list of tuples (element, index)
        sampled_indices = range(len(input_list))
        return (input_list, sampled_indices)
    else:
        # Get random sample of scale elements
        sampled_indices = random.sample(range(len(input_list)), scale)
        sampled_elements = [input_list[idx] for idx in sampled_indices]
        # return [(input_list[idx], idx) for idx in sampled_indices]
        return (sampled_elements, sampled_indices)
#
# # TODO 使用twhin-bert推荐时，取消注释
# def rec_sys_personalized_twh(
#         user_table: List[Dict[str, Any]],
#         post_table: List[Dict[str, Any]],
#         latest_post_count: int,
#         trace_table: List[Dict[str, Any]],
#         rec_matrix: List[List],
#         max_rec_post_len: int,
#         enable_like_score: bool = True,
#         use_openai_embedding: bool = False) -> List[List]:
#     enable_like_score = True
#     global twhin_model, twhin_tokenizer
#     if twhin_model is None or twhin_tokenizer is None:
#         twhin_tokenizer, twhin_model = get_recsys_model(
#             recsys_type="twhin-bert")
#     # Set some global variables to reduce time consumption
#     global date_score, t_items, u_items, user_previous_post
#     global user_previous_post_all, user_profiles
#     # Get the uid: follower_count dict
#     # Update only once, unless adding the feature to include new users midway.
#     if (not u_items) or len(u_items) != len(user_table):
#         u_items = {
#             user['user_id']: user["num_followers"]
#             for user in user_table
#         }
#     if not user_previous_post_all or len(user_previous_post_all) != len(
#             user_table):
#         # Each user must have a list of historical tweets
#         user_previous_post_all = {
#             index: []
#             for index in range(len(user_table))
#         }
#         user_previous_post = {index: "" for index in range(len(user_table))}
#     if not user_profiles or len(user_profiles) != len(user_table):
#         for user in user_table:
#             if user['bio'] is None:
#                 user_profiles.append('This user does not have profile')
#             else:
#                 user_profiles.append(user['bio'])
#     # TODO 保存user_profile的embbeding
#
#
#     if len(t_items) < len(post_table):
#         for post in post_table[-latest_post_count:]:
#             # Get the {post_id: content} dict, update only the latest tweets
#             t_items[post['post_id']] = post['content']
#             # Update the user's historical tweets
#             # TODO 删除发帖记录
#             user_previous_post_all[post['user_id']].append(post['content'])
#             user_previous_post[post['user_id']] = post['content']
#             # Get the creation times of all tweets, assigning scores based on
#             # how recent they are, note that this algorithm can run for a
#             # maximum of 90 time steps
#             date_score.append(1.0)
#
#     date_score_np = np.array(date_score)
#
#     like_post_ids_all = []
#     if enable_like_score:
#         # Calculate similarity with previously liked content, first gather
#         # liked post ids from the trace
#         for user in user_table:
#             user_id = user['agent_id']
#             like_post_ids = get_like_post_id(user_id,
#                                              ActionType.LIKE_POST.value,
#                                              trace_table)
#             # TODO 添加refresh的post_id
#             like_post_ids_all.append(like_post_ids)
#             rec_log.info(f"user_id:{user_id},like_post_ids:{like_post_ids}")
#     rec_log.info(f"like_post_ids_all:{like_post_ids_all}")
#     scores = date_score_np
#     rec_log.info(f"oasis.social_platform.recsys.rec_sys_personalized_twh:原始scores:{scores}")
#     new_rec_matrix = []
#     if len(post_table) <= max_rec_post_len:
#         # If the number of tweets is less than or equal to the max
#         # recommendation count, each user gets all post IDs
#         tids = [t['post_id'] for t in post_table]
#         new_rec_matrix = [tids] * (len(rec_matrix))
#
#     else:
#         # If the number of tweets is greater than the max recommendation
#         # count, each user randomly gets personalized post IDs
#
#         # This requires going through all users to update their profiles,
#         # which is a time-consuming operation
#         for post_user_index in user_previous_post:
#             break
#             try:
#                 # Directly replacing the profile with the latest tweet will
#                 # cause the recommendation system to repeatedly push other
#                 # reposts to users who have already shared that tweet
#                 # user_profiles[post_user_index] =
#                 # user_previous_post[post_user_index]
#                 # Instead, append the description of the Recent post's content
#                 # to the end of the user char
#                 update_profile = (
#                     f" # Recent post:{user_previous_post[post_user_index]}")
#                 if user_previous_post[post_user_index] != "":
#                     # If there's no update for the recent post, add this part
#                     if "# Recent post:" not in user_profiles[post_user_index]:
#                         user_profiles[post_user_index] += update_profile
#                     # If the profile has a recent post but it's not the user's
#                     # latest, replace it
#                     elif update_profile not in user_profiles[post_user_index]:
#                         user_profiles[post_user_index] = user_profiles[
#                             post_user_index].split(
#                                 "# Recent post:")[0] + update_profile
#             except Exception:
#                 print("update previous post failed")
#
#         # coarse filtering 4000 posts due to the memory constraint.
#         filtered_posts_tuple = coarse_filtering(list(t_items.values()), 10000)
#         corpus = user_profiles + filtered_posts_tuple[0]
#         # corpus = user_profiles + list(t_items.values())
#         tweet_vector_start_t = time.time()
#         if use_openai_embedding:
#             all_post_vector_list = generate_post_vector_openai(corpus,
#                                                                batch_size=1000)
#         else:
#             all_post_vector_list = generate_post_vector(twhin_model,
#                                                         twhin_tokenizer,
#                                                         corpus,
#                                                         batch_size=1000)
#         tweet_vector_end_t = time.time()
#         rec_log.info(
#             f"twhin model cost time: {tweet_vector_end_t-tweet_vector_start_t}"
#         )
#         user_vector = all_post_vector_list[:len(user_profiles)]
#         rec_log.info(f"user_vector.shape:{user_vector.shape}")
#         posts_vector = all_post_vector_list[len(user_profiles):]
#         rec_log.info(f"posts_vector.shape:{posts_vector.shape}")
#         embedding_file_path = '..\\embeddings\\embeddings.npz'
#         EmbeddingProcessor.save_embedding(user_vector, posts_vector, embedding_file_path)
#         (a,b) = EmbeddingProcessor.load_embedding(embedding_file_path)
#
#         user_behavior_vector_all = []
#         mean_like_posts_vector_all = []
#         if enable_like_score:
#             # Traverse all liked post ids, collecting liked post vectors from
#             # posts_vector for matrix acceleration calculation
#             rec_log.info(f"user_id_length:{len(like_post_ids_all)}")
#             for user_idx, like_post_ids in enumerate(like_post_ids_all):
#                 rec_log.info(f"user_idx:{user_idx};like_post_ids:{like_post_ids}")
#                 like_posts_vector = []
#                 for like_post_id in like_post_ids:
#                     try:
#                         like_posts_vector.append(posts_vector[like_post_id - 1])
#                         # rec_log.info(f"like_post_id:{like_post_id}")
#                     except Exception:
#                         rec_log.info(f"Exception!:user_idx:{user_idx},like_post_id:{like_post_id}")
#                         like_posts_vector.append(user_vector[user_idx])
#                 mean_like_posts_vector = np.mean(np.array(like_posts_vector),axis=0)
#                 # rec_log.info(f"mean_like_posts_vector.shape:{mean_like_posts_vector.shape}")
#                 user_behavior_vector = np.array(user_vector[user_idx] * 0.2 + mean_like_posts_vector * 0.8)
#                 # rec_log.info(f"user_behavior_vector.shape:{user_behavior_vector.shape}")
#                 # TODO attention
#                 user_behavior_vector_all.append(user_behavior_vector)
#                 mean_like_posts_vector_all.append(mean_like_posts_vector)
#
#         EmbeddingProcessor.save_behavior_embedding(np.array(mean_like_posts_vector_all))
#         # rec_log.info(f"{np.array(user_behavior_vector_all)},{np.array(user_behavior_vector_all).shape}")
#         get_similar_start_t = time.time()
#         cosine_similarities = cosine_similarity(np.array(user_behavior_vector_all), posts_vector)
#         # 计算帖子之间的余弦相似度
#         get_similar_end_t = time.time()
#         rec_log.info(f"get cosine_similarity time: "
#                      f"{get_similar_end_t-get_similar_start_t}")
#
#         for user_index, profile in enumerate(user_profiles):
#              user_like_post_ids = like_post_ids_all[user_index]
#              for like_post_id in user_like_post_ids:
#                 cosine_similarities[user_index][like_post_id-1] = -2
#
#         if os.path.exists(PathConstants.total_rec_history_path):
#             with open(PathConstants.total_rec_history_path,'r',encoding='utf-8') as f:
#                 rec_history_list = json.load(f)
#                 for user_index in range(len(rec_history_list)):
#                     rec_history = rec_history_list[user_index]
#                     for rec_post_id in rec_history:
#                         cosine_similarities[user_index][int(rec_post_id)-1] = -2
#
#
#         with open(PathConstants.generated_user_post_cosine_similarity_file_path, 'w') as f:
#             json.dump(cosine_similarities.tolist(), f)  # 直接保存
#         cosine_similarities = torch.tensor(cosine_similarities)
#         rec_log.info(f"oasis.social_platform.recsys.rec_sys_personalized_twh:cosine_similarities.shape:{cosine_similarities.shape}")
#         value, indices = torch.topk(cosine_similarities,
#                                     max_rec_post_len,
#                                     dim=1,
#                                     largest=True,
#                                     sorted=True)
#         filter_posts_index = filtered_posts_tuple[1]
#         filter_posts_index = torch.tensor(filter_posts_index)
#         indices = filter_posts_index[indices]
#         matrix_list = indices.cpu().numpy()
#         post_list = list(t_items.keys())
#         for rec_ids in matrix_list:
#             rec_ids = [post_list[i] for i in rec_ids]
#             new_rec_matrix.append(rec_ids)
#
#     return new_rec_matrix

#
# # TODO 使用bge推荐时，取消注释
# def rec_sys_personalized_twh(
#         user_table: List[Dict[str, Any]],
#         post_table: List[Dict[str, Any]],
#         latest_post_count: int,
#         trace_table: List[Dict[str, Any]],
#         rec_matrix: List[List],
#         max_rec_post_len: int,
#         enable_like_score: bool = True,
#         use_openai_embedding: bool = False) -> List[List]:
#     global twhin_model, twhin_tokenizer, bge_model
#     global date_score, t_items, u_items, user_previous_post
#     global user_previous_post_all, user_profiles
#
#     # 1. 模型初始化
#     twhin_tokenizer, twhin_model = get_recsys_model(
#         recsys_type="twhin-bert")
#
#     bge_model = SentenceTransformer(PathConstants.bge_model_path, device='cpu')
#
#     # 2. 基础数据维护
#     if (not u_items) or len(u_items) != len(user_table):
#         u_items = {user['user_id']: user["num_followers"] for user in user_table}
#
#     if not user_previous_post_all or len(user_previous_post_all) != len(user_table):
#         user_previous_post_all = {index: [] for index in range(len(user_table))}
#         user_previous_post = {index: "" for index in range(len(user_table))}
#
#     if not user_profiles or len(user_profiles) != len(user_table):
#         user_profiles = []
#         for user in user_table:
#             user_profiles.append(user.get('bio') or 'This user does not have profile')
#
#     # 3. 更新帖子缓存
#     if len(t_items) < len(post_table):
#         for post in post_table[-latest_post_count:]:
#             t_items[post['post_id']] = post['content']
#             user_previous_post_all[post['user_id']].append(post['content'])
#             user_previous_post[post['user_id']] = post['content']
#             date_score.append(1.0)
#
#     # 4. 收集用户点赞历史
#     like_post_ids_all = []
#     enable_like_score = True
#     if enable_like_score:
#         for user in user_table:
#             user_id = user['agent_id']
#             like_post_ids = get_like_post_id(user_id, ActionType.LIKE_POST.value, trace_table)
#             like_post_ids_all.append(like_post_ids)
#
#     new_rec_matrix = []
#
#     # 5. 推荐逻辑判断
#     if len(post_table) <= max_rec_post_len:
#         tids = [t['post_id'] for t in post_table]
#         new_rec_matrix = [tids] * (len(rec_matrix))
#     else:
#         # --- 向量生成阶段 ---
#         # 粗筛 10000 条帖子
#         filtered_posts_tuple = coarse_filtering(list(t_items.values()), 10000)
#         post_contents = filtered_posts_tuple[0]
#         filter_posts_index = filtered_posts_tuple[1]
#
#         # A. 生成 TWHIN-BERT 向量并保存 (存档需求)
#         rec_log.info("Generating TWHIN-BERT embeddings for storage...")
#         twhin_corpus = user_profiles + post_contents
#         # 调用你原有的 generate_post_vector 函数
#         twhin_all_vecs = generate_post_vector(twhin_model, twhin_tokenizer, twhin_corpus, batch_size=1000)
#
#         t_user_vec = twhin_all_vecs[:len(user_profiles)].numpy()
#         t_post_vec = twhin_all_vecs[len(user_profiles):].numpy()
#         EmbeddingProcessor.save_embedding(t_user_vec, t_post_vec, PathConstants.embedding_storage_file_path)
#
#         # B. 生成 BGE 向量 (实际推荐需求)
#         rec_log.info("Generating BGE embeddings for recommendation...")
#         # BGE 检索指令前缀
#         instruction = "Represent this sentence for searching relevant passages: "
#         bge_corpus = [instruction + p for p in user_profiles] + post_contents
#
#         bge_start_t = time.time()
#         # 直接使用 SentenceTransformer 的 encode，开启归一化
#         bge_all_vecs = bge_model.encode(
#             bge_corpus,
#             batch_size=1000,
#             normalize_embeddings=True,
#             convert_to_numpy=True
#         )
#         rec_log.info(f"BGE encoding cost time: {time.time() - bge_start_t:.2f}s")
#
#         user_vector = bge_all_vecs[:len(user_profiles)]
#         posts_vector = bge_all_vecs[len(user_profiles):]
#
#         # 6. 用户行为建模 (融合点赞偏好)
#         user_behavior_vector_all = []
#         for user_idx, like_post_ids in enumerate(like_post_ids_all):
#             like_vecs = []
#             for l_id in like_post_ids:
#                 try:
#                     # 注意：此处假设 l_id 对应 posts_vector 的索引，
#                     # 实际可能需要一个 mapping 转换全局 ID 到 filtered 索引
#                     like_vecs.append(posts_vector[l_id - 1])
#                 except:
#                     continue
#
#             if like_vecs:
#                 mean_like_vec = np.mean(np.array(like_vecs), axis=0)
#                 # 融合画像向量与行为向量
#                 combined_vec = user_vector[user_idx] * 0.2 + mean_like_vec * 0.8
#                 user_behavior_vector_all.append(combined_vec)
#             else:
#                 # 兜底：无点赞则使用原始画像向量
#                 user_behavior_vector_all.append(user_vector[user_idx])
#
#         # --- 7. 相似度计算 (修复 1D Array 报错) ---
#         query_vectors = np.array(user_behavior_vector_all)
#         if query_vectors.ndim == 1:
#             query_vectors = query_vectors.reshape(len(user_profiles), -1)
#
#         get_similar_start_t = time.time()
#         cosine_similarities = cosine_similarity(query_vectors, posts_vector)
#         rec_log.info(f"Similarity calculation cost time: {time.time() - get_similar_start_t:.2f}s")
#
#         # 8. 过滤与负采样
#         for user_idx in range(len(user_profiles)):
#             # 过滤已点赞
#             for l_id in like_post_ids_all[user_idx]:
#                 # 简单防越界处理
#                 if l_id - 1 < cosine_similarities.shape[1]:
#                     cosine_similarities[user_idx][l_id - 1] = -2
#
#         # 过滤历史推荐
#         if os.path.exists(PathConstants.total_rec_history_path):
#             with open(PathConstants.total_rec_history_path, 'r', encoding='utf-8') as f:
#                 rec_history_list = json.load(f)
#                 for u_idx in range(min(len(rec_history_list), len(user_profiles))):
#                     for h_id in rec_history_list[u_idx]:
#                         idx = int(h_id) - 1
#                         if idx < cosine_similarities.shape[1]:
#                             cosine_similarities[u_idx][idx] = -2
#
#         # 9. Top-K 选择与索引映射
#         cosine_similarities_torch = torch.from_numpy(cosine_similarities)
#         _, topk_indices = torch.topk(cosine_similarities_torch, max_rec_post_len, dim=1)
#
#         # 映射回原始 ID
#         filter_posts_index_tensor = torch.tensor(filter_posts_index)
#         final_indices = filter_posts_index_tensor[topk_indices].cpu().numpy()
#
#         post_keys_list = list(t_items.keys())
#         for user_rec_indices in final_indices:
#             rec_ids = [post_keys_list[i] for i in user_rec_indices]
#             new_rec_matrix.append(rec_ids)
#
#     return new_rec_matrix
#
# #TODO 使用narm推荐时，取消注释
# narm_engine = NarmRecommender()
# def rec_sys_personalized_twh(
#         user_table: List[Dict[str, Any]],
#         post_table: List[Dict[str, Any]],
#         latest_post_count: int,
#         trace_table: List[Dict[str, Any]],
#         rec_matrix: List[List],
#         max_rec_post_len: int,
#         enable_like_score: bool = True,
#         use_openai_embedding: bool = False) -> List[List]:
#     global twhin_model, twhin_tokenizer
#     global date_score, t_items, u_items, user_previous_post
#     global user_previous_post_all, user_profiles
#
#     # 1. 模型初始化 (移除 BGE，保留 TWHIN 用于存档)
#     twhin_tokenizer, twhin_model = get_recsys_model(recsys_type="twhin-bert")
#
#     # 2. 基础数据维护
#     if (not u_items) or len(u_items) != len(user_table):
#         u_items = {user['user_id']: user["num_followers"] for user in user_table}
#
#     if not user_previous_post_all or len(user_previous_post_all) != len(user_table):
#         user_previous_post_all = {index: [] for index in range(len(user_table))}
#         user_previous_post = {index: "" for index in range(len(user_table))}
#
#     if not user_profiles or len(user_profiles) != len(user_table):
#         user_profiles = []
#         for user in user_table:
#             user_profiles.append(user.get('bio') or 'This user does not have profile')
#
#     # 3. 更新帖子缓存
#     if len(t_items) < len(post_table):
#         for post in post_table[-latest_post_count:]:
#             t_items[post['post_id']] = post['content']
#             user_previous_post_all[post['user_id']].append(post['content'])
#             user_previous_post[post['user_id']] = post['content']
#             date_score.append(1.0)
#
#     # 4. 收集用户点赞历史 (作为 NARM 的输入序列)
#     like_post_ids_all = []
#     for user in user_table:
#         user_id = user['agent_id']
#         like_post_ids = get_like_post_id(user_id, ActionType.LIKE_POST.value, trace_table)
#         like_post_ids_all.append(like_post_ids)
#
#     new_rec_matrix = []
#
#     # 5. 推荐逻辑判断
#     if len(post_table) <= max_rec_post_len:
#         tids = [t['post_id'] for t in post_table]
#         new_rec_matrix = [tids] * (len(rec_matrix))
#     else:
#         # --- 存档阶段 (保留原有的 TWHIN 逻辑) ---
#         filtered_posts_tuple = coarse_filtering(list(t_items.values()), 10000)
#         post_contents = filtered_posts_tuple[0]
#
#         rec_log.info("Generating TWHIN-BERT embeddings for storage...")
#         twhin_corpus = user_profiles + post_contents
#         twhin_all_vecs = generate_post_vector(twhin_model, twhin_tokenizer, twhin_corpus, batch_size=1000)
#         t_user_vec = twhin_all_vecs[:len(like_post_ids_all)].numpy()
#         t_post_vec = twhin_all_vecs[len(like_post_ids_all):].numpy()
#         EmbeddingProcessor.save_embedding(t_user_vec, t_post_vec, PathConstants.embedding_storage_file_path)
#
#         # --- 6. 加载历史推荐记录作为黑名单 ---
#         rec_history_all = [[] for _ in range(len(like_post_ids_all))]
#         if os.path.exists(PathConstants.total_rec_history_path):
#             with open(PathConstants.total_rec_history_path, 'r', encoding='utf-8') as f:
#                 rec_history_all = json.load(f)
#
#         # --- 7. 利用 NARM 实例进行推荐 ---
#         rec_log.info("Executing NARM recommendation...")
#         with open('../model/dataset.pkl', 'rb') as f:
#             data = pickle.load(f)
#         item_id_and_index_in_narm_map = data['smap']
#         index_in_narm_and_item_id_map = {v: k for k, v in item_id_and_index_in_narm_map.items()}
#         for user_idx in range(len(like_post_ids_all)):
#             # A. 准备输入序列 (点赞历史)
#             user_history = []
#             for like_post_id in like_post_ids_all[user_idx]:
#                 index = -1
#                 try:
#                     index = item_id_and_index_in_narm_map[int(like_post_id+1)]
#                 except Exception:
#                     continue
#                 if index != -1:
#                     user_history.append(index)
#             # B. 准备过滤列表 (历史推荐 + 本次已点赞)
#             history_blacklist = []
#             for rec_idx in rec_history_all[user_idx]:
#                 index = -1
#                 try:
#                     index = item_id_and_index_in_narm_map[int(rec_idx)]
#                 except Exception:
#                     continue
#                 if index != -1:
#                     history_blacklist.append(index)
#             full_blacklist = list(set(history_blacklist + user_history))
#
#             # C. 调用 NARM 实例
#             # 内部已包含 int() 转换和左侧补 0
#             rec_ids, _ = narm_engine.recommend(
#                 custom_ids=user_history,
#                 neg_ids=full_blacklist,
#                 topk=max_rec_post_len
#             )
#
#             # 映射回post_id
#             new_rec_ids = []
#             for rec_idx in rec_ids:
#                 new_rec_ids.append(index_in_narm_and_item_id_map[rec_idx]-1)
#
#             new_rec_matrix.append(new_rec_ids)
#
#     return new_rec_matrix


#TODO 使用sasrec推荐时，取消注释
sasrec_engine = SasRecRecommender()
def rec_sys_personalized_twh(
        user_table: List[Dict[str, Any]],
        post_table: List[Dict[str, Any]],
        latest_post_count: int,
        trace_table: List[Dict[str, Any]],
        rec_matrix: List[List],
        max_rec_post_len: int,
        enable_like_score: bool = True,
        use_openai_embedding: bool = False) -> List[List]:
    global twhin_model, twhin_tokenizer
    global date_score, t_items, u_items, user_previous_post
    global user_previous_post_all, user_profiles

    # 1. 模型初始化 (移除 BGE，保留 TWHIN 用于存档)
    twhin_tokenizer, twhin_model = get_recsys_model(recsys_type="twhin-bert")

    # 2. 基础数据维护
    if (not u_items) or len(u_items) != len(user_table):
        u_items = {user['user_id']: user["num_followers"] for user in user_table}

    if not user_previous_post_all or len(user_previous_post_all) != len(user_table):
        user_previous_post_all = {index: [] for index in range(len(user_table))}
        user_previous_post = {index: "" for index in range(len(user_table))}

    if not user_profiles or len(user_profiles) != len(user_table):
        user_profiles = []
        for user in user_table:
            user_profiles.append(user.get('bio') or 'This user does not have profile')

    # 3. 更新帖子缓存
    if len(t_items) < len(post_table):
        for post in post_table[-latest_post_count:]:
            t_items[post['post_id']] = post['content']
            user_previous_post_all[post['user_id']].append(post['content'])
            user_previous_post[post['user_id']] = post['content']
            date_score.append(1.0)

    # 4. 收集用户点赞历史 (作为 NARM 的输入序列)
    like_post_ids_all = []
    for user in user_table:
        user_id = user['agent_id']
        like_post_ids = get_like_post_id(user_id, ActionType.LIKE_POST.value, trace_table)
        like_post_ids_all.append(like_post_ids)

    new_rec_matrix = []

    # 5. 推荐逻辑判断
    if len(post_table) <= max_rec_post_len:
        tids = [t['post_id'] for t in post_table]
        new_rec_matrix = [tids] * (len(rec_matrix))
    else:
        # --- 存档阶段 (保留原有的 TWHIN 逻辑) ---
        filtered_posts_tuple = coarse_filtering(list(t_items.values()), 10000)
        post_contents = filtered_posts_tuple[0]

        rec_log.info("Generating TWHIN-BERT embeddings for storage...")
        twhin_corpus = user_profiles + post_contents
        twhin_all_vecs = generate_post_vector(twhin_model, twhin_tokenizer, twhin_corpus, batch_size=16)
        t_user_vec = twhin_all_vecs[:len(like_post_ids_all)].numpy()
        t_post_vec = twhin_all_vecs[len(like_post_ids_all):].numpy()
        EmbeddingProcessor.save_embedding(t_user_vec, t_post_vec, PathConstants.embedding_storage_file_path)

        # --- 6. 加载历史推荐记录作为黑名单 ---
        rec_history_all = [[] for _ in range(len(like_post_ids_all))]
        if os.path.exists(PathConstants.total_rec_history_path):
            with open(PathConstants.total_rec_history_path, 'r', encoding='utf-8') as f:
                rec_history_all = json.load(f)

        # --- 7. 利用 SasRec 实例进行推荐 ---
        rec_log.info("Executing SasRec recommendation...")
        with open('../model/dataset.pkl', 'rb') as f:
            data = pickle.load(f)
        item_id_and_index_in_sasrec_map = data['smap']
        index_in_sasrec_and_item_id_map = {v: k for k, v in item_id_and_index_in_sasrec_map.items()}
        for user_idx in range(len(like_post_ids_all)):
            # A. 准备输入序列 (点赞历史)
            user_history = []
            for like_post_id in like_post_ids_all[user_idx]:
                index = -1
                try:
                    index = item_id_and_index_in_sasrec_map[int(like_post_id+1)]
                except Exception:
                    continue
                if index != -1:
                    user_history.append(index)
            # B. 准备过滤列表 (历史推荐 + 本次已点赞)
            history_blacklist = []
            for rec_idx in rec_history_all[user_idx]:
                index = -1
                try:
                    index = item_id_and_index_in_sasrec_map[int(rec_idx)]
                except Exception:
                    continue
                if index != -1:
                    history_blacklist.append(index)
            full_blacklist = list(set(history_blacklist + user_history))

            # C. 调用 NARM 实例
            # 内部已包含 int() 转换和左侧补 0
            rec_ids, _ = sasrec_engine.recommend(
                custom_ids=user_history,
                neg_ids=full_blacklist,
                topk=max_rec_post_len
            )

            # 映射回post_id
            new_rec_ids = []
            for rec_idx in rec_ids:
                new_rec_ids.append(index_in_sasrec_and_item_id_map[rec_idx]-1)

            new_rec_matrix.append(new_rec_ids)

    return new_rec_matrix

def normalize_similarity_adjustments(post_scores, base_similarity,
                                     like_similarity, dislike_similarity):
    """
    Normalize the adjustments to keep them in scale with overall similarities.

    Args:
        post_scores (list): List of post scores.
        base_similarity (float): Base similarity score.
        like_similarity (float): Similarity score for liked posts.
        dislike_similarity (float): Similarity score for disliked posts.

    Returns:
        float: Adjusted similarity score.
    """
    if len(post_scores) == 0:
        return base_similarity

    max_score = max(post_scores, key=lambda x: x[1])[1]
    min_score = min(post_scores, key=lambda x: x[1])[1]
    score_range = max_score - min_score
    adjustment = (like_similarity - dislike_similarity) * (score_range / 2)
    return base_similarity + adjustment


def swap_random_posts(rec_post_ids, post_ids, swap_percent=0.1):
    """
    Swap a percentage of recommended posts with random posts.

    Args:
        rec_post_ids (list): List of recommended post IDs.
        post_ids (list): List of all post IDs.
        swap_percent (float): Percentage of posts to swap.

    Returns:
        list: Updated list of recommended post IDs.
    """
    num_to_swap = int(len(rec_post_ids) * swap_percent)
    posts_to_swap = random.sample(post_ids, num_to_swap)
    indices_to_replace = random.sample(range(len(rec_post_ids)), num_to_swap)

    for idx, new_post in zip(indices_to_replace, posts_to_swap):
        rec_post_ids[idx] = new_post

    return rec_post_ids


def get_trace_contents(user_id, action, post_table, trace_table):
    """
    Get the contents of posts that a user has interacted with.

    Args:
        user_id (str): ID of the user.
        action (str): Type of action (like or unlike).
        post_table (list): List of posts.
        trace_table (list): List of user interactions.

    Returns:
        list: List of post contents.
    """
    # Get post IDs from trace table for the given user and action
    trace_post_ids = [
        trace['post_id'] for trace in trace_table
        if (trace['user_id'] == user_id and trace['action'] == action)
    ]
    # Fetch post contents from post table where post IDs match those in the
    # trace
    trace_contents = [
        post['content'] for post in post_table
        if post['post_id'] in trace_post_ids
    ]
    return trace_contents


def rec_sys_personalized_with_trace(
    user_table: List[Dict[str, Any]],
    post_table: List[Dict[str, Any]],
    trace_table: List[Dict[str, Any]],
    rec_matrix: List[List],
    max_rec_post_len: int,
    swap_rate: float = 0.1,
) -> List[List]:
    """
    This version:
    1. If the number of posts is less than or equal to the maximum
        recommended length, each user gets all post IDs

    2. Otherwise:
        - For each user, get a like-trace pool and dislike-trace pool from the
            trace table
        - For each user, calculate the similarity between the user's bio and
            the post text
        - Use the trace table to adjust the similarity score
        - Swap 10% of the recommended posts with the random posts

    Personalized recommendation system that uses user interaction traces.

    Args:
        user_table (List[Dict[str, Any]]): List of users.
        post_table (List[Dict[str, Any]]): List of posts.
        trace_table (List[Dict[str, Any]]): List of user interactions.
        rec_matrix (List[List]): Existing recommendation matrix.
        max_rec_post_len (int): Maximum number of recommended posts.
        swap_rate (float): Percentage of posts to swap for diversity.

    Returns:
        List[List]: Updated recommendation matrix.
    """

    start_time = time.time()

    new_rec_matrix = []
    post_ids = [post['post_id'] for post in post_table]
    if len(post_ids) <= max_rec_post_len:
        new_rec_matrix = [post_ids] * (len(rec_matrix) - 1)
    else:
        for idx in range(1, len(rec_matrix)):
            user_id = user_table[idx - 1]['user_id']
            user_bio = user_table[idx - 1]['bio']
            # filter out posts that belong to the user
            available_post_contents = [(post['post_id'], post['content'])
                                       for post in post_table
                                       if post['user_id'] != user_id]

            # filter out like-trace and dislike-trace
            like_trace_contents = get_trace_contents(
                user_id, ActionType.LIKE_POST.value, post_table, trace_table)
            dislike_trace_contents = get_trace_contents(
                user_id, ActionType.UNLIKE_POST.value, post_table, trace_table)
            # calculate similarity between user bio and post text
            post_scores = []
            for post_id, post_content in available_post_contents:
                if model is not None:
                    user_embedding = model.encode(user_bio)
                    post_embedding = model.encode(post_content)
                    base_similarity = np.dot(
                        user_embedding,
                        post_embedding) / (np.linalg.norm(user_embedding) *
                                           np.linalg.norm(post_embedding))
                    post_scores.append((post_id, base_similarity))
                else:
                    post_scores.append((post_id, random.random()))

            new_post_scores = []
            # adjust similarity based on like and dislike traces
            for _post_id, _base_similarity in post_scores:
                _post_content = post_table[post_ids.index(_post_id)]['content']
                like_similarity = sum(
                    np.dot(model.encode(_post_content), model.encode(like)) /
                    (np.linalg.norm(model.encode(_post_content)) *
                     np.linalg.norm(model.encode(like)))
                    for like in like_trace_contents) / len(
                        like_trace_contents) if like_trace_contents else 0
                dislike_similarity = sum(
                    np.dot(model.encode(_post_content), model.encode(dislike))
                    / (np.linalg.norm(model.encode(_post_content)) *
                       np.linalg.norm(model.encode(dislike)))
                    for dislike in dislike_trace_contents) / len(
                        dislike_trace_contents
                    ) if dislike_trace_contents else 0

                # Normalize and apply adjustments
                adjusted_similarity = normalize_similarity_adjustments(
                    post_scores, _base_similarity, like_similarity,
                    dislike_similarity)
                new_post_scores.append((_post_id, adjusted_similarity))

            # sort posts by similarity
            new_post_scores.sort(key=lambda x: x[1], reverse=True)
            # extract post ids
            rec_post_ids = [
                post_id for post_id, _ in new_post_scores[:max_rec_post_len]
            ]

            if swap_rate > 0:
                # swap the recommended posts with random posts
                swap_free_ids = [
                    post_id for post_id in post_ids
                    if post_id not in rec_post_ids and post_id not in [
                        trace['post_id']
                        for trace in trace_table if trace['user_id']
                    ]
                ]
                rec_post_ids = swap_random_posts(rec_post_ids, swap_free_ids,
                                                 swap_rate)

            new_rec_matrix.append(rec_post_ids)
    end_time = time.time()
    print(f'Personalized recommendation time: {end_time - start_time:.6f}s')
    return new_rec_matrix
