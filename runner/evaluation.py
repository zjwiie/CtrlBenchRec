import json
import random

import numpy as np
import torch
from sympy.codegen.cnodes import sizeof
from sympy.integrals.meijerint_doc import category
from sklearn.metrics.pairwise import cosine_similarity

from tool.amazon_loader import AmazonLoader
from tool.embedding_processor import EmbeddingProcessor
from tool.movie_len_loader import MovieLenLoader


def target_in_all_movies_rate(target_tag_list = []):
    print("Begin:target_in_all_movies_rate,target_tag_list:", target_tag_list)
    movie_id_and_info_map = MovieLenLoader.get_movie_id_and_info_map()
    movie_genre_and_count_map = {}
    for movie_id in movie_id_and_info_map:
        movie_detail = movie_id_and_info_map[movie_id]
        movie_genres = movie_detail['genres']
        # 拆分genres
        movie_genre_list = movie_genres.split('|')
        # 统计个数
        for genre in movie_genre_list:
            if movie_genre_and_count_map.get(genre) is not None:
                movie_genre_and_count_map[genre] += 1
            else:
                movie_genre_and_count_map[genre] = 1

    for tag in target_tag_list:
        print(f"Tag: {tag},{movie_genre_and_count_map[tag]},{(movie_genre_and_count_map[tag]/len(movie_id_and_info_map)):.1%}")
    print(movie_genre_and_count_map)
    print()

def movie_with_target(target_tag_list = []):
    print("Begin:movie_with_target,target_tag_list:", target_tag_list)
    movie_id_and_info_map = MovieLenLoader.get_movie_id_and_info_map()
    movie_id_and_genres_map = {}
    genre_count = 0
    for movie_id in movie_id_and_info_map:
        movie_detail = movie_id_and_info_map[movie_id]
        movie_genres = movie_detail['genres']
        # 拆分genres
        movie_genre_list = movie_genres.split('|')

        for genre in movie_genre_list:
            if genre in target_tag_list:
                movie_id_and_genres_map[movie_id] = movie_genre_list
                genre_count = genre_count + len(movie_genre_list)

    print(movie_id_and_genres_map)
    print("avg_genre_count:",genre_count/len(movie_id_and_genres_map))
    print("movie_count:",len(movie_id_and_genres_map))
    print()

    return movie_id_and_genres_map



def target_in_profile(target_tag_list = [],profile_path = ""):
    print(f"{profile_path},begin evaluation")
    user_profile_list = []
    with open(profile_path, 'r', encoding='utf-8') as file:
        user_profile_list = json.load(file)
    movie_id_and_info_map = MovieLenLoader.get_movie_id_and_info_map()
    movie_genre_and_count_map = {}
    all_behavior_count = 0
    for user_profile in user_profile_list:
        single_user_bahavior = user_profile['behavior']
        single_user_movie_genre_and_count_map = {}
        all_behavior_count = all_behavior_count + len(single_user_bahavior)
        for behavior in single_user_bahavior:

            behavior = str(behavior)
            movie_detail = movie_id_and_info_map.get(behavior)
            movie_genres = ''
            try:
                movie_genres = movie_detail['genres']
            except:
                print("cannot load genres,movie_id:", behavior)
            # 拆分genres
            movie_genre_list = movie_genres.split('|')
            # 统计个数
            for genre in movie_genre_list:
                if movie_genre_and_count_map.get(genre) is not None:
                    movie_genre_and_count_map[genre] += 1
                else:
                    movie_genre_and_count_map[genre] = 1

                if single_user_movie_genre_and_count_map.get(genre) is not None:
                    single_user_movie_genre_and_count_map[genre] += 1
                else:
                    single_user_movie_genre_and_count_map[genre] = 1

        print(f"userName:{user_profile['userName']},behavior:{single_user_bahavior},length:{len(single_user_bahavior)},count:{single_user_movie_genre_and_count_map}")

    print(movie_genre_and_count_map)
    for tag in target_tag_list:
        print(f"Tag: {tag},{movie_genre_and_count_map[tag]},{movie_genre_and_count_map[tag]/all_behavior_count}")
    print("average behavior:", all_behavior_count / len(user_profile_list))
    print()

def compare_two_profile(profile_path1,profile_path2):
    print("Begin:compare_two_profile({},{})".format(profile_path1,profile_path2))
    profile_list1 = []
    profile_list2 = []
    with open(profile_path1, 'r', encoding='utf-8') as file:
        profile_list1 = json.load(file)
    with open(profile_path2, 'r', encoding='utf-8') as file:
        profile_list2 = json.load(file)
    # 行为流去重
    for index in range(len(profile_list2)):
        front_length = len(profile_list1[index]['behavior'])
        profile_list2[index]['behavior'] = profile_list2[index]['behavior'][front_length:]

    movie_id_and_info_map = MovieLenLoader.get_movie_id_and_info_map()
    movie_genre_and_count_map = {}
    # 对电影根据id去重后的结果
    movie_genre_and_count_map_distinct = {}
    all_behavior_count = 0
    all_movie_id_set = set()
    # 持有目标标签的所有电影的id
    movie_id_set_in_target_tag = set()
    # 点赞位于目标标签的个数
    like_num_in_target_tag = 0
    # 随机选取26个进行评估
    # raw_profile_list2 = profile_list2
    # profile_list2 = []
    # random.seed(10)
    # numbers = random.sample(range(len(raw_profile_list2)), 27)
    # for number in numbers:
    #     profile_list2.append(raw_profile_list2[number])
    for user_profile in profile_list2:
        single_user_behavior = user_profile['behavior']
        single_user_movie_genre_and_count_map = {}
        all_behavior_count = all_behavior_count + len(single_user_behavior)
        for behavior in single_user_behavior:

            behavior = str(behavior)
            movie_detail = movie_id_and_info_map.get(behavior)
            movie_genres = ''
            try:
                movie_genres = movie_detail['genres']
            except:
                print("cannot load genres,movie_id:", behavior)
            # 拆分genres
            movie_genre_list = movie_genres.split('|')
            # 统计个数
            is_in_target_tag = False
            for genre in movie_genre_list:
                if genre in target_tag_list:
                    is_in_target_tag = True
                if movie_genre_and_count_map.get(genre) is not None:
                    movie_genre_and_count_map[genre] += 1
                else:
                    movie_genre_and_count_map[genre] = 1

                if single_user_movie_genre_and_count_map.get(genre) is not None:
                    single_user_movie_genre_and_count_map[genre] += 1
                else:
                    single_user_movie_genre_and_count_map[genre] = 1

                if behavior not in all_movie_id_set:
                    if movie_genre_and_count_map_distinct.get(genre) is not None:
                        movie_genre_and_count_map_distinct[genre] += 1
                    else:
                        movie_genre_and_count_map_distinct[genre] = 1
            if is_in_target_tag:
                movie_id_set_in_target_tag.add(behavior)
                like_num_in_target_tag = like_num_in_target_tag + 1
            all_movie_id_set.add(behavior)

        # print(f"userName:{user_profile['userName']},behavior:{single_user_bahavior},length:{len(single_user_bahavior)},count:{single_user_movie_genre_and_count_map}")

    print(movie_genre_and_count_map)
    for tag in target_tag_list:
        if tag not in movie_genre_and_count_map:
            movie_genre_and_count_map[tag] = 0
        if tag not in movie_genre_and_count_map_distinct:
            movie_genre_and_count_map_distinct[tag] = 0
        print(f"Tag: {tag},{movie_genre_and_count_map[tag]}")
        # print(f"Tag: {tag},{movie_genre_and_count_map_distinct[tag]},{movie_genre_and_count_map_distinct[tag]/MovieLenLoader.get_movie_genre_and_count_map()[tag]}")
    print()
    for tag in target_tag_list:
        if tag not in movie_genre_and_count_map:
            movie_genre_and_count_map[tag] = 0
        if tag not in movie_genre_and_count_map_distinct:
            movie_genre_and_count_map_distinct[tag] = 0
        print(f"Tag: {tag},{(movie_genre_and_count_map[tag]/all_behavior_count):.1%}")
        # print(f"Tag: {tag},{movie_genre_and_count_map_distinct[tag]},{movie_genre_and_count_map_distinct[tag]/MovieLenLoader.get_movie_genre_and_count_map()[tag]}")
    print()
    for tag in target_tag_list:
        if tag not in movie_genre_and_count_map:
            movie_genre_and_count_map[tag] = 0
        if tag not in movie_genre_and_count_map_distinct:
            movie_genre_and_count_map_distinct[tag] = 0
        # print(f"Tag: {tag},{movie_genre_and_count_map[tag]},{movie_genre_and_count_map[tag]/all_behavior_count}")
        print(f"Tag: {tag},{movie_genre_and_count_map_distinct[tag]}")
    print()
    for tag in target_tag_list:
        if tag not in movie_genre_and_count_map:
            movie_genre_and_count_map[tag] = 0
        if tag not in movie_genre_and_count_map_distinct:
            movie_genre_and_count_map_distinct[tag] = 0
        # print(f"Tag: {tag},{movie_genre_and_count_map[tag]},{movie_genre_and_count_map[tag]/all_behavior_count}")
        print(f"Tag: {tag},{(movie_genre_and_count_map_distinct[tag]/MovieLenLoader.get_movie_genre_and_count_map()[tag]):.1%}")
    print()

    # 冗余度
    for tag in target_tag_list:
        if tag not in movie_genre_and_count_map:
            movie_genre_and_count_map[tag] = 0
        if tag not in movie_genre_and_count_map_distinct:
            movie_genre_and_count_map_distinct[tag] = 0
        if movie_genre_and_count_map_distinct[tag] == 0:
            continue
        print(f"Tag: {tag},{round((movie_genre_and_count_map[tag]/movie_genre_and_count_map_distinct[tag]),2)}")
    print()

    print("average behavior:", all_behavior_count / len(profile_list2))
    print()
    print("average cover rate:",len(movie_id_set_in_target_tag)/MovieLenLoader.get_movie_count_in_target_tag_list(target_tag_list))
    print()
    print("average like/all:",like_num_in_target_tag/len(movie_id_set_in_target_tag))


def compare_single_profile(profile_path2):
    print("Begin:compare_single_profile({})".format(profile_path2))
    profile_list1 = []
    profile_list2 = []
    with open(profile_path2, 'r', encoding='utf-8') as file:
        profile_list2 = json.load(file)

    movie_id_and_info_map = MovieLenLoader.get_movie_id_and_info_map()
    movie_genre_and_count_map = {}
    # 对电影根据id去重后的结果
    movie_genre_and_count_map_distinct = {}
    all_behavior_count = 0
    all_movie_id_set = set()
    # 持有目标标签的所有电影的id
    movie_id_set_in_target_tag = set()
    # 点赞位于目标标签中的个数
    like_num_in_target_tag = 0
    # 截取前28个元素
    profile_list2 = profile_list2[:26]
    for user_profile in profile_list2:
        single_user_behavior = user_profile['behavior']
        single_user_movie_genre_and_count_map = {}
        all_behavior_count = all_behavior_count + len(single_user_behavior)
        for behavior in single_user_behavior:

            behavior = str(behavior)
            movie_detail = movie_id_and_info_map.get(behavior)
            movie_genres = ''
            try:
                movie_genres = movie_detail['genres']
            except:
                print("cannot load genres,movie_id:", behavior)
            # 拆分genres
            movie_genre_list = movie_genres.split('|')
            # 统计个数
            is_in_target_tag = False
            for genre in movie_genre_list:
                if genre in target_tag_list:
                    is_in_target_tag = True
                if movie_genre_and_count_map.get(genre) is not None:
                    movie_genre_and_count_map[genre] += 1
                else:
                    movie_genre_and_count_map[genre] = 1

                if single_user_movie_genre_and_count_map.get(genre) is not None:
                    single_user_movie_genre_and_count_map[genre] += 1
                else:
                    single_user_movie_genre_and_count_map[genre] = 1

                if behavior not in all_movie_id_set:
                    if movie_genre_and_count_map_distinct.get(genre) is not None:
                        movie_genre_and_count_map_distinct[genre] += 1
                    else:
                        movie_genre_and_count_map_distinct[genre] = 1
            all_movie_id_set.add(behavior)
            if is_in_target_tag:
                movie_id_set_in_target_tag.add(behavior)
                like_num_in_target_tag = like_num_in_target_tag + 1

        # print(f"userName:{user_profile['userName']},behavior:{single_user_bahavior},length:{len(single_user_bahavior)},count:{single_user_movie_genre_and_count_map}")


    print(movie_genre_and_count_map)
    for tag in target_tag_list:
        if tag not in movie_genre_and_count_map:
            movie_genre_and_count_map[tag] = 0
        if tag not in movie_genre_and_count_map_distinct:
            movie_genre_and_count_map_distinct[tag] = 0
        print(f"Tag: {tag},{movie_genre_and_count_map[tag]}")
        # print(f"Tag: {tag},{movie_genre_and_count_map_distinct[tag]},{movie_genre_and_count_map_distinct[tag]/MovieLenLoader.get_movie_genre_and_count_map()[tag]}")
    print()
    for tag in target_tag_list:
        if tag not in movie_genre_and_count_map:
            movie_genre_and_count_map[tag] = 0
        if tag not in movie_genre_and_count_map_distinct:
            movie_genre_and_count_map_distinct[tag] = 0
        print(f"Tag: {tag},{(movie_genre_and_count_map[tag]/all_behavior_count):.1%}")
        # print(f"Tag: {tag},{movie_genre_and_count_map_distinct[tag]},{movie_genre_and_count_map_distinct[tag]/MovieLenLoader.get_movie_genre_and_count_map()[tag]}")
    print()
    for tag in target_tag_list:
        if tag not in movie_genre_and_count_map:
            movie_genre_and_count_map[tag] = 0
        if tag not in movie_genre_and_count_map_distinct:
            movie_genre_and_count_map_distinct[tag] = 0
        # print(f"Tag: {tag},{movie_genre_and_count_map[tag]},{movie_genre_and_count_map[tag]/all_behavior_count}")
        print(f"Tag: {tag},{movie_genre_and_count_map_distinct[tag]}")
    print()
    for tag in target_tag_list:
        if tag not in movie_genre_and_count_map:
            movie_genre_and_count_map[tag] = 0
        if tag not in movie_genre_and_count_map_distinct:
            movie_genre_and_count_map_distinct[tag] = 0
        # print(f"Tag: {tag},{movie_genre_and_count_map[tag]},{movie_genre_and_count_map[tag]/all_behavior_count}")
        print(f"Tag: {tag},{(movie_genre_and_count_map_distinct[tag]/MovieLenLoader.get_movie_genre_and_count_map()[tag]):.1%}")
    print()
    # 冗余度
    for tag in target_tag_list:
        if tag not in movie_genre_and_count_map:
            movie_genre_and_count_map[tag] = 0
        if tag not in movie_genre_and_count_map_distinct:
            movie_genre_and_count_map_distinct[tag] = 0
        if movie_genre_and_count_map_distinct[tag] == 0:
            continue
        print(f"Tag: {tag},{round((movie_genre_and_count_map[tag]/movie_genre_and_count_map_distinct[tag]),2)}")
    print()
    print("average behavior:", all_behavior_count / len(profile_list2))
    print()
    print("average cover rate:",len(movie_id_set_in_target_tag)/MovieLenLoader.get_movie_count_in_target_tag_list(target_tag_list))
    print()
    print("average like/all:", like_num_in_target_tag/len(movie_id_set_in_target_tag))

def evaluate_amazon(profile_path,raw_profile_path):
    profile_list = []
    raw_profile_list = []
    item_id_and_info_map = AmazonLoader.get_item_id_and_info_map()
    all_catagory_count_map = AmazonLoader.get_catagory_count_map()
    all_behavior = []
    all_behavior_count = 0
    target_behavior_count = 0
    like_num_in_target_tag = 0
    like_num_map = {}
    with open(profile_path, 'r', encoding='utf-8') as file:
        profile_list = json.load(file)
    with open(raw_profile_path, 'r', encoding='utf-8') as file:
        raw_profile_list = json.load(file)
    # # 截取前27个元素
    # profile_list = profile_list[:28]
    for index in range(len(profile_list)):
        profile = profile_list[index]
        raw_profile = raw_profile_list[index]
        behaviors = profile['behavior']
        raw_behaviors = raw_profile['behavior']
        behaviors = [item for item in behaviors if item not in raw_behaviors]
        all_behavior_count = all_behavior_count + len(behaviors)
        for behavior in behaviors:
            if behavior not in all_behavior:
                all_behavior.append(behavior)
            categories = item_id_and_info_map[behavior]['Category']
            is_in_target = 0
            for category in categories:
                if category in target_tag_list:
                    is_in_target = 1
                    if category not in like_num_map:
                        like_num_map[category] = 1
                    else :
                        like_num_map[category] = like_num_map[category] + 1
                    is_in_target = 1
            if is_in_target == 1:
                like_num_in_target_tag = like_num_in_target_tag + 1
    category_and_count_map = {}
    for behavior in all_behavior:
        categories = item_id_and_info_map[behavior]['Category']
        is_in_target = 0
        for category in categories:
            if category in target_tag_list:
                is_in_target = 1
            if category not in category_and_count_map:
                category_and_count_map[category] = 1
            else:
                category_and_count_map[category] += 1
        if is_in_target == 1:
            target_behavior_count += 1
    for target_tag in target_tag_list:
        if target_tag in category_and_count_map:
            print(f"{target_tag},{category_and_count_map[target_tag]},{category_and_count_map[target_tag] / all_catagory_count_map[target_tag]},{like_num_map[target_tag]/category_and_count_map[target_tag]}")
    print()
    print("average behavior:", all_behavior_count / len(profile_list))
    print()
    print("average cover rate:",target_behavior_count / AmazonLoader.get_count_in_target_tag_list(target_tag_list))
    print()
    print("average like/all:", like_num_in_target_tag/target_behavior_count)

def one_shot():
    target_tag_list = ['Action', 'Adventure', 'Thriller', 'War', 'Romance', 'Film-Noir']
    target_in_all_movies_rate(target_tag_list)
    movie_with_target(target_tag_list)

    compare_two_profile("../generated_user_profile/behavior_length_1/user_100_profile.json",
                        "../generated_user_profile/behavior_length_1/base/user_100_profile_0_19.json")

    compare_two_profile("../generated_user_profile/behavior_length_1/test/user_25_profile_0_0.json",
                        "../generated_user_profile/behavior_length_1/test/user_25_profile_0_19.json")

def many_shot():
    target_tag_list = ['Action', 'Adventure', 'Thriller', 'War', 'Romance', 'Film-Noir']
    target_in_all_movies_rate(target_tag_list)
    movie_with_target(target_tag_list)

    compare_two_profile("../generated_user_profile/behavior_length_12/user_100_profile.json",
                        "../generated_user_profile/behavior_length_12/base/user_100_profile_0_19.json")


    compare_two_profile("../generated_user_profile/behavior_length_12/test/user_28_profile_0_0.json",
                        "../generated_user_profile/behavior_length_12/test/user_28_profile_0_19.json")

def few_shot():
    target_tag_list = ['Action', 'Adventure', 'Thriller', 'War', 'Romance', 'Film-Noir']
    target_in_all_movies_rate(target_tag_list)
    movie_with_target(target_tag_list)

    compare_two_profile("../generated_user_profile/behavior_length_6/epoch200/user_26_profile_0_0.json",
                        "../generated_user_profile/behavior_length_6/epoch200/user_26_profile_0_19.json")

def one_many_0_1():
    # compare_single_profile("../generated_user_profile/behavior_length_1/test/user_25_profile_0_0.json")
    compare_two_profile("../generated_user_profile/behavior_length_1/test/user_25_profile_0_0.json",
                        "generated_user_profile/behavior_length_6/test/user_26_profile_0_19.json")

    # compare_single_profile("../generated_user_profile/behavior_length_12/test/user_28_profile_0_0.json")
    compare_two_profile("../generated_user_profile/behavior_length_12/test/user_28_profile_0_0.json",
                        "../generated_user_profile/behavior_length_12/test/user_28_profile_0_5.json")

def add_rec_item_test_length_12():
    compare_single_profile("../generated_user_profile/behavior_length_12/test_raw/user_28_profile.json" )
    compare_single_profile("../generated_user_profile/behavior_length_12/add_max_rec_len_test/user_28_profile_0_19.json")
    compare_two_profile("../generated_user_profile/behavior_length_12/test_raw/user_28_profile.json",
                        "../generated_user_profile/behavior_length_12/add_max_rec_len_test/user_28_profile_0_19.json")

def add_rec_item_base_length_12():
    compare_single_profile(
        "../generated_user_profile/behavior_length_12/user_100_profile.json")

    compare_single_profile("../generated_user_profile/behavior_length_12/add_max_rec_len_base/user_28_profile_0_19.json")

    compare_two_profile("../generated_user_profile/behavior_length_12/user_100_profile.json",
                            "../generated_user_profile/behavior_length_12/add_max_rec_len_base/user_28_profile_0_19.json")

def add_rec_item_base_length_12_exponential():
    compare_single_profile(
        "../generated_user_profile/behavior_length_12/user_100_profile.json")

    compare_single_profile("../generated_user_profile/behavior_length_12/add_max_rec_exponential_base/user_28_profile_0_19.json")

    compare_two_profile("../generated_user_profile/behavior_length_12/user_100_profile.json",
                            "../generated_user_profile/behavior_length_12/add_max_rec_exponential_base/user_28_profile_0_19.json")

def add_rec_item_test_length_12_exponential():
    compare_single_profile(
        "../generated_user_profile/behavior_length_12/test_raw/user_28_profile.json")

    compare_single_profile("../generated_user_profile/behavior_length_12/add_max_rec_exponential_test/user_28_profile_0_19.json")

    compare_two_profile("../generated_user_profile/behavior_length_12/test_raw/user_28_profile.json",
                            "../generated_user_profile/behavior_length_12/add_max_rec_exponential_test/user_28_profile_0_19.json")

def add_rec_item_test_50_length_12():
    compare_single_profile("../generated_user_profile/behavior_length_12/test_raw/user_100_profile_0_2.json" )
    compare_single_profile("../generated_user_profile/behavior_length_12/add_max_rec_len_test_50/user_50_profile_0_19.json")
    compare_two_profile("../generated_user_profile/behavior_length_12/test_raw/user_100_profile_0_2.json",
                        "../generated_user_profile/behavior_length_12/add_max_rec_len_test_50/user_50_profile_0_19.json")

def evaluate_each_epoch(file_prefix,max_small_epoch):
    for i in range(max_small_epoch):
        last_profile = "../generated_user_profile/behavior_length_12/test_raw/user_28_profile.json"
        now_profile = file_prefix+f"_0_{i}.json"
        compare_two_profile(last_profile, now_profile)

def add_rec_item_test_50_length_12_exponential():
    compare_single_profile(
        "../generated_user_profile/behavior_length_12/test_raw/user_100_profile_0_2.json")

    compare_single_profile("../generated_user_profile/behavior_length_12/add_max_rec_exponential_test_50/user_50_profile_0_19.json")

    compare_two_profile("../generated_user_profile/behavior_length_12/test_raw/user_100_profile_0_2.json",
                            "../generated_user_profile/behavior_length_12/add_max_rec_exponential_test_50/user_50_profile_0_19.json")

def test_50_with_3_items():
    compare_single_profile(
        "../generated_user_profile/behavior_length_12/test_raw/user_100_profile_0_2.json")

    compare_single_profile("../generated_user_profile/behavior_length_12/test_50_v1/user_50_profile_0_19.json")

    compare_two_profile("../generated_user_profile/behavior_length_12/test_raw/user_100_profile_0_2.json",
                            "../generated_user_profile/behavior_length_12/test_50_v1/user_50_profile_0_19.json")

def epoch_200_evaluation(epoch_num):
    compare_single_profile(f"../generated_user_profile/behavior_length_6/epoch200/user_26_profile_0_{epoch_num}.json")

    compare_two_profile("../generated_user_profile/behavior_length_6/test_raw/user_28_profile.json",
                        f"../generated_user_profile/behavior_length_6/epoch200/user_26_profile_0_{epoch_num}.json")

def epoch_200_base_evaluation(epoch_num):
    compare_single_profile(f"../generated_user_profile/behavior_length_6/epoch200_base/user_100_profile_0_{epoch_num}.json")

    compare_two_profile("../generated_user_profile/behavior_length_6/epoch200_base/user_100_profile.json",
                        f"../generated_user_profile/behavior_length_6/epoch200_base/user_100_profile_0_{epoch_num}.json")

def epoch_200_debate_evaluation(epoch_num):
    compare_single_profile(f"../generated_user_profile/behavior_length_6/debate/user_28_profile_0_{epoch_num}.json")

    compare_two_profile("../generated_user_profile/behavior_length_6/test_debate_raw/user_100_profile_0_2.json",
                        f"../generated_user_profile/behavior_length_6/debate/user_28_profile_0_{epoch_num}.json")


def few_shot_base_with_epoch(epoch_num):
    profile_path = f"../generated_user_profile/behavior_length_6/base-new/user_100_profile_0_{epoch_num}.json"
    compare_single_profile(profile_path)
    compare_two_profile("../generated_user_profile/behavior_length_6/user_100_profile.json",
                        profile_path)

def few_shot_test_with_epoch(epoch_num):
    profile_path = f"../generated_user_profile/behavior_length_6/test/user_26_profile_0_{epoch_num}.json"
    compare_single_profile(profile_path)
    compare_two_profile("../generated_user_profile/behavior_length_6/test_raw/user_28_profile.json",
                        profile_path)

def task3_evaluation(epoch_num):
    profile_path = f"../generated_user_profile/task2/task3/user_28_profile_0_{epoch_num}.json"
    compare_single_profile(profile_path)
    compare_two_profile("../generated_user_profile/task2/task3/user_28_profile.json",
                        profile_path)

def bge_ml1m_base_evaluation(epoch_num):
    profile_path = f"../generated_user_profile/behavior_length_6_bge/epoch30_ml1m_base/user_100_profile_0_{epoch_num}.json"
    # compare_single_profile(profile_path)
    compare_two_profile("../generated_user_profile/behavior_length_6_bge/epoch30_ml1m_base/user_100_profile.json",
                        profile_path)

def bge_ml1m_test_evaluation(epoch_num):
    profile_path = f"../generated_user_profile/behavior_length_6_bge/epoch30_ml1m_v2/user_29_profile_0_{epoch_num}.json"
    # compare_single_profile(profile_path)
    compare_two_profile("../generated_user_profile/behavior_length_6_bge/epoch30_ml1m_v2/user_29_profile.json",
                        profile_path)

def bge_amazon_base_evaluation(epoch_num):
    profile_path = f"../generated_user_profile/behavior_length_6_bge/epoch30_amazon_base/user_100_profile_0_{epoch_num}.json"
    evaluate_amazon(profile_path,"../generated_user_profile/behavior_length_6_bge/epoch30_amazon_base/user_100_profile.json")

def bge_amazon_test_evaluation(epoch_num):
    profile_path = f"../generated_user_profile/behavior_length_6_bge/epoch30_amazon/user_27_profile_0_{epoch_num}.json"
    evaluate_amazon(profile_path,"../generated_user_profile/behavior_length_6_bge/epoch30_amazon/user_27_profile.json")

def narm_ml1m_test_evaluation(epoch_num):
    profile_path = f"../generated_user_profile/behavior_length_6_narm/ml1m_debate1_epoch20/user_31_profile_0_{epoch_num}.json"
    # compare_single_profile(profile_path)
    compare_two_profile("../generated_user_profile/behavior_length_6_narm/ml1m_debate1_epoch20/user_31_profile.json",
                        profile_path)

def narm_ml1m_base_evaluation(epoch_num):
    profile_path = f"../generated_user_profile/behavior_length_6_narm/ml1m_base_epoch20/user_100_profile_0_{epoch_num}.json"
    # compare_single_profile(profile_path)
    compare_two_profile("../generated_user_profile/behavior_length_6_narm/ml1m_base_epoch20/user_100_profile.json",
                        profile_path)

def task2_l1_evaluation():
    vector_list = []
    with open("../embeddings/task2/l1_user_embedding/0.json", "r") as f:
        epoch_vector_0 = np.array(json.load(f))
    with open("../embeddings/task2/l1_user_embedding/3.json", "r") as f:
        epoch_vector_3 = np.array(json.load(f))
    with open("../embeddings/task2/l1_user_embedding/6.json", "r") as f:
        epoch_vector_6 = np.array(json.load(f))
    with open("../embeddings/task2/l1_user_embedding/9.json", "r") as f:
        epoch_vector_9 = np.array(json.load(f))
    with open("../embeddings/task2/l1_user_embedding/12.json", "r") as f:
        epoch_vector_12 = np.array(json.load(f))
    with open("../embeddings/task2/l1_user_embedding/15.json", "r") as f:
        epoch_vector_15 = np.array(json.load(f))
    with open("../embeddings/task2/l1_user_embedding/18.json", "r") as f:
        epoch_vector_18 = np.array(json.load(f))
    with open("../embeddings/task2/l1_user_embedding/20.json", "r") as f:
        epoch_vector_20 = np.array(json.load(f))
    with open("../embeddings/task2/all_post_average/action.json", "r") as f:
        action_vector = np.array(json.load(f))
    with open("../embeddings/task2/all_post_average/adventure.json", "r") as f:
        adventure_vector = np.array(json.load(f))
    with open("../embeddings/task2/all_post_average/war.json", "r") as f:
        war_vector = np.array(json.load(f))
    with open("../embeddings/task2/all_post_average/romance.json", "r") as f:
        romance_vector = np.array(json.load(f))
    with open("../embeddings/task2/all_post_average/thriller.json", "r") as f:
        thriller_vector = np.array(json.load(f))
    with open("../embeddings/task2/all_post_average/film-noir.json", "r") as f:
        film_noir_vector = np.array(json.load(f))

    # target_vector = action_vector

    vector_list.append(epoch_vector_0)
    vector_list.append(epoch_vector_3)
    vector_list.append(epoch_vector_6)
    vector_list.append(epoch_vector_9)
    vector_list.append(epoch_vector_12)
    vector_list.append(epoch_vector_15)
    vector_list.append(epoch_vector_18)
    vector_list.append(epoch_vector_20)
    vector_list.append(action_vector)
    vector_list.append(adventure_vector)
    vector_list.append(thriller_vector)
    vector_list.append(romance_vector)
    vector_list.append(war_vector)

    # 1. 得到 n*768 的矩阵
    concated_vector = np.concatenate(vector_list, axis=0)

    # 2. 计算每一行的 L2 范数（模长）
    # axis=1 表示按行计算，keepdims=True 为了保持维度方便后续广播计算
    norms = np.linalg.norm(concated_vector, axis=1, keepdims=True)

    # 3. 归一化：每个元素除以其对应行的模长
    # 归一化后的向量模长均为 1
    normalized_vector = concated_vector / norms

    # 4. 计算余弦相似度
    # 单位向量的内积即为余弦相似度
    result = normalized_vector @ normalized_vector.T

    # 打印设置
    np.set_printoptions(
        precision=3,
        suppress=True,
        linewidth=100,
        edgeitems=15,
    )
    print(result)


def task2_l2_evaluation():
    vector_list = []
    with open("../embeddings/task2/l2_user_embedding/0.json", "r") as f:
        epoch_vector_0 = np.array(json.load(f))
    with open("../embeddings/task2/l2_user_embedding/3.json", "r") as f:
        epoch_vector_3 = np.array(json.load(f))
    with open("../embeddings/task2/l2_user_embedding/6.json", "r") as f:
        epoch_vector_6 = np.array(json.load(f))
    with open("../embeddings/task2/l2_user_embedding/9.json", "r") as f:
        epoch_vector_9 = np.array(json.load(f))
    with open("../embeddings/task2/l2_user_embedding/12.json", "r") as f:
        epoch_vector_12 = np.array(json.load(f))
    with open("../embeddings/task2/l2_user_embedding/15.json", "r") as f:
        epoch_vector_15 = np.array(json.load(f))
    with open("../embeddings/task2/l2_user_embedding/18.json", "r") as f:
        epoch_vector_18 = np.array(json.load(f))
    with open("../embeddings/task2/l2_user_embedding/20.json", "r") as f:
        epoch_vector_20 = np.array(json.load(f))
    with open("../embeddings/task2/all_post_average/action.json", "r") as f:
        action_vector = np.array(json.load(f))
    with open("../embeddings/task2/all_post_average/adventure.json", "r") as f:
        adventure_vector = np.array(json.load(f))
    with open("../embeddings/task2/all_post_average/war.json", "r") as f:
        war_vector = np.array(json.load(f))
    with open("../embeddings/task2/all_post_average/romance.json", "r") as f:
        romance_vector = np.array(json.load(f))
    with open("../embeddings/task2/all_post_average/thriller.json", "r") as f:
        thriller_vector = np.array(json.load(f))
    with open("../embeddings/task2/all_post_average/film-noir.json", "r") as f:
        film_noir_vector = np.array(json.load(f))

    target_vector = (2*romance_vector+4*thriller_vector+2*war_vector+5*action_vector+3*adventure_vector)/16
    # target_vector = (romance_vector+thriller_vector+war_vector+action_vector+adventure_vector+film_noir_vector)/6

    vector_list.append(epoch_vector_0)
    vector_list.append(epoch_vector_3)
    vector_list.append(epoch_vector_6)
    vector_list.append(epoch_vector_9)
    vector_list.append(epoch_vector_12)
    vector_list.append(epoch_vector_15)
    vector_list.append(epoch_vector_18)
    vector_list.append(epoch_vector_20)
    vector_list.append(target_vector)

    # 1. 得到 n*768 的矩阵
    concated_vector = np.concatenate(vector_list, axis=0)

    # 2. 计算每一行的 L2 范数（模长）
    # axis=1 表示按行计算，keepdims=True 为了保持维度方便后续广播计算
    norms = np.linalg.norm(concated_vector, axis=1, keepdims=True)

    # 3. 归一化：每个元素除以其对应行的模长
    # 归一化后的向量模长均为 1
    normalized_vector = concated_vector / norms

    # 4. 计算余弦相似度
    # 单位向量的内积即为余弦相似度
    result = normalized_vector @ normalized_vector.T

    # 打印设置
    np.set_printoptions(
        precision=3,
        suppress=True,
        linewidth=100,
        edgeitems=10,
    )
    print(result)

def task2_l3_evaluation():
    with open("../embeddings/task2/all_post_average/action.json", "r") as f:
        action_vector = np.array(json.load(f))
    with open("../embeddings/task2/all_post_average/adventure.json", "r") as f:
        adventure_vector = np.array(json.load(f))
    with open("../embeddings/task2/all_post_average/war.json", "r") as f:
        war_vector = np.array(json.load(f))
    with open("../embeddings/task2/all_post_average/romance.json", "r") as f:
        romance_vector = np.array(json.load(f))
    with open("../embeddings/task2/all_post_average/thriller.json", "r") as f:
        thriller_vector = np.array(json.load(f))
    with open("../embeddings/task2/all_post_average/film-noir.json", "r") as f:
        film_noir_vector = np.array(json.load(f))

    with open("../embeddings/task2/l3_user_embedding/0.json", "r") as f:
        epoch_vector_map_0 = json.load(f)
    with open("../embeddings/task2/l3_user_embedding/3.json", "r") as f:
        epoch_vector_map_3 = json.load(f)
    with open("../embeddings/task2/l3_user_embedding/6.json", "r") as f:
        epoch_vector_map_6 = json.load(f)
    with open("../embeddings/task2/l3_user_embedding/9.json", "r") as f:
        epoch_vector_map_9 = json.load(f)
    with open("../embeddings/task2/l3_user_embedding/12.json", "r") as f:
        epoch_vector_map_12 = json.load(f)
    with open("../embeddings/task2/l3_user_embedding/15.json", "r") as f:
        epoch_vector_map_15 = json.load(f)
    with open("../embeddings/task2/l3_user_embedding/18.json", "r") as f:
        epoch_vector_map_18 = json.load(f)
    with open("../embeddings/task2/l3_user_embedding/20.json", "r") as f:
        epoch_vector_map_20 = json.load(f)

    for index in range(28):
        index = str(index)
        epoch_vector_0 = np.array(epoch_vector_map_0[index]).reshape(1, 768)
        epoch_vector_3 = np.array(epoch_vector_map_3[index]).reshape(1, 768)
        epoch_vector_6 = np.array(epoch_vector_map_6[index]).reshape(1, 768)
        epoch_vector_9 = np.array(epoch_vector_map_9[index]).reshape(1, 768)
        epoch_vector_12 = np.array(epoch_vector_map_12[index]).reshape(1, 768)
        epoch_vector_15 = np.array(epoch_vector_map_15[index]).reshape(1, 768)
        epoch_vector_18 = np.array(epoch_vector_map_18[index]).reshape(1, 768)
        epoch_vector_20 = np.array(epoch_vector_map_20[index]).reshape(1, 768)

        vector_list = []
        vector_list.append(epoch_vector_0)
        vector_list.append(epoch_vector_3)
        vector_list.append(epoch_vector_6)
        vector_list.append(epoch_vector_9)
        vector_list.append(epoch_vector_12)
        vector_list.append(epoch_vector_15)
        vector_list.append(epoch_vector_18)
        vector_list.append(epoch_vector_20)
        vector_list.append(action_vector)
        vector_list.append(adventure_vector)
        vector_list.append(thriller_vector)
        vector_list.append(war_vector)
        vector_list.append(romance_vector)
        vector_list.append(film_noir_vector)

        # 1. 得到 n*768 的矩阵
        concated_vector = np.concatenate(vector_list, axis=0)

        # 2. 计算每一行的 L2 范数（模长）
        # axis=1 表示按行计算，keepdims=True 为了保持维度方便后续广播计算
        norms = np.linalg.norm(concated_vector, axis=1, keepdims=True)

        # 3. 归一化：每个元素除以其对应行的模长
        # 归一化后的向量模长均为 1
        normalized_vector = concated_vector / norms

        # 4. 计算余弦相似度
        # 单位向量的内积即为余弦相似度
        result = normalized_vector @ normalized_vector.T

        # 打印设置
        np.set_printoptions(
            precision=3,
            suppress=True,
            linewidth=100,
            edgeitems=10,
        )
        print(f"index: {index}")
        print("0       3     6     9     12    15    18    20    ac    ad    th    wa    ro   fn")
        print(result)
        print()



if __name__ == '__main__':
    target_tag_list = ['Action', 'Adventure', 'Thriller', 'War', 'Romance', 'Film-Noir']
    # target_tag_list= ["Action Figures & Statues","Arts & Crafts","Puzzles","Drawing & Painting Supplies","Science"]
    compare_two_profile("../generated_user_profile/behavior_length_6_sasrec/behavior_length_6_sasrec/ml1m_debate_epoch20/user_27_profile.json",
                        "../generated_user_profile/behavior_length_6_sasrec/behavior_length_6_sasrec/ml1m_debate_epoch20/user_27_profile_0_4.json")

    compare_two_profile("../generated_user_profile/behavior_length_6_sasrec/behavior_length_6_sasrec/ml1m_debate_epoch20/user_27_profile.json",
                        "../generated_user_profile/behavior_length_6_sasrec/behavior_length_6_sasrec/ml1m_debate_epoch20/user_27_profile_0_9.json")

    compare_two_profile("../generated_user_profile/behavior_length_6_sasrec/behavior_length_6_sasrec/ml1m_debate_epoch20/user_27_profile.json",
                        "../generated_user_profile/behavior_length_6_sasrec/behavior_length_6_sasrec/ml1m_debate_epoch20/user_27_profile_0_14.json")

    compare_two_profile("../generated_user_profile/behavior_length_6_sasrec/behavior_length_6_sasrec/ml1m_debate_epoch20/user_27_profile.json",
                        "../generated_user_profile/behavior_length_6_sasrec/behavior_length_6_sasrec/ml1m_debate_epoch20/user_27_profile_0_19.json")

    # compare_two_profile(
    #     "../generated_user_profile/behavior_length_6_sasrec/behavior_length_6_sasrec/ml1m_base_epoch20/user_100_profile.json",
    #     "../generated_user_profile/behavior_length_6_sasrec/behavior_length_6_sasrec/ml1m_base_epoch20/user_100_profile_0_4.json")
    #
    # compare_two_profile(
    #     "../generated_user_profile/behavior_length_6_sasrec/behavior_length_6_sasrec/ml1m_base_epoch20/user_100_profile.json",
    #     "../generated_user_profile/behavior_length_6_sasrec/behavior_length_6_sasrec/ml1m_base_epoch20/user_100_profile_0_9.json")
    #
    # compare_two_profile(
    #     "../generated_user_profile/behavior_length_6_sasrec/behavior_length_6_sasrec/ml1m_base_epoch20/user_100_profile.json",
    #     "../generated_user_profile/behavior_length_6_sasrec/behavior_length_6_sasrec/ml1m_base_epoch20/user_100_profile_0_14.json")
    #
    # compare_two_profile(
    #     "../generated_user_profile/behavior_length_6_sasrec/behavior_length_6_sasrec/ml1m_base_epoch20/user_100_profile.json",
    #     "../generated_user_profile/behavior_length_6_sasrec/behavior_length_6_sasrec/ml1m_base_epoch20/user_100_profile_0_19.json")
    #
