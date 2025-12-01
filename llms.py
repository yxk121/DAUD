from openai import OpenAI
from openai import AzureOpenAI
import openai
from tqdm import tqdm
import random
import itertools
import math
import os
from openai import AzureOpenAI

random.seed(42)

client_openai = OpenAI(
        api_key='api_key'
    )
client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://llmsagent.openai.azure.com/",
    api_key="api_key",
)
model = "gpt-4.1-mini"


def comment_generation(id_content_list, gender, age, education):
    """
    System Prompt: Suppose you are a [gender] Twitter user. You are [age] .
                   Educationally, you [education]. You will be provided with an article.
                   You should write one comment about the article.
                   Note that your comment needs to match your identity,
                   and should be brief and natural, like normal Twitter users.
    Context Prompt: news: [the given news 𝑜].
    """
    system_message = "Suppose you are a " + gender + " Twitter user. You are " + age + ". " \
                     "Educationally, you " + education + ". You will be provided with an article. " \
                     "You should write one comment about the article. " \
                     "Note that your comment needs to match your identity, " \
                     "and should be brief and natural, like normal Twitter users."
    generated_comment_dict = {}
    print("")
    print("Generating Comments...")
    for i, content in tqdm(id_content_list):
        input_text = "news: " + content
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_text}
            ]
        )
        generated_comment = response.choices[0].message.content
        generated_comment_dict[i] = generated_comment

    return generated_comment_dict


def stance_classification(id_content_list, comment_dict):
    """
    FOR SITUATION WHERE NEWS : COMMENT = 1 : 1
    id_content_list: [(news_id, content)]
    comment_dict: {news_id: comment}
    ------
    System Prompt: You are an advanced language model trained to analyze the stance
                   (Agree / Disagree / Neutral) of text.
                   Your task is to determine the stance of a given comment
                   in relation to a provided news article.

    Context Prompt: news article: [the given news 𝑜];
                    comment: [the given comment 𝑐].
    """
    system_message = "You are an advanced language model trained to analyze the stance " \
                     "(Agree / Disagree / Neutral) of text. " \
                     "Your task is to determine the stance of a given comment " \
                     "in relation to a provided news article."

    stance_dict = {} 
    print("")
    print("Classifying Stances...")
    for i, content in tqdm(id_content_list):
        input_text = "news article: " + content + "; comment: " + comment_dict[i]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_text}
            ]
        )
        stance_analysis = response.choices[0].message.content
        stance_dict[i] = stance_analysis

    return stance_dict


def stance_classification_n(selected_news_list, news_comment_dict, news_dict, comments_dict):
    """
    FOR SITUATION WHERE NEWS : COMMENT = 1 : N (or 1 : 1)
    id_content_list: [(news_id, content)]
    comment_dict: {news_id: comment}
    ------
    System Prompt: You are an advanced language model trained to analyze the stance
                   (Agree / Disagree / Neutral) of text.
                   Your task is to determine the stance of a given comment
                   in relation to a provided news article.

    Context Prompt: news article: [the given news 𝑜];
                    comment: [the given comment 𝑐].
    """
    system_message = "You are an advanced language model trained to analyze the stance " \
                     "(Agree / Disagree / Neutral) of text. " \
                     "Your task is to determine the stance of a given comment " \
                     "in relation to a provided news article."

    stance_dict = {} 
    print("")
    print("Classifying Stances...")
    for news_id in tqdm(selected_news_list):
        news_content = news_dict[news_id]
        comment_ids_list = news_comment_dict[news_id]
        print("")
        print("Processing Comments of the News...")
        for comment_id in tqdm(comment_ids_list):
            comment = comments_dict[comment_id].split("###")[0]
            input_text = "news article: " + news_content + "; comment: " + comment

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": input_text}
                    ]
                )
            except openai.BadRequestError as e:
                print("\n=== Azure content filter triggered ===")
                response = client_openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": input_text}
                    ]
                )
            try:
                stance_analysis = response.choices[0].message.content
            except (AttributeError, KeyError, IndexError) as e:
                if hasattr(response, "choices") and response.choices:
                    filter_results = getattr(response.choices[0], "content_filter_results", {})
                    is_filtered = any(v.get("filtered", False) for v in filter_results.values())
                else:
                    filter_result = getattr(
                        getattr(response, "error", None),
                        "innererror",
                        {}
                    ).get("content_filter_result", {})
                    is_filtered = any(v.get("filtered", False) for v in filter_result.values())
                if is_filtered:
                    print("\n=== Azure content filter triggered ===")
                    response_openai = client_openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": input_text}
                        ]
                    )
                    stance_analysis = response_openai.choices[0].message.content
                else:
                    raise e
            stance_dict[comment_id] = stance_analysis
        print("=====================================")
    return stance_dict


def stance_result_counting(stance_dict):
    agree_count, disagree_count, neutral_count = 0, 0, 0
    stance_label_dict = {}
    for i, stance in stance_dict.items():
        stance = stance.lower()
        labels = ["agree", "disagree", "neutral"]
        earliest_word = min(labels, key=lambda word: stance.find(word) if stance.find(word) != -1 else float('inf'))

        if earliest_word == "disagree":
            disagree_count += 1
            stance_label_dict[i] = "disagree"
        elif earliest_word == "agree":
            agree_count += 1
            stance_label_dict[i] = "agree"
        elif earliest_word == "neutral":
            neutral_count += 1
            stance_label_dict[i] = "neutral"
        else:
            print("Error: Not Found!")

    return agree_count, disagree_count, neutral_count, stance_label_dict


def generate_comments_2_users(news_list, news_dict):
    news_content_list = []
    for nid in news_list:
        content = news_dict[nid]
        news_content_list.append([nid, content])

    gender_pool = ["male", "female"]
    age_pool = [" under 17 years old", " 18 to 29 years old", " 30 to 49 years old",
                " 50 to 64 years old", " over 65 years old"]
    education_pool = ["a college graduate", "has not graduated from college", "has a high school diploma or less"]
    gender_1, age_1, education_1 = random.choice(gender_pool), random.choice(age_pool), random.choice(education_pool)
    generated_1 = comment_generation(news_content_list, gender_1, age_1, education_1)

    gender_2, age_2, education_2 = random.choice(gender_pool), random.choice(age_pool), random.choice(education_pool)
    generated_2 = comment_generation(news_content_list, gender_2, age_2, education_2)

    generated = {}
    for nid in news_list:
        generated[nid] = generated_1[nid] + "###" + generated_2[nid]
    return generated


def generate_n_comments(news_list, news_dict, n):
    user_number = n // len(news_list)  # per news
    extra_user = n - user_number * len(news_list)
    print("")
    print("{} comments are needed for {} news acticles -> {} users per news, {} extra users".format(n, len(news_list),
                                                                                                user_number, extra_user))

    news_content_list = []
    for nid in news_list:
        content = news_dict[nid]
        news_content_list.append([nid, content])

    gender_pool = ["male", "female"]
    age_pool = [" under 17 years old", " 18 to 29 years old", " 30 to 49 years old",
                " 50 to 64 years old", " over 65 years old"]
    education_pool = ["a college graduate", "has not graduated from college", "has a high school diploma or less"]
    generated_comment_dict = {}
    for user in range(user_number):
        gender, age, education = random.choice(gender_pool), random.choice(age_pool), random.choice(education_pool)
        generated_comment = comment_generation(news_content_list, gender, age, education)

        for nid in news_list:
            if nid not in generated_comment_dict:
                generated_comment_dict[nid] = generated_comment[nid]
            else:
                generated_comment_dict[nid] += "###" + generated_comment[nid]

    if extra_user > 0:
        extra_news_list = random.sample(news_list, extra_user)
        extra_news_content_list = []
        for nid in extra_news_list:
            content = news_dict[nid]
            extra_news_content_list.append([nid, content])

        extra_gender, extra_age, extra_education = (random.choice(gender_pool), random.choice(age_pool),
                                                    random.choice(education_pool))
        extra_generated_comment = comment_generation(extra_news_content_list, extra_gender, extra_age, extra_education)

        for nid in extra_news_list:
            if nid not in generated_comment_dict:
                generated_comment_dict[nid] = extra_generated_comment[nid]
            else:
                generated_comment_dict[nid] += "###" + extra_generated_comment[nid]

    return generated_comment_dict


def generate_comments(news_dict, sim_engagements_dict):
    user_ids = list(sim_engagements_dict.keys())

    gender_pool = ["male", "female"]
    age_pool = [" under 17 years old", " 18 to 29 years old", " 30 to 49 years old",
                " 50 to 64 years old", " over 65 years old"]
    education_pool = ["a college graduate", "has not graduated from college", "has a high school diploma or less"]

    all_combinations = list(itertools.product(gender_pool, age_pool, education_pool))

    num_users = len(user_ids)
    num_combinations = len(all_combinations)
    repeat_times = (num_users + num_combinations - 1) // num_combinations 

    extended_profiles = all_combinations * repeat_times
    random.shuffle(extended_profiles)

    user_profiles_dict = {}
    generated_comments_dict = {}
    for user_id, profile in zip(user_ids, extended_profiles):
        user_profiles_dict[user_id] = {
            "gender": profile[0],
            "age": profile[1],
            "education": profile[2]
        }

        engaged_news_list = (list(sim_engagements_dict[user_id]["politics"].keys()) +
                             list(sim_engagements_dict[user_id]["entertainment"].keys()))

        engaged_news_content_list = []
        for nid in engaged_news_list:
            content = news_dict[nid]
            engaged_news_content_list.append([nid, content])

        generated_comment = comment_generation(engaged_news_content_list, profile[0], profile[1], profile[2])

        for nid in engaged_news_list:
            if nid not in generated_comments_dict:
                generated_comments_dict[nid] = generated_comment[nid]
            else:
                generated_comments_dict[nid] += "###" + generated_comment[nid]

    return user_profiles_dict, generated_comments_dict
