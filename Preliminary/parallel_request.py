# imports
import aiohttp  # for making API calls concurrently
import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import re  # for matching endpoint from request URL
import threading
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
from dataclasses import (
    dataclass,
    field,
)  # for storing API inputs, outputs, and metadata
from toolkits import save_ndjson, read_ndjson
from sklearn.metrics import f1_score, precision_score, recall_score
from openai import OpenAI
import traceback

OPENAI_API_KEY = "OPENAI_API_KEY"


async def call_openai_api(prompt, session, request_url, request_header, user_no, fallback_to_openai=True):
    request_json = {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    async with session.post(url=request_url, headers=request_header, json=request_json) as response:
        try:
            response_json = await response.json()
            try:
                output = response_json["choices"][0]["message"]["content"]
            except KeyError:
                if response_json is None:
                    raise Exception("Response JSON is None.")
                elif "choices" in response_json:
                    filter_results = response_json["choices"][0]["content_filter_results"]
                    is_filtered = any(v["filtered"] for v in filter_results.values())
                else:
                    filter_result = response_json["error"]["innererror"]["content_filter_result"]
                    is_filtered = any(v["filtered"] for v in filter_result.values())
                if is_filtered:
                    print("=== Azure content filter triggered ===")
                    if fallback_to_openai:
                        print(f"User #{user_no} Falling back to OpenAI...")

                        openai_url = "https://api.openai.com/v1/chat/completions"
                        openai_header = {
                            "Authorization": f"Bearer {OPENAI_API_KEY}",
                            "Content-Type": "application/json"
                        }
                        openai_request_json = {
                            "model": "gpt-4o-mini",
                            "messages": [{"role": "user", "content": prompt}]
                        }
                        async with session.post(url=openai_url,
                                                headers=openai_header, json=openai_request_json) as openai_response:
                            openai_response_json = await openai_response.json()
                            openai_output = openai_response_json["choices"][0]["message"]["content"]
                            print(f"User #{user_no} OpenAI output generated successfully.")
                            return openai_output
                    else:
                        raise Exception("Blocked by Azure content filter, and fallback_to_openai is False.")
        except Exception as e:
            raise Exception(f"[Azure/OpenAI API Error] Failed to generate results: {type(e).__name__}: {e}") from e
        return output


def extract_action_and_explanation(text):
    match = re.search(
        r"Chosen Action:\s*(.*?)\s*Explanation:\s*(.*)",
        text.strip(),
        re.DOTALL | re.IGNORECASE
    )
    if not match:
        pattern_md = r"\*{0,2}Chosen Action\*{0,2}:\s*(.*?)\s*\*{0,2}Explanation\*{0,2}:\s*(.*)"
        match = re.search(pattern_md, text.strip(), re.DOTALL | re.IGNORECASE)
    if match:
        chosen_action = match.group(1).strip()
        explanation = match.group(2).strip()
        return chosen_action, explanation
    else:
        raise ValueError("\n Unable to extract 'Chosen Action' and 'Explanation' from the text. Input text was:\n{}".format(text))


def news_features_prompt(news_article):
    news_prompt = f"""
    Recently, the user browsed a news article, its article content is: "{news_article}". Your task is to analyze the 
    content of the news text, and then summarize the characteristics of true and fake information within the news.\n
    Follow these steps:\n
    1. Identify and ignore parts of the text that are website noise, such as ads, image captions, or unrelated links.
    2. Analyze the news text based on: News Domain  (e.g., political, entertainment, medical), Sentiment (e.g., neutral, 
    emotional, exaggerated), Structural Features (e.g., Fake news may lack a proper news structure), Factual 
    Plausibility (i.e., logical sense within the text), Source Credibility (If names, quotes, or numbers are used, check 
    whether they are backed by clear sources; Be suspicious if the article refers to real people but provides no 
    official link or reference).\n
    Important Notes:\n
    1. Fake news can still look professional. Focus not just on style, but also on whether claims can be confirmed or 
    are just made to sound real.
    2. Your output should be in the following format: "The description of the news is [news features description]." Each 
    news description cannot exceed 180 words!!! 
    3. Be specific and base the description on facts. Any features that cannot distinguish this news from others are not 
    worth recording.
    """
    return news_prompt


def forward_user_agent_prompt(user_agent_memo, news_article, news_features_desc):
    forward_user_prompt = f"""
    You are simulating the behavior of a Twitter (X) user, who is either a regular user, a debunking user, or a malicious user (e.g., spammer or troll).
Here is your self-introduction about what type of user you are as well as the types and characteristics of news you like or dislike engaging with: "{user_agent_memo}". 
Now, by simulating this user’s behavior, you are considering whether to repost the following news: "{news_article}", its news features are as follows: "{news_features_desc}". 
Follow these steps:
1. Identify user type and news preferences (including both likes and dislikes) based on your self-introduction.
2. Assess whether the news aligns with your user type and preferences. Consider its content and features.
3. Decide whether to Repost or Ignore the news, and explain your reasoning based on the match or mismatch between your user type as well as your preferences (including both likes and dislikes) and the news.
Important note:
1. Your output should be in the format: "Chosen Action: [Repost or Ignore] 
Explanation: [Detailed reasoning based on the match between your user type/preferences and the news content/features]."
2. Do not fabricate preferences. If your self-introduction lacks specifics, assume a plausible user type and judge based on its typical response to the news characteristics (e.g., popularity, relevance, factual plausibility, and source credibility).
3. Your explanation must be comprehensive and specific. A general preference, such as a certain genre of news or “I like politics”, is insufficient. 
4. Highlight how the truthfulness or misleading nature of the news relates to your user type (e.g., malicious users may favor false claims).
5. Base your explanation on facts and evidence in the news, not assumptions about your user type.
    """
    return forward_user_prompt


def backward_user_agent_prompt(user_agent_memo, news_article, news_features_desc, user_explanation):
    backward_user_prompt = f"""
    You are simulating the behavior of a Twitter (X) user, who is either a regular user, a debunking user, or a malicious user (e.g., spammer or troll).
Here is your current self-introduction describing your user type and the types and characteristics of news you like or dislike engaging with: "{user_agent_memo}". 
Recently, you predicted that this user would ignore the following news article: "{news_article}"; 
Its news features are: "{news_features_desc}"; 
Your explanation was: "{user_explanation}".
However, the user actually reposted the news article. This indicates that your self-introduction may be inaccurate, incomplete, or missing a key motivational factor that caused this action.

Your task is to revise your self-introduction so that it can explain the reposting behavior naturally and accurately.
To do so, follow these steps:
1. Identify what you overlooked or misunderstood about the news article that led to the repost.
2. Analyze what specific features of the article (e.g., emotional framing, public figure, donation amount) may have motivated the user to repost it.
3. Consider how your user type or value system may need to be updated to reflect this motivation.
4. Decide which past preferences should be retained, revised, or discarded to avoid future contradiction.
5. Write an updated self-introduction that: Starts with your new user type; Describes your newfound preferences reflected in this interaction; Summarizes any relevant retained preferences; Describes what types of news you now dislike.
Important note:
1. Your output should use the following format: "My updated self-introduction: [Please write your revised self-introduction here]." Do not exceed 250 words.
2. Do not include any summarization or explanation of the update process.
3. Be specific and personalized. Preferences and dislikes that are too generic are not worth recording.

    """
    return backward_user_prompt


def user_introduction_prompt(news_article, news_features_desc):
    user_introduction_prompt = f"""
    You are simulating the behavior of a Twitter (X) user, who is either a regular user, a debunking user, or a malicious user (e.g., spammer or troll).
Recently, you chose to repost the following news article: "{news_article}", whose features are as follows: "{news_features_desc}".
Your task now is to generate the self-introduction based solely on this interaction.
Follow these steps:
1. Determine your new user type, preferences, and dislikes based on the news article content and features.
2. Please start by describing your new determined user type. Then describe your preferences reflected in this interaction. Afterward, please describe your dislikes.
Important note:
1. Your output should use the following format: "My self-introduction: [Please write your self-introduction here]." Do not exceed 250 words.
2. Any overall assessments or summarization in your self-introduction are forbidden.
3. Only describe your user type, preferred types and characteristics of news, and news you dislike.
4. Your self-introduction should be specific and personalized. Any preferences and dislikes that cannot distinguish you from others are not worth recording.
    """
    return user_introduction_prompt


def comment_features_prompt(comments_list_str):
    comment_features_prompt = f"""
Recently, the user has posted multiple comments on different news articles, the comments are listed as follows: "{comments_list_str}". Your task is to analyze these comments and summarize the user’s typical tone and commenting style.
Follow these steps:
1. Identify and ignore parts of the comments that are formulaic (e.g., hashtags, URLs, emojis, or repost tags) unless they contribute meaningfully to the tone.
2. Analyze the comments based on: Tone (e.g., sarcastic, sincere, angry, humorous, accusatory, supportive); Intent (e.g., inform, provoke, agree, debunk, mock); Linguistic Style (e.g., formal, casual, concise, emotionally charged, rhetorical); Stance Consistency (e.g., does the user consistently take a certain side or shift based on domain or content); Targeting Pattern (e.g., does the user address individuals, institutions, abstract ideas, or the public).
Important Notes:
1. Avoid generic stylistic summaries. Your description must reflect features that distinguish this user from others.
2. Your output should be in the following format: "The comment style of this user is [comment features description]." Each comment features description must not exceed 100 words.
3. Base your analysis strictly on the user's actual comments. Do not infer user intentions beyond the evidence.
    """
    return comment_features_prompt


def comment_generation_prompt(user_agent_memo, news_article, news_features_desc, comment_features_desc):
    comment_generation_prompt = f"""
    You are simulating the behavior of a Twitter (X) user.
Here is your current self-introduction describing your user type and the types and characteristics of news you like or dislike engaging with: "{user_agent_memo}".
Recently, this user repost the following news article: "{news_article}"; 
Its news features are: "{news_features_desc}"; 
Your task now is to generate a public comment on this news article, based on the user’s behavior and preferences. The desired tone and style of your comment are described as: "{comment_features_desc}".
Follow these steps:
1. Review the news article and its features. Consider the News Domain, Sentiment, Structure, Factual Plausibility, and Source Credibility.
2. Consider the user’s self-introduction. Identify what aspects of the article align with their interests, values, or intentions.
3. Generate a comment that reflects the user’s tone and typical posting style, as described in comment features. The comment should express the user’s reaction, opinion, or purpose in reposting this article.
Important note:
1. Your output should use the following format: "User Comment is [Please write your generated comment here]." Do not exceed 25 words.
2. Do not include explanations of the comment.
3. The comment should reflect both the content of the news and the user’s personality, preferences, and motives.
4. Avoid vague or generic remarks. Make the comment specific, personalized, and consistent with the user's established behavior.

    """
    return comment_generation_prompt

# The implementation is inspired by the code from:
# https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py


async def process_api_requests_from_file(
    save_path: str,
    selected_user_num: int,
    input_dicts: list,
    mode: str,

    log_save_filepath: str,
    request_url: str,
    api_key: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int,
    logging_level: int,
):
    """Processes API requests for user agents in parallel, throttling to stay under rate limits."""
    news_features, user_agent, comment_features, comment_generation = False, False, False, False
    if mode == "user_agent_train" or mode == "user_agent_test":
        u_at_dict, news_id_idx_dict, news_content_dict = input_dicts
        user_agent = True
        user_ids_list = list(u_at_dict.keys())
    elif mode == "comment_features":
        u_at_dict, news_id_idx_dict, news_content_dict, comments_dict = input_dicts
        comment_features = True
        user_ids_list = list(u_at_dict.keys())
    elif mode == "comment_generation":
        u_at_dict, news_id_idx_dict, news_content_dict, similar_news_dict = input_dicts
        comment_generation = True
        user_ids_list = list(u_at_dict.keys())
    elif mode == "news_features":
        news_id_idx_dict, news_content_dict = input_dicts
        news_features = True
        news_ids_list = list(news_content_dict.keys())
    else:
        raise ValueError(f"Invalid mode: {mode}. Expected 'user_agent_train', 'user_agent_test', "
                         f"'comment_features', or 'comment_generation'.")


    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = (
        0.001  
    )

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    request_header = {
        "api-key": api_key,  
        "Content-Type": "application/json"
    }

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    user_no_generator = (
        user_no_generator_function()
    )  # generates integer IDs of 0, 1, 2, ...
    status_tracker = (
        StatusTracker()
    )  # single instance to track a collection of variables
    next_user = None  # variable to hold the next user to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    users_not_finished = True  # after users is empty, we'll skip iterating it
    logging.debug(f"Initialization complete.")

    # `user_ids` will provide user_ids one at a time; so will `news_ids`
    if news_features:
        news_ids = news_ids_list.__iter__()
    else:
        user_ids = user_ids_list.__iter__()
    logging.debug(f"Loading started. Entering main loop")
    async with aiohttp.ClientSession() as session:  # Initialize ClientSession here
        while True:
            # get next request (if one is not already waiting for capacity)
            if next_user is None:
                if not queue_of_requests_to_retry.empty():
                    next_user = queue_of_requests_to_retry.get_nowait()
                    logging.debug(
                        f"Retrying user {next_user.user_no}: {next_user}"
                    )
                elif users_not_finished:
                    try:
                        # get user_id for new request
                        if news_features:
                            news_id = next(news_ids)  
                            news_content = news_content_dict[news_id].replace("\n", " ")
                            next_user = APIRequest(  
                                user_no=next(user_no_generator),
                                user_id=news_id,  
                                input_dicts=input_dicts,
                                token_consumption=longest_prompt_of_news_features(
                                    news_content,
                                    token_encoding_name,
                                ),
                                attempts_left=max_attempts,
                                save_path=save_path,
                                selected_user_num=selected_user_num,
                                mode=mode,
                            )
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            logging.debug(
                                f"Loading news {next_user.user_no}: {next_user}"
                            )

                        if user_agent:
                            user_id = next(user_ids)
                            engagements_idx = [news_id_idx_dict[engagement[0]] for engagement in u_at_dict[user_id]]
                            engagements = [[idx, news_content_dict[idx].replace("\n", " ")] for idx in engagements_idx]

                            next_user = APIRequest(
                                user_no=next(user_no_generator),
                                user_id=user_id,
                                input_dicts=input_dicts,
                                token_consumption=longest_prompt_of_user(
                                    engagements,
                                    token_encoding_name,
                                ),
                                attempts_left=max_attempts,
                                save_path=save_path,
                                selected_user_num=selected_user_num,
                                mode=mode,
                            )
                            status_tracker.num_tasks_started += len(engagements)
                            status_tracker.num_tasks_in_progress += len(engagements)
                            logging.debug(
                                f"Loading user {next_user.user_no}: {next_user}"
                            )

                        if comment_features:
                            user_id = next(user_ids) 
                            comments_list = []
                            for i, engagement in enumerate(u_at_dict[user_id]):
                                tweet_id = engagement[1]
                                comment_content = comments_dict.get(tweet_id, None)
                                if comment_content:
                                    try:
                                        comment_content = comment_content.split("###")[0]
                                    except Exception as e:
                                        comment_content = comment_content
                                    comments_list.append(f"{i + 1}. {comment_content};")
                            comments_text = " ".join(comments_list)

                            next_user = APIRequest(
                                user_no=next(user_no_generator),
                                user_id=user_id,
                                input_dicts=input_dicts,
                                token_consumption=longest_prompt_of_comment_features(
                                    comments_text,
                                    token_encoding_name,
                                ),
                                attempts_left=max_attempts,
                                save_path=save_path,
                                selected_user_num=selected_user_num,
                                mode=mode,
                            )
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            logging.debug(
                                f"Loading user {next_user.user_no}: {next_user}"
                            )

                        if comment_generation:
                            user_id = next(user_ids) 
                            try:
                                engagements_tobe = [[idx, news_content_dict[idx].replace("\n", " ")]
                                                     for idx in similar_news_dict[user_id]]
                            except KeyError:
                                engagements_tobe = []
                            next_user = APIRequest(
                                user_no=next(user_no_generator),
                                user_id=user_id,
                                input_dicts=input_dicts,
                                token_consumption=longest_prompt_of_comment_generation(
                                    engagements_tobe,
                                    token_encoding_name,
                                ),
                                attempts_left=max_attempts,
                                save_path=save_path,
                                selected_user_num=selected_user_num,
                                mode=mode,
                            )
                            status_tracker.num_tasks_started += len(engagements_tobe)
                            status_tracker.num_tasks_in_progress += len(engagements_tobe)
                            logging.debug(
                                f"Loading user {next_user.user_no}: {next_user}"
                            )

                    except StopIteration:
                        logging.debug("Load user exhausted")
                        users_not_finished = False

            # update available capacity
            current_time = time.time()
            seconds_since_update = current_time - last_update_time
            available_request_capacity = min(
                available_request_capacity
                + max_requests_per_minute * seconds_since_update / 60.0,
                max_requests_per_minute,
            )
            available_token_capacity = min(
                available_token_capacity
                + max_tokens_per_minute * seconds_since_update / 60.0,
                max_tokens_per_minute,
            )
            last_update_time = current_time

            # if enough capacity available, call API
            if next_user:
                next_user_tokens = next_user.token_consumption
                if (
                    available_request_capacity >= 1
                    and available_token_capacity >= next_user_tokens
                ):
                    if mode == "user_agent_train":
                        available_request_capacity -= len(engagements) * 6 
                    elif mode == "user_agent_test":
                        available_request_capacity -= len(engagements) * 1
                    elif mode == "comment_generation":
                        available_request_capacity -= len(engagements_tobe) * 1
                    else:
                        available_request_capacity -= 1
                    available_token_capacity -= next_user_tokens
                    next_user.attempts_left -= 1

                    # call API
                    asyncio.create_task(
                        next_user.call_api(
                            session=session,
                            request_url=request_url,
                            request_header=request_header,
                            retry_queue=queue_of_requests_to_retry,
                            log_save_filepath=log_save_filepath,
                            status_tracker=status_tracker,
                        )
                    )
                    next_user = None  

            print(f"Current status: {status_tracker.num_tasks_in_progress} tasks in progress, ")
            # if all tasks are finished, break
            if status_tracker.num_tasks_in_progress == 0:
                break

            # main loop sleeps briefly so concurrent tasks can run
            await asyncio.sleep(seconds_to_sleep_each_loop)

            # if a rate limit error was hit recently, pause to cool down
            seconds_since_rate_limit_error = (
                time.time() - status_tracker.time_of_last_rate_limit_error
            )
            if (
                seconds_since_rate_limit_error
                < seconds_to_pause_after_rate_limit_error
            ):
                remaining_seconds_to_pause = (
                    seconds_to_pause_after_rate_limit_error
                    - seconds_since_rate_limit_error
                )
                await asyncio.sleep(remaining_seconds_to_pause)
                # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                logging.warn(
                    f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
                )
    # after finishing, log final status
    logging.info(
        f"""Parallel processing complete. Results saved to {log_save_filepath}"""
    )
    if status_tracker.num_tasks_failed > 0:
        logging.warning(
            f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. "
            f"Errors logged to {log_save_filepath}."
        )
    if status_tracker.num_rate_limit_errors > 0:
        logging.warning(
            f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
        )


# dataclasses
news_features_lock = threading.Lock()
comment_features_lock = threading.Lock()
comment_generation_lock = threading.Lock()


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  


@dataclass
class APIRequest:
    """Stores inputs, outputs of a series of API requests for each user.
    Contains a method to make a loop of API calls."""

    user_no: int
    user_id: str
    input_dicts: list
    save_path: str
    selected_user_num: int
    mode: str

    token_consumption: int
    attempts_left: int
    result: list = field(default_factory=list)

    async def call_api(
            self,
            session: aiohttp.ClientSession,
            request_url: str,
            request_header: dict,
            retry_queue: asyncio.Queue,
            log_save_filepath: str,
            status_tracker: StatusTracker,
    ):
        if self.mode == "user_agent_train" or self.mode == "user_agent_test":
            u_at_dict, news_id_idx_dict, news_content_dict = self.input_dicts
            engagements_idx = [news_id_idx_dict[engagement[0]] for engagement in u_at_dict[self.user_id]]
            engagements = [[idx, news_content_dict[idx].replace("\n", " ")] for idx in engagements_idx]

            if self.mode == "user_agent_train":
                await self._call_api_train(
                    engagements, news_content_dict,
                    session, request_url, request_header, retry_queue, log_save_filepath, status_tracker
                )
            elif self.mode == "user_agent_test":
                await self._call_api_test(
                    engagements, news_content_dict,
                    session, request_url, request_header, retry_queue, log_save_filepath, status_tracker
                )
            else:
                raise ValueError(f"Invalid mode: {self.mode}.")
        elif self.mode == "news_features":
            news_id_idx_dict, news_content_dict = self.input_dicts
            news_article = news_content_dict[self.user_id].replace("\n", " ")
            await self._call_api_news(
                news_article,
                session, request_url, request_header, retry_queue, log_save_filepath, status_tracker
            )

        elif self.mode == "comment_features":
            u_at_dict, news_id_idx_dict, news_content_dict, comments_dict = self.input_dicts
            comments_list = []
            for i, engagement in enumerate(u_at_dict[self.user_id]):
                tweet_id = engagement[1]
                comment_content = comments_dict[tweet_id].split("###")[0]
                comments_list.append(f"{i + 1}. {comment_content};")
            comments_text = " ".join(comments_list)

            comment_features_dict_path = self.save_path + f"{self.selected_user_num}/" + "comment_features_dict"
            try:
                comment_features_dict = read_ndjson(comment_features_dict_path)
            except FileNotFoundError:
                comment_features_dict = {}
            if self.user_id not in comment_features_dict:
                await self._call_api_comment_features(
                    comments_text,
                    session, request_url, request_header, retry_queue, log_save_filepath, status_tracker
                )
            else:
                logging.info(f"User {self.user_no} comment features already exist. Skipping.")
                status_tracker.num_tasks_succeeded += 1
                status_tracker.num_tasks_in_progress -= 1

        elif self.mode == "comment_generation":
            u_at_dict, news_id_idx_dict, news_content_dict, similar_news_dict = self.input_dicts

            news_features_dict_path = self.save_path + "news_features_dict"
            user_memory_dict_path = self.save_path + f"{self.selected_user_num}/" + "user_memory_dict"
            comment_features_dict_path = self.save_path + f"{self.selected_user_num}/" + "comment_features_dict"

            news_features_dict = read_ndjson(news_features_dict_path)
            user_memory_dict = read_ndjson(user_memory_dict_path)
            comment_features_dict = read_ndjson(comment_features_dict_path)

            similar_news_idx_list = similar_news_dict[self.user_id]
            user_memory = user_memory_dict[self.user_id]
            comment_features_description = comment_features_dict[self.user_id]

            await self._call_api_comment_generation(
                news_content_dict, news_features_dict, similar_news_idx_list, user_memory, comment_features_description,
                session, request_url, request_header, retry_queue, log_save_filepath, status_tracker
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}.")

    async def _call_api_news(self, news_article, session, request_url, request_header, retry_queue,
                             train_log_save_filepath, status_tracker):
        """Calls the OpenAI API and saves user memory results."""
        logging.info(f"Starting request [news features] for #{self.user_no} news")
        news_features_dict_path = self.save_path + "news_features_dict"
        if os.path.exists(news_features_dict_path + '.ndjson'):
            news_features_dict = read_ndjson(news_features_dict_path)
        else:
            news_features_dict = {}
        try:
            # ==== news feature generation ====
            with news_features_lock:
                news_features_description = news_features_dict.get(self.user_id, None)
            if news_features_description is None:
                try:
                    prompt1 = news_features_prompt(news_article)
                    news_features_description = await call_openai_api(prompt1, session,
                                                                      request_url, request_header, self.user_no)
                    with news_features_lock:
                        news_features_dict[self.user_id] = news_features_description
                        save_ndjson(news_features_dict_path, self.user_id, news_features_description)
                except Exception as e:
                    raise RuntimeError(f"Failed to generate news features for news_idx {self.user_id}: {e}")
            # status tracking
            status_tracker.num_tasks_succeeded += 1
            status_tracker.num_tasks_in_progress -= 1
            logging.debug(f"News {self.user_no} features saved.")
        except Exception as e:
            logging.warning(f"News {self.user_no} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            self.result.append(str(e))
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(f"News {self.user_no} failed after all retries: {self.result}")
                append_to_jsonl([self.user_id, self.result], train_log_save_filepath)
                status_tracker.num_tasks_failed += 1
                status_tracker.num_tasks_in_progress -= 1

    async def _call_api_train(self, engagements, news_content_dict, session, request_url, request_header, retry_queue,
                              train_log_save_filepath, status_tracker):
        """Calls the OpenAI API and saves user memory results."""
        logging.info(f"Starting request [user agent] for #{self.user_no} user")

        news_features_dict_path = self.save_path + "news_features_dict"
        user_memory_dict_path = self.save_path + f"{self.selected_user_num}/user_memory_dict"

        if os.path.exists(user_memory_dict_path + '.ndjson'):
            user_memory_dict = read_ndjson(user_memory_dict_path)
        else:
            user_memory_dict = {}

        if os.path.exists(news_features_dict_path + '.ndjson'):
            news_features_dict = read_ndjson(news_features_dict_path)
        else:
            raise RuntimeError(f"News features dict not found for user {self.user_id}.")

        if self.user_id not in user_memory_dict:
            user_agent_memory = "no memory yet"
            try:
                for idx_engagement in engagements:
                    news_idx = idx_engagement[0]
                    news_article = news_content_dict[news_idx].replace("\n", " ")
                    with news_features_lock:
                        news_features_description = news_features_dict[news_idx]
                
                    # ==== user-agent forward reasoning ====
                    try:
                        prompt2 = forward_user_agent_prompt(user_agent_memory, news_article,
                                                            news_features_description)
                        prediction_expl = await call_openai_api(prompt2, session, request_url, request_header, self.user_no)
                        chosen_action, user_explanation = extract_action_and_explanation(prediction_expl)

                        # ==== backward retry loop if ignored ====
                        modify_count = 0
                        while chosen_action.lower() == "ignore" and modify_count < 5:
                            prompt3 = backward_user_agent_prompt(user_agent_memory, news_article,
                                                                 news_features_description, user_explanation)
                            user_agent_memory = await call_openai_api(prompt3, session, request_url, request_header,
                                                                      self.user_no)

                            prompt4 = forward_user_agent_prompt(user_agent_memory, news_article,
                                                                news_features_description)
                            prediction_expl = await call_openai_api(prompt4, session, request_url, request_header,
                                                                    self.user_no)
                            chosen_action, user_explanation = extract_action_and_explanation(prediction_expl)
                            modify_count += 1

                        # ==== update user memory only if needed ====
                        if user_agent_memory == "no memory yet":
                            prompt5 = user_introduction_prompt(news_article, news_features_description)
                            user_agent_memory = await call_openai_api(prompt5, session, request_url, request_header,
                                                                      self.user_no)

                    except Exception as e:
                        raise RuntimeError(f"Failed to complete user {self.user_id}'s engagement loop: {e}")

                # ==== save final memory ====
                save_ndjson(user_memory_dict_path, self.user_id, user_agent_memory)

                # status tracking
                status_tracker.num_tasks_succeeded += len(engagements)
                status_tracker.num_tasks_in_progress -= len(engagements)
                logging.debug(f"User {self.user_no} memory saved.")

            except Exception as e:
                logging.warning(f"User {self.user_no} failed with Exception {e}")
                status_tracker.num_other_errors += 1

                self.result.append(str(e))
                if self.attempts_left:
                    retry_queue.put_nowait(self)
                else:
                    logging.error(f"User {self.user_no} failed after all retries: {self.result}")
                    append_to_jsonl([self.user_id, self.result], train_log_save_filepath)
                    status_tracker.num_tasks_failed += len(engagements)
                    status_tracker.num_tasks_in_progress -= len(engagements)
        else:
            logging.info(f"User {self.user_no} memory already exists, skipping.")
            status_tracker.num_tasks_succeeded += len(engagements)
            status_tracker.num_tasks_in_progress -= len(engagements)

    async def _call_api_test(self, engagements, news_content_dict, session, request_url, request_header, retry_queue,
                             test_log_save_filepath, status_tracker):
        metrics_save_path = os.path.dirname(test_log_save_filepath) + "/test_output.jsonl"
        logging.info(f"Starting request [TEST user agent] for #{self.user_no} user")

        news_features_dict_path = self.save_path + "news_features_dict"
        user_memory_dict_path = self.save_path + f"{self.selected_user_num}/" + "user_memory_dict"

        news_features_dict = read_ndjson(news_features_dict_path)
        user_memory_dict = read_ndjson(user_memory_dict_path)
        try:
            user_agent_memory = user_memory_dict[self.user_id]
            actions_list = []

            for idx_engagement in engagements:
                try:
                    news_idx = idx_engagement[0]
                    news_content = news_content_dict[news_idx].replace("\n", " ")
                    news_article = news_content
                    news_features_description = news_features_dict[news_idx]

                    prompt2 = forward_user_agent_prompt(user_agent_memory, news_article, news_features_description)
                    prediction_expl = await call_openai_api(prompt2, session, request_url, request_header, self.user_no)

                    while prediction_expl is None:
                        prediction_expl = await call_openai_api(prompt2, session, request_url, request_header,
                                                                self.user_no)

                    chosen_action, user_explanation = extract_action_and_explanation(prediction_expl)

                    if chosen_action.lower() == "repost":
                        actions_list.append(1)
                    elif chosen_action.lower() == "ignore":
                        actions_list.append(0)

                except Exception as e:
                    raise RuntimeError(
                        f"[TEST] Failed during user {self.user_id} engagement with news_idx {idx_engagement[0]}: {e}")

            y_pred_each_user = actions_list
            y_true_each_user = [1] * len(y_pred_each_user)
            f1_score_each_user = f1_score(y_true_each_user, y_pred_each_user, zero_division=0)

            append_to_jsonl([self.user_id, y_pred_each_user, f1_score_each_user], metrics_save_path)

            status_tracker.num_tasks_succeeded += len(engagements)
            status_tracker.num_tasks_in_progress -= len(engagements)
            logging.debug(f"User {self.user_no} [TEST] result saved.")

        except Exception as e:
            self.result.append(str(e))
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(f"User {self.user_no} failed after all retries: {self.result}")
                append_to_jsonl([self.user_id, self.result], test_log_save_filepath)
                status_tracker.num_tasks_failed += len(engagements)
                status_tracker.num_tasks_in_progress -= len(engagements)

    async def _call_api_comment_features(self, comments_text, session, request_url, request_header, retry_queue,
                                         train_log_save_filepath, status_tracker):
        """Calls the OpenAI API and saves comment style of users, then generate comments."""
        logging.info(f"Starting request [comment features] for #{self.user_no} user")

        comment_features_dict_path = self.save_path + f"{self.selected_user_num}/comment_features_dict"
        if os.path.exists(comment_features_dict_path + '.ndjson'):
            comment_features_dict = read_ndjson(comment_features_dict_path)
        else:
            comment_features_dict = {}

        try:
            prompt1 = comment_features_prompt(comments_text)
            comment_features_description = await call_openai_api(prompt1, session, request_url, request_header,
                                                                 self.user_no)
            with comment_features_lock:
                if self.user_id not in comment_features_dict:
                    comment_features_dict[self.user_id] = comment_features_description
                    save_ndjson(comment_features_dict_path, self.user_id, comment_features_description)

            # status tracking
            status_tracker.num_tasks_succeeded += 1
            status_tracker.num_tasks_in_progress -= 1
            logging.debug(f"User {self.user_id} [comment features] saved.")

        except Exception as e:
            self.result.append(str(e))
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(f"User {self.user_no} failed after all retries: {self.result}")
                append_to_jsonl([self.user_id, self.result], train_log_save_filepath)
                status_tracker.num_tasks_failed += 1
                status_tracker.num_tasks_in_progress -= 1

    async def _call_api_comment_generation(self, news_content_dict, news_features_dict, similar_news_idx_list,
                                           user_memory, comment_features_description, session, request_url,
                                           request_header, retry_queue, train_log_save_filepath, status_tracker):
       
        logging.info(f"Starting request [comment generation] for #{self.user_no} user")

        similarity_threshold = 20 
        augmented_nu_dict_path = self.save_path + f"{self.selected_user_num}/augmented_nu_dict_{similarity_threshold}"
        augmented_uc_dict_path = self.save_path + f"{self.selected_user_num}/augmented_uc_dict_{similarity_threshold}"
        augmented_comments_dict_path = self.save_path + f"{self.selected_user_num}/augmented_comments_dict_{similarity_threshold}"
        if os.path.exists(augmented_nu_dict_path + '.ndjson'):
            augmented_nu_dict = read_ndjson(augmented_nu_dict_path)
            augmented_uc_dict = read_ndjson(augmented_uc_dict_path)
            augmented_comments_dict = read_ndjson(augmented_comments_dict_path)
        else:
            augmented_nu_dict = {}
            augmented_uc_dict = {}
            augmented_comments_dict = {}

        try:
            if len(similar_news_idx_list) > 0:
                for i, similar_news_idx in enumerate(similar_news_idx_list):
                    similar_news_content = news_content_dict[similar_news_idx].replace("\n", " ")
                    similar_news_feature = news_features_dict[similar_news_idx]

                    prompt1 = comment_generation_prompt(user_memory, similar_news_content,
                                                        similar_news_feature, comment_features_description)
                    generated_comment = await call_openai_api(prompt1, session, request_url, request_header, self.user_no)
                    generated_comment_id = f"{self.user_id}_{i}"

                    with comment_generation_lock:
                        if augmented_nu_dict.get(similar_news_idx, None) is None:
                            augmented_nu_dict[similar_news_idx] = [self.user_id]
                            save_ndjson(augmented_nu_dict_path, similar_news_idx, [self.user_id])
                        else:
                            augmented_nu_dict[similar_news_idx].append(self.user_id)
                            save_ndjson(augmented_nu_dict_path, similar_news_idx, augmented_nu_dict[similar_news_idx])

                        if generated_comment_id not in augmented_comments_dict:
                            augmented_comments_dict[generated_comment_id] = generated_comment
                            save_ndjson(augmented_comments_dict_path, generated_comment_id, generated_comment)

                        if augmented_uc_dict.get(self.user_id, None) is None:
                            augmented_uc_dict[self.user_id] = [generated_comment_id]
                            save_ndjson(augmented_uc_dict_path, self.user_id, [generated_comment_id])
                        else:
                            augmented_uc_dict[self.user_id].append(generated_comment_id)
                            save_ndjson(augmented_uc_dict_path, self.user_id, augmented_uc_dict[self.user_id])
            else:
                logging.info(f"User {self.user_no} has no similar news for comment generation. Skipping.")

            # status tracking
            status_tracker.num_tasks_succeeded += len(similar_news_idx_list)
            status_tracker.num_tasks_in_progress -= len(similar_news_idx_list)
            logging.debug(f"User {self.user_no} [comment generation] saved.")

            print(f"User {self.user_no} [comment generation] saved.")

        except Exception as e:
            self.result.append(str(e))

            error_message = traceback.format_exc()
            logging.error(f"User {self.user_no} encountered an error:\n{error_message}")

            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(f"User {self.user_no} failed after all retries: {self.result}")
                append_to_jsonl([self.user_id, self.result], train_log_save_filepath)
                status_tracker.num_tasks_failed += len(similar_news_idx_list)
                status_tracker.num_tasks_in_progress -= len(similar_news_idx_list)

# functions
def evaluate_test_results(test_result_path):
    all_preds = []
    all_f1s = []

    with open(test_result_path, "r") as f:
        for line in f:
            user_id, y_pred_each_user, f1_each_user = json.loads(line)
            all_preds.extend(y_pred_each_user)
            all_f1s.append(f1_each_user)

    y_true = [1] * len(all_preds)
    accuracy = sum(all_preds) / len(all_preds)
    precision = precision_score(y_true, all_preds, zero_division=0)
    recall = recall_score(y_true, all_preds)
    f1_micro = f1_score(y_true, all_preds, average='micro')
    f1_macro = sum(all_f1s) / len(all_f1s)

    print("\n======== Testing Agent Overall Results ========")
    print("Precision: {:.2f}".format(precision))
    print("Recall: {:.2f}".format(recall))
    print("Accuracy: {:.2f}".format(accuracy))
    print("Micro F1 Score: {:.2f}".format(f1_micro))
    print("Macro F1 Score: {:.2f}".format(f1_macro))



def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""

    encoding = tiktoken.get_encoding(token_encoding_name)
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError(
                    'Expecting either string or list of strings for "prompt" field in completion request'
                )
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError(
                'Expecting either string or list of strings for "inputs" field in embedding request'
            )
    else:
        raise NotImplementedError(
            f'API endpoint "{api_endpoint}" not implemented in this script'
        )


def estimate_chat_token(prompt: str, encoding) -> int:

    base_tokens = 4  
    content_tokens = len(encoding.encode(prompt))
    return base_tokens + content_tokens + 2  


def longest_prompt_of_user(
    engagements: list,
    token_encoding_name: str,
):

    encoding = tiktoken.get_encoding(token_encoding_name)
    max_tokens = 0

    init_user_agent_memory = "no memory yet"
    placeholder = " " 
    max_news_features_description = 240
    max_user_explanation_tokens = 400
    max_generated_user_agent_memory_tokens = 360

    for idx, article in engagements:
        news_article = article.replace("\n", " ")
        prompt1 = news_features_prompt(news_article)
        tokens1 = estimate_chat_token(prompt1, encoding)
        prompt2 = forward_user_agent_prompt(init_user_agent_memory, news_article, placeholder)
        tokens2 = estimate_chat_token(prompt2, encoding) + max_news_features_description
        prompt3 = backward_user_agent_prompt(placeholder, news_article, placeholder, placeholder)
        tokens3 = (estimate_chat_token(prompt3, encoding) +
                   max_generated_user_agent_memory_tokens + max_user_explanation_tokens)
        prompt4 = forward_user_agent_prompt(placeholder, news_article, placeholder)
        tokens4 = (estimate_chat_token(prompt4, encoding) +
                   max_news_features_description + max_generated_user_agent_memory_tokens)
        max_article_tokens = max(tokens1, tokens2, tokens3, tokens4)
        max_tokens = max(max_tokens, max_article_tokens)
    return max_tokens


def longest_prompt_of_comment_features(
    comments_text: str,
    token_encoding_name: str,
):
    encoding = tiktoken.get_encoding(token_encoding_name)
    prompt = comment_features_prompt(comments_text)
    max_tokens = estimate_chat_token(prompt, encoding)
    return max_tokens


def longest_prompt_of_news_features(
    news_content: str,
    token_encoding_name: str,
):
    encoding = tiktoken.get_encoding(token_encoding_name)
    prompt = news_features_prompt(news_content)
    max_tokens = estimate_chat_token(prompt, encoding)
    return max_tokens


def longest_prompt_of_comment_generation(
    engagements_tobe: str,
    token_encoding_name: str,
):
    encoding = tiktoken.get_encoding(token_encoding_name)
    max_tokens = 0

    placeholder = " " 
    max_news_features_description = 240
    max_generated_user_agent_memory_tokens = 360
    max_comment_features_description = 240

    for idx, article in engagements_tobe:
        news_article = article.replace("\n", " ")
        prompt = comment_generation_prompt(placeholder, news_article, placeholder, placeholder)
        tokens = (estimate_chat_token(prompt, encoding) + max_news_features_description +
                  max_generated_user_agent_memory_tokens + max_comment_features_description)
        max_tokens = max(max_tokens, tokens)
    return max_tokens


def user_no_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    user_no = 0
    while True:
        yield user_no
        user_no += 1


# run script
def run_script(save_path, selected_user_num, input_dicts, mode):
    api_key = "api_key"
    target_uri = ("target_uri")

    log_save_filepath = save_path + f"{selected_user_num}/requests_results.jsonl"
    request_url = target_uri
    max_requests_per_minute = 200 * 0.3  
    max_tokens_per_minute = 200_000 * 0.3  
    token_encoding_name = "cl100k_base"
    max_attempts = 5
    logging_level = logging.INFO

    if not os.path.exists(save_path + f"{selected_user_num}/"):
        os.makedirs(save_path + f"{selected_user_num}/")

    # run script
    asyncio.run(
        process_api_requests_from_file(
            save_path=save_path,
            selected_user_num=selected_user_num,
            input_dicts=input_dicts,
            mode=mode,

            log_save_filepath=log_save_filepath,
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            token_encoding_name=token_encoding_name,
            max_attempts=max_attempts,
            logging_level=logging_level,
        )
    )
