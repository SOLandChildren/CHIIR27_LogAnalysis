import pandas as pd
import numpy as np
import json

# global topic number: g<grade>t<topic_num>
def assign_topic_number(uid, topic_num):
    if int(uid) <=  8 or int(uid) >= 17:
        grade = "3"
    else: 
        grade = "4"
    return "g"+grade+"t"+str(topic_num)

def get_result_rank(page_rank, page_num):
    return int(page_rank) + ((int(page_num)-1)*10)


def get_queries(log_entries):
    query_actions = []
    for entry in log_entries:
        if entry["type"] == "querySubmitted":
            if entry["query"] != "":
                query_actions.append([entry["uid"], assign_topic_number(entry["uid"], entry["actual_topic_number"]), entry["rawQuery"], entry["timestamp"]])

    
    query_df = pd.DataFrame(query_actions, columns = ["uid", "topic_number", "query", "timestamp"])
    query_df["uid"] = query_df["uid"].astype("Int64")
    query_df["timestamp"] = pd.to_datetime(query_df["timestamp"])

    return query_df

def get_query_actions(log_entries):
    query_actions = []
    for entry in log_entries:
        topic_num = assign_topic_number(entry["uid"], entry["actual_topic_number"])
        if entry["type"] == "querySubmitted":
            query_actions.append([entry["uid"], topic_num, entry["rawQuery"], "UserSubmittedQuery", entry["timestamp"]])
        elif entry["type"] == "choseAutoCompleteSuggestion":
            query_actions.append([entry["uid"], topic_num, entry["selectedSuggestion"], "AutoCompleteSuggestion", entry["timestamp"]])
        elif entry["type"] == "clickedDidYouMeanSuggestion":
            query_actions.append([entry["uid"], topic_num, entry["suggested query"], "QueryCorrection", entry["timestamp"]])
        elif entry["type"] == "taskQuestionHelpClicked":
            query_actions.append([entry["uid"], topic_num, "", "TaskQuestionHelp", entry["timestamp"]])


    query_actions_df = pd.DataFrame(query_actions, columns = ["uid", "topic_number", "query", "query_action", "timestamp"])
    query_actions_df["uid"] = query_actions_df["uid"].astype("Int64")
    query_actions_df["timestamp"] = pd.to_datetime(query_actions_df["timestamp"])
    return query_actions_df

def get_generated_serps(log_entries):
    generated_serps = []
    for entry in log_entries:
        topic_num = assign_topic_number(entry["uid"], entry["actual_topic_number"])
        if entry["type"] == "searchResultGenerated":
            generated_serps.append([entry["uid"], topic_num, entry["query"], entry["url"], entry["title"], entry["snippet"], get_result_rank(entry["rank"], entry["page"]), entry["timestamp"]])
        if entry["type"] == "searchNoResults":
            # print("No results for query:", entry["query"])
            generated_serps.append([entry["uid"], topic_num, entry["query"], None, None, None, None, entry["timestamp"]])

    serp_df = pd.DataFrame(generated_serps, columns = ["uid", "topic_number", "query", "url", "title", "snippet", "rank", "timestamp"])
    # serp_df["rank"] = serp_df["rank"].astype("Int64")
    serp_df["timestamp"] = pd.to_datetime(serp_df["timestamp"])
    serp_df["uid"] = serp_df["uid"].astype("Int64")
    
    return serp_df

def get_clicked_results(log_entries):
    clicked_results = []
    for entry in log_entries:
        if entry["type"] == "clickedResult":
            clicked_results.append([entry["uid"], assign_topic_number(entry["uid"], entry["actual_topic_number"]), entry["query"], entry["url"], get_result_rank(entry["rank"], entry["page"]), entry["timestamp"]])

    clicked_results_df = pd.DataFrame(clicked_results, columns = ["uid", "topic_number", "query", "url", "rank", "timestamp"])
    clicked_results_df["uid"] = clicked_results_df["uid"].astype("Int64")
    clicked_results_df["rank"] = clicked_results_df["rank"].astype("Int64")
    clicked_results_df["timestamp"] = pd.to_datetime(clicked_results_df["timestamp"])

    return clicked_results_df

def get_viewed_results(log_entries):
    viewed_results = []
    for entry in log_entries:
        if (entry["type"] == "resultExposureEnded") or (entry["type"]=="resultExposureStarted"):
            viewed_results.append([entry["uid"], assign_topic_number(entry["uid"], entry["actual_topic_number"]), entry["query"], entry["url"], get_result_rank(entry["rank"], entry["page"]), entry["type"], entry["timestamp"]])

    viewed_results_df = pd.DataFrame(viewed_results, columns = ["uid", "topic_number", "query", "url", "rank", "viewType", "timestamp"])
    viewed_results_df["uid"] = viewed_results_df["uid"].astype("Int64")
    viewed_results_df["rank"] = viewed_results_df["rank"].astype("Int64")
    viewed_results_df["timestamp"] = pd.to_datetime(viewed_results_df["timestamp"])

    return viewed_results_df


def get_final_answers(log_entries):
    final_answer = []
    topics = pd.read_csv("Data/topics.csv")
    for entry in log_entries:
        if entry["type"] == "TaskEnded":
            # print(entry["uid"])
            topic_num = assign_topic_number(entry["uid"], entry["actual_topic_number"])
            topic = topics.loc[topics["tid"]==topic_num]["question"].values[0]
            final_answer.append([entry["uid"], topic_num, topic, entry["answer"], entry["timestamp"]])

    final_answer_df = pd.DataFrame(final_answer, columns = ["uid", "topic_number", "prompt", "answer", "timestamp"])
    final_answer_df["uid"] = final_answer_df["uid"].astype("Int64")
    final_answer_df["timestamp"] = pd.to_datetime(final_answer_df["timestamp"])
    final_answer_df = final_answer_df.drop_duplicates(ignore_index=True)
    return final_answer_df

