import pandas as pd
import numpy as np
import json
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import re
import textstat
import spacy
from tqdm.notebook import tqdm
from scipy.stats import permutation_test

# dataset creation

## global topic number: g<grade>t<topic_num>
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
        topic_num = assign_topic_number(entry["uid"], entry["actual_topic_number"])
        if (entry["type"] == "querySubmitted"):
            if entry["query"].strip() != "":
                query_actions.append([entry["uid"], topic_num, entry["query"].strip(), "UserFormulated", entry["timestamp"]])
        elif (entry["type"]=="clickedDidYouMeanSuggestion"):
            query_actions.append([entry["uid"], topic_num, entry["suggested query"], "DidYouMean", entry["timestamp"]])

    
    query_df = pd.DataFrame(query_actions, columns = ["uid", "topic_number", "query", "source", "timestamp"])
    query_df["uid"] = query_df["uid"].astype("Int64")
    query_df["timestamp"] = pd.to_datetime(query_df["timestamp"])

    return query_df

def get_query_corrections(log_entries):
    query_corrections = []
    for entry in log_entries:
        topic_num = assign_topic_number(entry["uid"], entry["actual_topic_number"])
        if (entry["type"]=="generatedDidYouMean"):
            try:
                query_corrections.append([entry["uid"], topic_num, entry["user query"].strip(), entry["suggested query"], entry["timestamp"]])
            except Exception as e:
                print(entry["uid"], entry["timestamp"])
                query_corrections.append([entry["uid"], topic_num, entry["query"].strip(), entry["suggested query"], entry["timestamp"]])

    query_corrections_df = pd.DataFrame(query_corrections, columns = ["uid", "topic_number", "original_query", "corrected_query", "timestamp"])
    query_corrections_df["uid"] = query_corrections_df["uid"].astype("Int64")
    query_corrections_df["timestamp"] = pd.to_datetime(query_corrections_df["timestamp"])
    return query_corrections_df

def get_query_actions(log_entries):
    query_actions = []
    for entry in log_entries:
        topic_num = assign_topic_number(entry["uid"], entry["actual_topic_number"])
        if entry["type"] == "querySubmitted":
            query_actions.append([entry["uid"], topic_num, entry["query"].strip(), "UserSubmittedQuery", entry["timestamp"]])
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
            generated_serps.append([entry["uid"], topic_num, entry["query"].strip(), entry["url"], entry["title"], entry["snippet"], get_result_rank(entry["rank"], entry["page"]), entry["timestamp"]])
        if entry["type"] == "searchNoResults":
            # print("No results for query:", entry["query"].strip())
            generated_serps.append([entry["uid"], topic_num, entry["query"].strip(), None, None, None, None, entry["timestamp"]])

    serp_df = pd.DataFrame(generated_serps, columns = ["uid", "topic_number", "query", "url", "title", "snippet", "rank", "timestamp"])
    # serp_df["rank"] = serp_df["rank"].astype("Int64")
    serp_df["timestamp"] = pd.to_datetime(serp_df["timestamp"])
    serp_df["uid"] = serp_df["uid"].astype("Int64")
    
    return serp_df

def get_clicked_results(log_entries):
    clicked_results = []
    for entry in log_entries:
        if entry["type"] == "clickedResult":
            clicked_results.append([entry["uid"], assign_topic_number(entry["uid"], entry["actual_topic_number"]), entry["query"].strip(), entry["url"], get_result_rank(entry["rank"], entry["page"]), entry["timestamp"]])

    clicked_results_df = pd.DataFrame(clicked_results, columns = ["uid", "topic_number", "query", "url", "rank", "timestamp"])
    clicked_results_df["uid"] = clicked_results_df["uid"].astype("Int64")
    clicked_results_df["rank"] = clicked_results_df["rank"].astype("Int64")
    clicked_results_df["timestamp"] = pd.to_datetime(clicked_results_df["timestamp"])

    return clicked_results_df

def get_viewed_results(log_entries):
    viewed_results = []
    for entry in log_entries:
        if entry["type"] == "resultExposureEnded":
            viewed_results.append([entry["uid"], assign_topic_number(entry["uid"], entry["actual_topic_number"]), entry["query"].strip(), entry["url"], get_result_rank(entry["rank"], entry["page"]), entry["type"], entry["durationMs"]/1000, entry["timestamp"]])
        elif entry["type"]=="resultExposureStarted":
            viewed_results.append([entry["uid"], assign_topic_number(entry["uid"], entry["actual_topic_number"]), entry["query"].strip(), entry["url"], get_result_rank(entry["rank"], entry["page"]), entry["type"], None, entry["timestamp"]])
    viewed_results_df = pd.DataFrame(viewed_results, columns = ["uid", "topic_number", "query", "url", "rank", "viewType", "viewDurationInSeconds", "timestamp"])
    viewed_results_df["uid"] = viewed_results_df["uid"].astype("Int64")
    viewed_results_df["rank"] = viewed_results_df["rank"].astype("Int64")
    viewed_results_df["timestamp"] = pd.to_datetime(viewed_results_df["timestamp"])

    return viewed_results_df

def get_explored_web_pages(log_entries):
    web_pages = []
    for entry in log_entries:
        if entry["type"] == "webpageClosed":
            web_pages.append([entry["uid"], assign_topic_number(entry["uid"], entry["actual_topic_number"]), entry["url"], entry["type"], entry["durationMs"]/1000, entry["timestamp"]])
        elif entry["type"] == "webpageOpened":
            web_pages.append([entry["uid"], assign_topic_number(entry["uid"], entry["actual_topic_number"]), entry["url"], entry["type"], None, entry["timestamp"]])

    web_pages_df = pd.DataFrame(web_pages, columns = ["uid", "topic_number", "url", "exploreType", "viewDurationInSeconds", "timestamp"])
    web_pages_df["uid"] = web_pages_df["uid"].astype("Int64")
    web_pages_df["timestamp"] = pd.to_datetime(web_pages_df["timestamp"])

    return web_pages_df

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

# visualization

def get_boxplots(df, categories, group, artefact, n_cols, n_rows, stats_df=None):
    plt.figure(figsize=(16,9))
    # plt.subplots_adjust(hspace=0.5)
    for i, category in enumerate(categories): 
        ax = plt.subplot(n_rows, n_cols, i+1)
        sns.boxplot(data=df, x=group, y=category, hue=group, showmeans=True,     
                        meanprops={
                            "marker": "D",                    # Circle marker
                            "markerfacecolor": "white",       # White inside
                            "markeredgecolor": "black",       # Black border
                            "markersize": "7"                 # Size of marker
                        }
                    )
        ax.set_title(category)

        # xtick_map = {'0':'ineffective '+artefact, '1':'effective '+artefact}
        ax.set_xticklabels("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        if (stats_df is not None) and (stats_df.loc[stats_df["feature"]==category]["significant"].values[0]==True):
            ax.set_title(category+" *")
        ax.legend().remove()

    legend_map = {'0':'ineffective '+artefact, '1':'effective '+artefact}

    handles, labels = ax.get_legend_handles_labels()
    # print(labels)
    plt.figlegend(handles, [legend_map[l] for l in labels], loc="upper center", ncol=2)

    # plt.figlegend(handles=legend_elements, loc="upper right")
    plt.show()


def get_radar_plot(df, categories, artefact, group, normalized=True, stats_df=None):

    # for scaling in ["zscore", "minmax"]:
    #     if scaling=="zscore":

    scaled_df = df[categories].apply(zscore, nan_policy='omit')
        # print(z_standardized)
    # elif scaling == "minmax":

    #     scaler = MinMaxScaler()

    #     scaled_df = pd.DataFrame(
    #         scaler.fit_transform(df[categories]),
    #         columns=categories,
    #         index=df.index
    #     )

    scaled_df[group] = df[group]
    cluster_means = scaled_df.groupby(group).mean()
    # cluster_means = cluster_means.apply(zscore)
    # print(cluster_means)

    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  

    fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))

    for i, row in cluster_means.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, label=f'${i}$', linewidth=2)
        ax.fill(angles, values, alpha=0.1)

    if normalized:
        ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])

    category_names = []
    for category in categories:
        if (stats_df is not None) and (stats_df.loc[stats_df["feature"]==category]["significant"].values[0]==True):
            category_names.append(category+" *")
        else:
            category_names.append(category)
    ax.set_xticklabels(category_names)

    legend_map = {'$0$':'ineffective '+artefact, '$1$':'effective '+artefact}

    handles, labels = ax.get_legend_handles_labels()
    # print(labels)
    ax.legend(handles, [legend_map[l] for l in labels], loc='upper right', bbox_to_anchor=(1.1, 1.1))
    # plt.title(f"Scaled using {scaling}", y=1.05, fontsize=14)

    plt.show()

# text processing
def clean_text(text):
    try:
        if pd.isna(text):
            return ""
        else:
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r"\n+", ". ", text)
            text = re.sub(r"\\'", "'", text)
            text = re.sub(r"\"", "", text)
            text = re.sub(r"\\'|[’‘]", "'", text)
            text = text.replace("\xa0", " ") # Remove accidental escape sequences
            return text.strip()
    except Exception as e:
        print(f"Error cleaning text {text}: {e}")
        return text

def get_text_stats(df, text_col_name):
    textstat.set_lang("it")

    nlp = spacy.load("it_core_news_sm")
    ITaoa = pd.read_excel("Resources/ItAoA_database.xlsx")
    concreteness = pd.read_excel("Resources/IT_concreteness_database.xlsx")

    content_len = []
    mean_aoa = []
    gulpease_index = []
    # fres = []
    mean_conc = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        doc = nlp(row[text_col_name])
        
        # content length: number of words
        content_len.append(sum(
            1 for token in doc
            if token.is_alpha or token.like_num
        ))

        # aoa and concreteness
        aoa_vals = []
        conc_vals = []
        for token in doc:
            if token.text.lower() in ITaoa["Ita_Word"].tolist():
                aoa_vals.append(ITaoa.loc[ITaoa["Ita_Word"]==token.text.lower()]["M_AoA"].values[0])
            if token.text.lower() in concreteness["Ita_Word"].tolist():
                conc_vals.append(concreteness.loc[concreteness["Ita_Word"]==token.text.lower()]["M_Con"].values[0])
        mean_aoa.append(np.mean(aoa_vals))
        mean_conc.append(np.mean(conc_vals))

        # readability
        gulpease_index.append(textstat.gulpease_index(row[text_col_name]))
        # fres.append(textstat.flesch_reading_ease(row[text_col_name]))

    return {"content_len": content_len, "mean_aoa": mean_aoa, "mean_conc": mean_conc, "gulpease_index": gulpease_index}

def get_significance(df, performance_metric, threshold, stage, categories):
    sig_test_results = []

    effective = df.loc[df[performance_metric]>=threshold]
    ineffective = df.loc[df[performance_metric]<threshold]

    def statistic(x, y):
        x = pd.Series(x, copy=False)
        y = pd.Series(y, copy=False)
        return x.mean() - y.mean()

    for category in categories:
        ineffective_category = ineffective[category]
        effective_category = effective[category]
        pval = permutation_test(data=(ineffective_category, effective_category), statistic=statistic).pvalue
        sig_test_results.append([stage, category, statistic(ineffective_category, effective_category), pval, pval<0.05])

    sig_test_df = pd.DataFrame(sig_test_results, columns=["search_stage", "feature", "feature_mean_val_diff", "pval", "significant"])
    # sig_test_df.to_csv("sig_test_"+stage+".csv", index=False, encoding="utf8")

    return sig_test_df
