import pandas as pd
import numpy as np
import json
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import re
import textstat
import spacy
from tqdm.notebook import tqdm
from scipy.stats import permutation_test, median_abs_deviation,mannwhitneyu

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

def get_partial_answers(log_entries):
    part_answers = []
    for entry in log_entries:
        if (entry["type"] == "AnswerBoxClosed") or (entry["type"] == "ClickedOnSendAndTerminateTask"):
            uid = entry["uid"]
            topic_num = entry["actual_topic_number"]
            answer = entry["answer"]
            if pd.isna(answer):
                answer = ""
            timestamp = entry["timestamp"]
            part_answers.append([uid, assign_topic_number(uid, topic_num), answer, timestamp])

    part_answers_df = pd.DataFrame(part_answers, columns = ["uid", "topic_number", "answer", "timestamp"])
    part_answers_df["uid"] = part_answers_df["uid"].astype("Int64")
    part_answers_df["timestamp"] = pd.to_datetime(part_answers_df["timestamp"])
    return part_answers_df

def get_final_answers(log_entries):
    final_answer = []
    topics = pd.read_csv("Data/topics.csv")
    for entry in log_entries:
        if entry["type"] == "TaskStarted":
            topic_num = assign_topic_number(entry["uid"], entry["actual_topic_number"])
            topic = topics.loc[topics["tid"]==topic_num]["question"].values[0]
            final_answer.append([entry["uid"], topic_num, topic, None, "TaskStart", entry["timestamp"]])
        elif entry["type"] == "TaskEnded":
            # print(entry["uid"])
            topic_num = assign_topic_number(entry["uid"], entry["actual_topic_number"])
            topic = topics.loc[topics["tid"]==topic_num]["question"].values[0]
            final_answer.append([entry["uid"], topic_num, topic, entry["answer"], "TaskEnd", entry["timestamp"]])

    final_answer_df = pd.DataFrame(final_answer, columns = ["uid", "topic_number", "prompt", "answer", "TaskStartOrEnd", "timestamp"])
    final_answer_df["uid"] = final_answer_df["uid"].astype("Int64")
    final_answer_df["timestamp"] = pd.to_datetime(final_answer_df["timestamp"])
    final_answer_df = final_answer_df.drop_duplicates(ignore_index=True)
    return final_answer_df

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

    nlp = spacy.load("it_core_news_lg")
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
        words = []
        for token in doc:
            if token.pos_ not in ["SYM", "X", "PUNCT"]:
                words.append(token.text.strip().lower())
        content_len.append(len(words))

        # aoa and concreteness
        aoa_vals = []
        conc_vals = []
        for token in doc:
            if token.text.lower() in ITaoa["Ita_Word"].tolist():
                aoa_vals.append(ITaoa.loc[ITaoa["Ita_Word"]==token.text.lower()]["M_AoA"].values[0])
            if token.text.lower() in concreteness["Ita_Word"].tolist():
                conc_vals.append(concreteness.loc[concreteness["Ita_Word"]==token.text.lower()]["M_Con"].values[0])

        if len(aoa_vals)==0:
            mean_aoa.append(None)
        else:
            mean_aoa.append(np.mean(aoa_vals))
        if len(conc_vals) == 0:
            mean_conc.append(None)
        else:
            mean_conc.append(np.mean(conc_vals))

        # readability
        gulpease_index.append(textstat.gulpease_index(row[text_col_name]))
        # fres.append(textstat.flesch_reading_ease(row[text_col_name]))

    return {"content_len": content_len, "mean_aoa": mean_aoa, "mean_conc": mean_conc, "gulpease_index": gulpease_index}

# statistical significance tests
def get_significance(df, effective_label, search_stage, categories):
    
    effective = df.loc[df[effective_label]==1]
    ineffective = df.loc[df[effective_label]==0]


    desc_stats = []
    for category in categories:

        effective_cat = effective[category].dropna()
        ineffective_cat = ineffective[category].dropna()

        if len(ineffective) == 0:
                ineffective_med = None
                ineffective_MAD = None
        else:
            ineffective_med = ineffective_cat.median()
            ineffective_MAD = (ineffective_cat - ineffective_med).abs().median()
        if len(effective) == 0:
            effective_med = None
            effective_MAD = None
        else:
            effective_med = effective_cat.median()
            effective_MAD = (effective_cat - effective_med).abs().median()
        desc_stats.append([search_stage, category, ineffective_med, ineffective_MAD, effective_med, effective_MAD])

    desc_stats_df = pd.DataFrame(desc_stats, columns=["search_stage", "characteristic", "ineffective_median", "ineffective_MAD", "effective_median", "effective_MAD"])
    desc_stats_df.to_csv(f"Results/{search_stage}_effective_ineffective_desc_stats.csv", index=False, encoding="utf8")

    sig_test_results = []
    rng = np.random.default_rng(seed=111)
    

    def statistic(x, y):
        # x = np.array(x)
        x = x[~np.isnan(x)] # removing rows with NaN values
        # y = np.array(y)
        y = y[~np.isnan(y)]
        return np.median(x) - np.median(y)
        # return x.median() - y.median()

    for category in categories:
        res = permutation_test(data=(ineffective[category], effective[category]), statistic=statistic, rng=rng)
        # res = mannwhitneyu(ineffective[category], effective[category], nan_policy="omit", alternative="two-sided")
        sig_test_results.append([search_stage, category, res.statistic, res.pvalue, res.pvalue<0.05])

    sig_test_df = pd.DataFrame(sig_test_results, columns=["search_stage", "characteristic", "test_statistic", "pval", "significant"])
    sig_test_df.to_csv(f"Results/sig_test_{search_stage}.csv", index=False, encoding="utf8")

    return desc_stats_df, sig_test_df

# visualization

def get_boxplots(df, search_stage, categories, group, artefact, n_cols, n_rows, stats_df=None):
    plt.figure(figsize=(16,9))
    plt.subplots_adjust(hspace=0.65)

    df = df.loc[pd.notna(df[group])]

    for i, category in enumerate(categories): 
        ax = plt.subplot(n_rows, n_cols, i+1) #nrows, ncols, index
        outlier_flag = 0
        # removing outliers for visualisation
        if search_stage == "query_formulation":
            if category == "gulpease_index":
                outlier_flag = 1
                outliers = df.loc[df[category] >= 300].index
                data_df = df.drop(outliers, axis=0)
                # ax.set_xlabel(f"Removed outliers ({len(outliers)}) with value >= 300")
            elif category == "levenshtein_dist":
                outlier_flag = 1
                outliers = df.loc[df[category] > 4].index
                data_df = df.drop(outliers, axis=0)
                # ax.set_xlabel(f"Removed outliers ({len(outliers)}) with value > 4")
            else:
                data_df = df
        elif "results_navigation" in search_stage:
            if category == "avg_scroll_time_before_click":
                outlier_flag = 1
                outliers = df.loc[df[category] > 80].index
                data_df = df.drop(outliers, axis=0)
                # ax.set_xlabel(f"Removed outliers ({len(outliers)}) with value > 80")
            elif category == "avg_snippet_dwell_time_before_click":
                outlier_flag = 1
                outliers = df.loc[df[category]>150].index
                data_df = df.drop(outliers, axis=0)
                # ax.set_xlabel(f"Removed outliers ({len(outliers)}) with value > 150")
            elif category == "first_click_time":
                outlier_flag = 1
                outliers = df.loc[df[category] > 50].index
                data_df = df.drop(outliers, axis=0)
                # ax.set_xlabel(f"Removed outliers ({len(outliers)}) with value > 50")
            else:
                data_df = df
        elif search_stage == "information_distillation":
            if category == "mean_irrel_time":
                outlier_flag = 1
                outliers = df.loc[df[category] > 100].index
                data_df = df.drop(outliers, axis=0)
                # ax.set_xlabel(f"Removed outliers ({len(outliers)}) with value > 100")
            else:
                data_df = df
                
        sns.boxplot(data=data_df, x=group, y=category, hue=group,
                    # showmeans=True,     
                    #     meanprops={
                    #         "marker": "D",                    # Circle marker
                    #         "markerfacecolor": "white",       # White inside
                    #         "markeredgecolor": "black",       # Black border
                    #         "markersize": "7"                 # Size of marker
                    #     }
                    )
        ax.set_xlabel(category, fontdict={"fontsize": 15})
        # if outlier_flag == 0:
        #     ax.set_xlabel("")

        # xtick_map = {'0':'ineffective '+artefact, '1':'effective '+artefact}
        ax.set_xticklabels("")
        ax.set_ylabel("")
        if (stats_df is not None) and (stats_df.loc[stats_df["characteristic"]==category]["significant"].values[0]==True):
            ax.set_xlabel(category+" *", fontdict={"fontsize": 15})
        ax.legend().remove()

    legend_map = {'0.0':'ineffective '+artefact, '0':'ineffective '+artefact, '1.0':'effective '+artefact, '1':'effective '+artefact}

    handles, labels = ax.get_legend_handles_labels()
    # print(labels)
    plt.figlegend(handles, [legend_map[l] for l in labels], loc="upper center", ncol=2, fontsize=20)
    plt.savefig("Figures/"+search_stage+"_bar.pdf", dpi=300, bbox_inches='tight')

    # plt.figlegend(handles=legend_elements, loc="upper right")
    plt.show()

def get_descriptive_stats(df, search_stage, categories, ncols, nrows, binwidth_map=None):
    desc_stats = []

    plt.figure(figsize=(16,9))
    plt.subplots_adjust(hspace=0.45)

    for i, category in enumerate(categories): 
        ax = plt.subplot(nrows, ncols, i+1) #nrows, ncols, index
    # for category in categories:
        if category in ["num_concepts", "levenshtein_dist", "num_typos", "query_len", "rank_lowest_view", "SERP_total_results", "SERP_highest_rel_result", "tail", "repetitions", "avg_snippet_view_counts_before_click", "total_clicks", "rank", "page_rank"]:
            sns.histplot(
                data=df[category],
                ax=ax,
                stat="proportion",
                alpha=0.1,
                discrete=True,
                kde=True,
                kde_kws={"cut":0}
            )

            # sns.histplot(
            #     data=df[category],
            #     ax=ax,
            #     stat="proportion",
            #     element="poly",
            #     fill=False,
            #     discrete=True
            # )
            # # Get bin edges from the bar patches
            # bin_edges = [p.get_x() for p in ax.patches]
            # bin_edges.append(ax.patches[-1].get_x() + ax.patches[-1].get_width())
            # # print(bin_edges)

            # # Show all bin edges as xticks
            # ax.set_xticks(bin_edges)
            # ax.set_xticklabels([f'{x:.0f}' for x in bin_edges], rotation=45)
        else:
            if binwidth_map == None:
                sns.histplot(
                        data=df[category],
                        ax=ax,
                        stat="proportion",
                        alpha=0.1,
                        kde=True,
                        kde_kws={"cut":0}
                    )
                # sns.histplot(
                #         data=df[category],
                #         ax=ax,
                #         stat="proportion",
                #         fill=False,
                #         element="poly"
                #     )
            else:
                sns.histplot(
                    data=df[category],
                    ax=ax,
                    binwidth=binwidth_map[category],
                    stat="proportion",
                    alpha=0.1,
                    kde=True,
                    kde_kws={"cut":0}
                )
                # sns.histplot(
                #     data=df[category],
                #     ax=ax,
                #     binwidth=binwidth_map[category],
                #     stat="proportion",
                #     fill=False,
                #     element="poly"
                # )
                # # Get bin edges from the bar patches
                # bin_edges = [p.get_x() for p in ax.patches]
                # bin_edges.append(ax.patches[-1].get_x() + ax.patches[-1].get_width())
                # print(bin_edges)

                # # Show all bin edges as xticks
                # ax.set_xticks(bin_edges)
                # ax.set_xticklabels([f'{x:.2f}' for x in bin_edges], rotation=45)

        # if category=="gulpease_index":
        #     ax.axvline(80, linestyle="--", color="red")

        x = df[category].dropna()
        mean = x.mean()
        median = x.median()
        modes = x.mode().tolist()
        std_dev = x.std()
        maximum = x.max()
        minimum = x.min()

        # ax.axvline(mean, linestyle="--", color="green", label="mean")
        ax.axvline(median, linestyle=":", color="red", label="median")
        # for mode in modes:
        #     ax.axvline(mode, linestyle=":", color="black", label="mode")
        # ax.axvspan(mean-std_dev, mean+std_dev, alpha=0.2, fill=True, color="red", label="mean ± std")

        desc_stats.append([category, mean, std_dev, maximum, minimum, median, modes])
        
        # ax.set_xlabel(category, fontdict={"fontsize":15})
    #     ax.set_ylabel("")
        # ax.set_title("")

    #     # ax.set_yticklabels([])

    #     # plt.savefig(f"Figures/desc_stats_{search_stage}_{category}.pdf", dpi=300, bbox_inches="tight")
    #     # plt.show()
    handles, labels = ax.get_legend_handles_labels()
    # plt.figlegend(handles, labels, loc="upper center", ncol=1, fontsize=20)

    plt.savefig(f"Figures/desc_stats_{search_stage}.pdf", dpi=300, bbox_inches="tight")
    plt.show()
    desc_stats_df = pd.DataFrame(desc_stats, columns=["category", "mean", "std_dev", "max", "min", "median", "modes"])
    desc_stats_df.to_csv(f"Results/desc_stats_{search_stage}.csv", index=False, encoding="utf8")
    return desc_stats_df

def get_radar_plot(df, search_stage, categories, artefact, group, normalized=True):

    scaled_df = df[categories].apply(zscore, nan_policy='omit')
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

    # category_names = []
    # for category in categories:
    #     if (stats_df is not None) and (stats_df.loc[stats_df["feature"]==category]["significant"].values[0]==True):
    #         category_names.append(category+" *")
    #     else:
    #         category_names.append(category)
    # ax.set_xticklabels(category_names)
    ax.set_xticklabels(categories)

    legend_map = {'$0$':'failed '+artefact, '$1$':'partially successsful '+artefact, '$2$': 'successful '+artefact}

    handles, labels = ax.get_legend_handles_labels()
    # print(labels)
    ax.legend(handles, [legend_map[l] for l in labels], loc='upper right', bbox_to_anchor=(1.5, 1.1))
    # plt.savefig("Figures/"+search_stage+"_radar.pdf", dpi=300, bbox_inches='tight')
    # plt.title(f"Scaled using {scaling}", y=1.05, fontsize=14)

    plt.show()
