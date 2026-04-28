import os
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HUB_OFFLINE"] = "1"
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import re
import pandas as pd
import hdbscan
import umap
import numpy as np
import random
import time
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from thesis.connect_sql import connect_sql
from thesis.cluster_analysis.pmi_miner import PMIMiner
from thesis.cluster_analysis.jieba_cut import JieBaCut
from pyecharts import options as opts
from pyecharts.charts import Scatter3D
from functools import wraps
from collections import Counter
from Sentiment_Analysis.S_Analysis import SA


def how_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        s_time = time.time()
        result = func(*args, **kwargs)
        e_time = time.time()
        all_time = e_time - s_time
        print(f"总共耗时:{all_time / 60: .2f}分钟")

        return result

    return wrapper


# 分句，避免混合情感，也便于聚类
def split_comment(text) -> list[str]:
    text = re.split(r'[。！？!?，,\s]+|(?=\d+\.)', str(text))

    return [t for t in text if len(t) > 2]


# 类似c-tf-idf函数
def top_keywords_per_cluster(dataframe, n, max_df):
    # 跳过噪声
    df_valid = dataframe[dataframe["cluster"] != -1].copy()
    noise = dataframe[dataframe["cluster"] == -1]

    df_clean = df_valid[df_valid["cut_comment"].str.strip() != ""].copy()
    print(f"噪音{len(noise)}条,正在对 {len(df_clean)} 条有效关键词进行权重计算")

    #  c-TF-IDF
    docs_per_cluster = df_clean.groupby(['cluster'], as_index=False).agg({'cut_comment': ' '.join})
    # with open('cut_result.txt', 'w', encoding='UTF-8') as f:
    #     for index, row in docs_per_cluster.iterrows():
    #         cluster_id = row['cluster']
    #         content = row['cut_comment']
    #
    #         f.write(f"Cluster {cluster_id} 分词结果\n")
    #         f.write("--------------------------------------------------\n")
    #         f.write(content)
    #         f.write("\n\n" + "=" * 50 + "\n\n")
    # 计算
    vectorizer = TfidfVectorizer(
        max_features=1000,
        max_df=max_df,
        # ngram_range=(1, 2), # 加入ngram 短语匹配
        token_pattern=r"(?u)\b\w+\b"
    )
    try:
        x = vectorizer.fit_transform(docs_per_cluster['cut_comment'])
        feature_names = vectorizer.get_feature_names_out()
    except ValueError:
        print("词表为空，无法提取关键词。")
        return

    cluster_keywords_dict = {}
    # 遍历输出
    for i, row in docs_per_cluster.iterrows():
        c = row['cluster']

        vector = x[i].toarray().flatten()
        indices = vector.argsort()[::-1][:n]

        print("-----------", f"cluster{c}核心词及其权重", "-----------", sep='\n')

        current_cluster_words = []
        for ind in indices:
            word = feature_names[ind]
            score = vector[ind]  # 获取权重分数
            if score < 0.05:
                continue
            print(f"{word}: {score:.4f}")

            current_cluster_words.append((word, score))

        cluster_keywords_dict[c] = current_cluster_words


@how_time
def cluster_main(embeddings, xid, comment, cut_word=None, is_first=True, best_param=None, file_prefix="全局"):
    best_params = dict()
    best_score = -100
    best_labels = None

    # 网格搜索参数
    if best_param is not None:
        param_grid = {
            "n_neighbors": [best_param["n_neighbors"]],
            "min_cluster_size": [best_param["min_cluster_size"]],
            "min_samples": [best_param["min_samples"]]
        }
    else:
        param_grid = {
        # "n_neighbors": list(range(10, 51)),
        # "min_cluster_size": list(range(10, 51)),
        # "min_samples": list(range(2, 6))
        #     "n_neighbors": list(range(10, 61)),
        #     "min_cluster_size": list(range(30, 151)),
        #     "min_samples": list(range(2, 9))
        }
        param_grid = get_hyper_parameter(len(embeddings), is_first=is_first)

    print("网格搜索启动！！")
    for n_n in param_grid["n_neighbors"]:
        umap_model = umap.UMAP(
            n_neighbors=n_n,
            n_components=10,
            min_dist=0.1,
            metric='cosine',
            random_state=42,
            n_jobs=1
        )
        reduced_embeddings = umap_model.fit_transform(embeddings)

        try:
            for mcs in param_grid["min_cluster_size"]:
                for ms in param_grid["min_samples"]:
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=mcs,
                        min_samples=ms,
                        metric='euclidean',
                        cluster_selection_method='eom',
                        prediction_data=True,
                        gen_min_span_tree=True,
                        core_dist_n_jobs=-1
                    )
                    labels = clusterer.fit_predict(reduced_embeddings)

                    noise = np.sum(labels == -1)
                    noise_ratio = noise / len(labels)
                    u_labels = np.unique(labels[labels != -1])
                    n_clusters = len(u_labels)

                    if n_clusters < 5 or n_clusters >= 50:
                        continue

                    if noise_ratio > 0.4:
                        continue

                    mask = labels != -1
                    x_valid = reduced_embeddings[mask]
                    l_valid = labels[mask]

                    if len(np.unique(l_valid)) < 2:
                        score = -1
                    else:
                        try:
                            score = hdbscan.validity.validity_index(x_valid.astype('float64'), l_valid)
                        except Exception as e:
                            print(f"DBCV报错: {e}")
                            score = -1

                    current_per = pd.Series(labels).value_counts().iloc[0] / len(labels)
                    if current_per > 0.6:
                        score = -0.5

                    # print(f"{score},{min_cluster_size},{min_samples}")
                    if score > best_score and score > 0.03:
                        best_score = score
                        best_labels = labels
                        best_params = {
                            "n_neighbors": n_n,
                            "min_samples": ms,
                            "min_cluster_size": mcs
                        }
                        print(f"最高分{best_score:.4f}, 参数{n_n, mcs, ms}")
        except Exception as e:
            print(e)
        continue

    if best_labels is None:
        print("未找到有效聚类结果")
        return None, None
    else:
        print("=" * 30)
        print(f"最终最佳参数: {best_params}")
        print(f"最终最高分数: {best_score:.4f}")
        print("=" * 30)

        _df_all = pd.DataFrame({
            "id": xid,
            "comment": comment,
            # "cut_comment": cut_word,
            "cluster": best_labels
        })

        if cut_word is not None:
            _df_all["cut_comment"] = cut_word

    # 保存
    filename = f"{file_prefix}_最终聚类结果_{best_params['n_neighbors']}_{best_params['min_samples']}_{best_params['min_cluster_size']}.xlsx"
    _df_all.to_excel(filename, index=False)
    print("搜索完成")

    return best_params, _df_all


def get_hyper_parameter(n, is_first=True):
    # n_neighbors = range(max(5, int(n * 0.005)), min(150, max(15, int(n * 0.03))))
    # 发现确定步长有点麻烦 改为用变量存储
    nn_min = max(15, int(n * 0.01))
    nn_max = min(150, max(30, int(n * 0.05)))

    nn_step = max(1, (nn_max - nn_min) // 3)
    nn_list = list(range(nn_min, nn_max + 1, nn_step))

    if is_first:
        mcs_min = max(30, int(n * 0.02))
        mcs_max = max(80, int(n * 0.06))
    else:
        mcs_min = max(10, int(n * 0.01))
        mcs_max = max(20, int(n * 0.03))

    mcs_step = max(1, (mcs_max - mcs_min) // 3)
    mcs_list = list(range(mcs_min, mcs_max + 1, mcs_step))

    ms_list = []

    for mcs in mcs_list:
        ms_radical = max(2, int(mcs * 0.3))
        ms_conservative = max(2, int(mcs * 0.6))

        ms_list.extend([ms_radical, ms_conservative])

    ms_list = sorted(list(set(ms_list)))

    param_grid = {
        "n_neighbors": nn_list,
        "min_cluster_size": mcs_list,
        "min_samples": ms_list
    }
    print(f"参数范围确定 输入数量{n} 是否一阶 {is_first} 参数{param_grid}")

    return param_grid


def calc_semantic_dispersion(embeddings, df):
    print("计算语义离散值")

    results = []
    c_id = [c for c in df['cluster'].unique() if c != -1]
    for i in c_id:
        df_index = df.index[df['cluster'] == i].tolist()
        cluster_vectors = embeddings[df_index]
        centroid = np.mean(cluster_vectors, axis=0).reshape(1, -1)
        distances = cosine_distances(cluster_vectors, centroid)
        avg_distance = np.mean(distances)

        results.append({
            "聚类簇标签": f"Cluster {i}",
            "样本数量 (条)": len(df_index),
            "平均语义离散度": round(avg_distance, 4)
        })

    # 将结果转化为 DataFrame 方便展示成表格
    result_df = pd.DataFrame(results)
    print("各聚类簇语义离散度统计")
    print(result_df.sort_values(by="平均语义离散度", ascending=False).to_string(index=False))

    all_dispersions = result_df["平均语义离散度"].values
    mean_disp = np.mean(all_dispersions)
    std_disp = np.std(all_dispersions)
    threshold = mean_disp + std_disp

    print(f"所有簇离散度均值 (μ): {mean_disp:.4f}")
    print(f"所有簇离散度标准差 (σ): {std_disp:.4f}")
    print(f"二次聚类触发绝对阈值 (μ+σ): {threshold:.4f}")

    # 找出超过阈值的簇
    scattered_clusters = result_df[result_df["平均语义离散度"] > threshold]
    bad_clusters = []
    if not scattered_clusters.empty:
        bad_clusters = scattered_clusters["聚类簇标签"].tolist()
        print(f"{bad_clusters} 突破阈值 启动二次聚类！")
    else:
        print("无需二次聚类")

    print("=" * 60 + "\n")
    return result_df, bad_clusters
# 单组离散值计算
def get_dispersion(vectors):
    if len(vectors) == 0:
        return 0
    centroid = np.mean(vectors, axis=0).reshape(1, -1)
    distances = cosine_distances(vectors, centroid)
    return np.mean(distances)

# 画图
def draw_3d(embeddings, n_neighbors, best_labels, picture_name, file_name, cluster=None):
    print("\n正在生成3D图...")

    umap_all_model = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=3,
        min_dist=0.1,
        metric='cosine',
        random_state=42,
        n_jobs=1
    )
    # 重新对原始embeddings进行降维画图
    vis_embeddings = umap_all_model.fit_transform(embeddings)

    result_df = pd.DataFrame(vis_embeddings, columns=['x', 'y', 'z'])
    result_df['label'] = best_labels

    scatter3d = Scatter3D(init_opts=opts.InitOpts(width="1200px", height="800px", page_title="聚类分析"))

    unique_labels = sorted(result_df['label'].unique())

    # 随机颜色函数
    def get_random_color():
        return "#%06x" % random.randint(0, 0xFFFFFF)

    for label in unique_labels:
        safe_label = int(label)
        data = result_df[result_df['label'] == label][['x', 'y', 'z']].values.tolist()

        if safe_label == -1:
            series_name = "噪音 (Noise)"
            my_opacity = 0.1  # 噪音设得非常透明
            my_color = "#d3d3d3"  #
        else:
            if cluster == '3':
                series_name = f"C3_{safe_label}"
            elif cluster == '4':
                series_name = f"C4_{safe_label}"
            else:
                series_name = f"Cluster {safe_label}"
            my_opacity = 1.0
            my_color = get_random_color()

        print(f"  -> Cluster {safe_label}: {len(data)} 点, 颜色 {my_color}")

        scatter3d.add(
            series_name=series_name,
            data=data,
            xaxis3d_opts=opts.Axis3DOpts(name="X"),
            yaxis3d_opts=opts.Axis3DOpts(name="Y"),
            zaxis3d_opts=opts.Axis3DOpts(name="Z"),
            itemstyle_opts=opts.ItemStyleOpts(color=my_color, opacity=my_opacity)
        )

    scatter3d.set_global_opts(
        title_opts=opts.TitleOpts(title=picture_name),
        legend_opts=opts.LegendOpts(type_="scroll", pos_left="right", orient="vertical"),
        tooltip_opts=opts.TooltipOpts(is_show=True)
    )

    output_file = file_name
    scatter3d.render(output_file)

    print(f"成功，图{output_file}")

def update_sql(table_name, insert_data):
    update_data = insert_data[["comment", "final_cluster"]].values.tolist()
    sql_update = f"""INSERT INTO {table_name} (comment, final_cluster) VALUES (?, ?)"""

    try:
        cursor.executemany(sql_update, update_data)
        conn.commit()
        print("插入成功")
    except:
        cursor.rollback()
        print("更新失败")


if __name__ == '__main__':
    select_table = "豫西大峡谷清洗完毕"

    conn = connect_sql.Connect()
    cursor = conn.cursor()
    cursor.execute(f"select id, evaluation from {select_table} where date >= '2023-01-01' order by id")
    reviews = cursor.fetchall()

    # all_id = [review[0] for review in reviews]
    # all_comment = [review[1].replace(' ', '') for review in reviews]

    all_comment_cut_ok = []

    for review in reviews:
        if review[1]:
            all_comment_cut_ok.append({'id': review[0], 'comment': str(review[1]).replace(' ', '')})
    df_all_com = pd.DataFrame(all_comment_cut_ok)
    print(f"原始数据加载ok，总共{len(df_all_com)}条", "开始分句", sep='\n')

    # pmi挖掘词典
    raw_texts = df_all_com['comment'].tolist()
    pmi_miner = PMIMiner(min_freq=5, min_pmi=3.0)
    # dict_path  = os.path.join(os.path.dirname(__file__), 'dict.txt')
    dict_path = 'dict.txt'
    pmi_miner.build_dict(raw_texts, dict_path, 200)

    df_all_com['split_content'] = df_all_com['comment'].apply(split_comment)
    df = df_all_com.explode('split_content').reset_index(drop=True)
    df = df.rename(columns={'comment': 'origin_comment', 'split_content': 'comment'})
    df = df[df['comment'].notna() & (df['comment'] != '')]

    print(f"分句ok 样本量从{len(df_all_com)}变化到{len(df)}")

    # 全量分词 并入df
    jieba_tool = JieBaCut(dict_path=dict_path)
    df["cut_comment"] = jieba_tool.cut_text(df["comment"].tolist())

    # 统计高频词
    all_text = " ".join(df["cut_comment"])
    all_word = all_text.split()
    word_freq = Counter(all_word)
    top_words = word_freq.most_common(50)
    with open('top_words.txt', 'w', encoding='UTF-8') as f:
        for word, freq in top_words:
            f.write(f"{word}{freq}\n")
    print(top_words)

    # 剔除分词后为空的行
    # df_clean = df[df["cut_comment"].str.strip() != ""].copy()

    all_id = df['id'].tolist()
    all_comment: list[str] = df['comment'].tolist()
    all_cut_word = df['cut_comment'].tolist()

    model = SentenceTransformer("shibing624/text2vec-base-chinese", device="cuda")
    all_embeddings = model.encode(all_comment,
                                  batch_size=64,
                                  show_progress_bar=True)
    # 整体
    best_param_grid_all, new_df_all = cluster_main(all_embeddings, all_id, all_comment, all_cut_word, best_param=None)
    # best_param={
    #     "n_neighbors": 16,
    #     "min_cluster_size": 26,
    #     "min_samples": 3})

    top_keywords_per_cluster(new_df_all, n=10, max_df=0.6)

    semantic_dispersion_df, bad_clusters = calc_semantic_dispersion(all_embeddings, new_df_all)

    new_df_all['final_cluster'] = new_df_all['cluster'].apply(
        lambda x: f"c{x}" if x != -1 else "noise"
    )
    print(f"二次聚类标签{bad_clusters}")
    b_ids = [int(label.split(' ')[1]) for label in bad_clusters]

    for b_id in b_ids:
        b_index = new_df_all[new_df_all['cluster'] == b_id].index

        sub_embeddings = all_embeddings[b_index]
        sub_ids = new_df_all.loc[b_index, 'id'].tolist()
        sub_comments = new_df_all.loc[b_index, 'comment'].tolist()
        sub_cut_words = new_df_all.loc[b_index, 'cut_comment'].tolist()

        print(f"正在计算二层聚类cluster {b_id} 总数{len(sub_embeddings)}")

        sub_params, sub_df = cluster_main(
            sub_embeddings,
            sub_ids,
            sub_comments,
            sub_cut_words,
            is_first=False,
            best_param=None,
            file_prefix=f"{b_id}"
        )

        if sub_df is not None:
            new_labels = [f"c{b_id}_{label}" if label != -1 else f"{b_id}_noise" for label in sub_df['cluster']]
            top_keywords_per_cluster(sub_df, n=5, max_df=0.6)
            new_df_all.loc[b_index, 'final_cluster'] = new_labels
            print(f"cluster {b_id} 二次聚类完成 新标签进入总表")

            print(f"Cluster {b_id} 拆分前后离散度对比")

            dispersion_before = get_dispersion(sub_embeddings)
            total_samples = len(sub_embeddings)

            dispersion_after_weighted = 0
            valid_sub_clusters = [c for c in sub_df['cluster'].unique() if c != -1]

            print(f"拆分前母簇 c{b_id} 离散度 {dispersion_before:.4f}")
            print("拆分后各有效子簇离散度")

            for sc_id in valid_sub_clusters:
                sc_index = sub_df[sub_df['cluster'] == sc_id].index
                sc_vectors = sub_embeddings[sc_index]
                d = get_dispersion(sc_vectors)

                weight = len(sc_vectors) / total_samples
                dispersion_after_weighted += d * weight
                print(f" - c{b_id}_{sc_id} 离散度: {d:.4f} (样本量: {len(sc_vectors)})")

            if dispersion_before > 0:
                drop_rate = (dispersion_before - dispersion_after_weighted) / dispersion_before * 100
                print(f"剔除边缘噪音后，加权平均离散度降至{dispersion_after_weighted:.4f}")
                print(f"离散度下降: {drop_rate:.2f}%\n")

            draw_3d(sub_embeddings, sub_params['n_neighbors'], sub_df['cluster'].tolist(), " ", f"cluster{b_id}二次聚类图.html", f"{b_id}")
        else:
            print(f"cluster {b_id} 二次聚类未找到有效簇")

    print("二次聚类完毕", new_df_all['final_cluster'].value_counts(), sep='\n')

    new_df_all.to_excel("最终双层聚类.xlsx", index=False)
    # 发现c0很乱 对c0二次聚类 要获取两个返回值
    # df_c0 = new_df_all[new_df_all['cluster'] == 0].copy()
    #
    # embeddings_c0 = all_embeddings[df_c0.index]
    # c0_ids = df_c0['id'].tolist()
    # c0_comments = df_c0['comment'].tolist()
    # c0_cut_word = df_c0['cut_comment'].tolist()
    #
    # print(f"成功从原始矩阵中提取出 {len(embeddings_c0)} 条 Cluster 0 的向量。")
    # best_param_grid_c0, c0_df = cluster_main(embeddings_c0, c0_ids, c0_comments, c0_cut_word,
    #                                          best_param={
    #                                              "n_neighbors": 38,
    #                                              "min_cluster_size": 11,
    #                                              "min_samples": 3
    #                                                     })
    #
    # top_keywords_per_cluster(c0_df, n = 10, max_df = 0.6)
    #
    # print(best_param_grid_all, best_param_grid_c0, c0_df, sep='\n')

    # 调用画图函数
    draw_3d(all_embeddings, best_param_grid_all['n_neighbors'], new_df_all["cluster"].tolist(), "  ", "总体聚类图.html")
    # draw_3d(embeddings_c0, best_param_grid_c0['n_neighbors'], c0_df["cluster"].tolist(), " ", "C0聚类图.html")

    # 插入数据库
    # update_sql("聚类完毕", new_df_all)
    # update_sql("小聚类", c0_df)

    # 针对每个聚类单独情感分
    # sentiment = SA()
    # sentiment.analysis("小聚类")
    # sentiment.analysis("大聚类")
    # sentiment.analysis("聚类完毕")
    cursor.close()
    conn.close()
