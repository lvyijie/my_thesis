# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : jieba_cut_test.py
# Time       ：2025/11/5 22:20
# Author     ：lv
# version    ：python 3.13
# Description：
"""
import re
import os
import jieba


class JieBaCut:
    def __init__(self, dict_path=None):
        if dict_path and os.path.exists(dict_path):
            jieba.load_userdict(dict_path)
            print(f"成功加载动态领域词典 {dict_path}")
        # jieba切词太碎 加入自定义词
        custom_add_words = [
            "双龙湾", "一个半小时", "十多公里", "真气人", "别人拿盆", "听说免费", "说免票"
        ]
        for word in custom_add_words:
            jieba.add_word(word)
        # 导入停用词表
        additional_stop_words = [
            "没有", "地方", "特别", "非常", "比较", "感觉", "真的",
            "很多", "不是", "里面", "大家", "一次", "有点",
            "需要", "觉得", "一起", "不能", "一定", "只能", "才能", "已经", "之后", "大概", "起来", "再也", "继续",
            "直接", "不用", "回到", "想着", "一下", "来说", "要说", "接纳",
            "满足", "设有", "提供", "可能", "搭配", "看也不看", "三个", "买个",
            "不到", "多个", "不同", "别人", "般的", "真体", "盆才",
            "非常", "特别", "确实", "真的", "感觉", "觉得", "比较", "挺", "就",
            "的", "了", "在", "是", "我", "有", "和", "也", "很", "到", "说",
            "要", "去", "会", "着", "没有", "看", "好", "自己", "这", "那", "就", "都", "还",
            "位于", "境内", "一幅", "喜欢", "及时", "回家", "洗完", "过程",
            "提前", "订好", "出发", "居然", "当时", "挺大", "很方便", "时间",
            "总体", "无论是", "支持", "操作", "身上", "出来", "准备", "相对",
            "下去", "不让", "不要", "忘记", "着实", "机会", "安排", "希望",
            "过来", "感受", "今天", "本来", "漂完", "百万种", "千变万化",
            "人生", "需求", "玩意", "好看", "说免", "真气", "多公里", "一个半", "有如",
            '知道', '先说', '随便', '游玩', '全程', '小时', '公里', '两个小时', '分钟', '小时左右',
            '推荐', '值得', '适合', '豫西大峡谷', '卢氏县', '三门峡', '峡谷', '漂流', '景区',
            '官道口镇', '豫西', '三门峡市', '河南省', '看到', '还要',' 一个', '一点'
        ]
        with open(r'C:\Users\qq150\source\repos\NewRepo\thesis\cluster_analysis\stopwords_hit.txt', 'r',
                  encoding='UTF-8') as f:
            self.stop_words = set(line.strip() for line in f.readlines())
            self.stop_words.update(additional_stop_words)

    # 定义函数使用正则表达式去除多余符号
    def clean_text(self, text) -> str:
        cleaned_text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s]", "", text)  # 去符号
        cleaned_text = re.sub(r"\s+", " ", cleaned_text.strip())
        cleaned_text = re.sub(r"\d", "", cleaned_text)
        return cleaned_text

    def cut_text(self, texts) -> list:
        texts_cut: list[str] = []
        for i in texts:
            all_word = jieba.lcut(self.clean_text(i))
            words = [w for w in all_word if w not in self.stop_words and len(w) > 1]
            texts_cut.append(' '.join(words))
        return texts_cut
