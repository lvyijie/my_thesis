import math
import jieba
import os
from collections import defaultdict

class PMIMiner:
    def __init__(self, min_freq = 5, min_pmi = 3.0):
        self.min_freq = min_freq
        self.min_pmi = min_pmi
    
    def build_dict(self, texts, s_path = 'dict.txt', top_n = 200):
        word_freq = defaultdict(int)
        bigram_freq = defaultdict(int)
        total_words = 0
        bigram_total = 0

        print(f"进行pmi挖掘，{len(texts)}条数据")

        for t in texts:
            clean_text = "".join([i for i in str(t) if '\u4e00' <= i <= '\u9fa5'])

            if not clean_text:
                continue
            
            words = list(jieba.cut(clean_text))

            for i, w in enumerate(words):
                if len(w) > 0:
                    word_freq[w] += 1
                    total_words += 1
                
                if i < len(words) - 1:
                    w1, w2 = words[i], words[i + 1]
                    if len(w1) > 0 and len(w2) > 0:
                        bigram_freq[(w1, w2)] += 1
                        bigram_total += 1 
        
        print("计算pmi值")
        phrases = []
        
        for (w1, w2), freq in bigram_freq.items():
            if freq < self.min_freq:
                continue
            
            p1 = word_freq[w1] / total_words
            p2 = word_freq[w2] / total_words
            p1_2 = freq / bigram_total

            pmi = math.log2(p1_2 / (p1 * p2))
            phrase_text = f"{w1}{w2}"
            phrase_len = len(phrase_text)

            bad_edges = {'则', '及', '上', '的', '了', '啊', '等', '由', '和', '与', '在', '个', '是', '象', '刚', '没', '过', '就', '要'}

            if pmi > self.min_pmi and (len(w1) > 1 or len(w2) > 1):
                if 2 <= phrase_len <= 6:
                    if phrase_text[0] not in bad_edges and phrase_text[-1] not in bad_edges:
                        phrases.append((f"{w1}{w2}", freq, pmi))

        phrases.sort(key = lambda x: x[2], reverse= True)
        top_phrases = phrases[:top_n]

        print(f"pmi挖掘完成 保存top{top_n}")

        with open(s_path, 'w', encoding = 'UTF-8') as f:
            for phrases, freq, pmi in top_phrases:
                f.write(f"{phrases} {freq}\n")
        
        print("词典生成成功")
        return s_path