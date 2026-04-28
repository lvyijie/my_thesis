from transformers import pipeline
from thesis.connect_sql import connect_sql


class SA:
    def __init__(self):
        # 创建数据库对象
        self.conn = connect_sql.Connect()
        # 用于情感分析的模型
        self.classifier = pipeline(
            "text-classification",
            # model="uer/roberta-base-finetuned-jd-binary-chinese", # 二分类模型
            # tokenizer="uer/roberta-base-finetuned-jd-binary-chinese",

            # model="nlptown/bert-base-multilingual-uncased-sentiment", # 五分类模型
            # tokenizer="nlptown/bert-base-multilingual-uncased-sentiment",

            model=r"C:\Users\qq150\source\repos\NewRepo\thesis\model\tourism_sentiment_model", # 微调过的本地三分类模型
            tokenizer=r"C:\Users\qq150\source\repos\NewRepo\thesis\model\tourism_sentiment_model",
            device=0,  # 调用GPU用0
            batch_size=64,
            truncation=True,
            max_length=512
        )

        self.count = 0 # 记数
        self.batch_size = 200 # 每一批次中包含的数据量
        self.cursor = self.conn.cursor()

    def analysis(self, select_table):
        while True:
            sql = f"""
            select id, comment from {select_table}
            where id > {self.count} * {self.batch_size} and id <= ({self.count} + 1) * {self.batch_size}
            order by id
            """

            self.cursor.execute(sql)
            reviews = self.cursor.fetchall()

            if not reviews:
                break

            print("第%d批次数据读取完毕" % (self.count + 1))

            comment = [review[1] for review in reviews]

            self.count += 1

            results = self.classifier(comment, batch_size=4)

            sql = f"""UPDATE {select_table} SET confidence_score = ?, sentiment_label = ? WHERE id = ?"""

            for review,result in zip(reviews,results):
                sentiment_label = result["label"]
                score = result["score"]
                # print(sentiment_label, score)
                self.cursor.execute(sql, (score, sentiment_label, review[0]))
            self.cursor.commit()

if __name__ == '__main__':
    A = SA()
    A.analysis("聚类完毕")

