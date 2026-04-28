from thesis.connect_sql import connect_sql
import re

conn = connect_sql.Connect()
cursor = conn.cursor()
sql = """SELECT name, evaluation, date FROM 豫西大峡谷简易去重
"""

html_pattern = r'<[^>]+>'
pattern = r"[^\u4e00-\u9fa50-9\s，。！？,.!?、]"
cursor.execute(sql)
data = cursor.fetchall()

for i in data:
    text = str(i[1])
    del_text = text.split("商家回复")[0]
    s1 = re.sub(html_pattern, '', del_text)
    s2 = re.sub(pattern, '', s1.replace('\xa0', ''))
    s3 = re.sub(r'\s+', '', s2).strip()
    cursor.execute(
        "INSERT INTO 豫西大峡谷字符清洗(name, evaluation, date) VALUES (?, ?, ?)",
        (i[0], s3, i[2])
)
cursor.commit()
cursor.close()
conn.close()