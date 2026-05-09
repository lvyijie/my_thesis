"""
主文件
"""
import request_page as req
import time
from thesis.connect_sql import connect_sql

url = r"https://you.ctrip.com/sight/lushi2273/61660.html"

# browser = req.AutoControl(r"https://you.ctrip.com/sight/lushi2273/61660.html")

browser = req.XieChengSpider(url)
content = req.AyHtml(browser.html)

all_user_name = []
all_score = []
all_commentDetail = []
all_commentTime = []

# comment_list = content.finder('div', "commentList")
page_list = content.finder('ul', "ant-pagination")
total_page = page_list.find_all('li')[-3].find('a').get_text(strip=True)
count = 1
for page in range(int(total_page)):
    print(f"第{count}页")
    comment_list = content.finder('div', "commentList")
    page_list = content.finder('ul', "ant-pagination")
    for comment in comment_list.find_all('div', class_='commentItem'):
        user_name = comment.find('div', class_='userName').get_text(strip=True)
        score = comment.find('span', class_='averageScore').get_text(strip=True)
        commentDetail = comment.find('div', class_='commentDetail').get_text(strip=True).replace('\n' or '\r', '')
        commentTime = comment.find('div', class_='commentTime').find_all(string=True,recursive=False)
        all_user_name.append(user_name)
        all_score.append(score)
        all_commentDetail.append(commentDetail)
        all_commentTime.append(commentTime)
    browser.click_next_page()
    time.sleep(3)
    print("等待页面加载")
    new_page = browser.browser.page_source
    content = req.AyHtml(new_page)
    count+=1

conn = connect_sql.Connect()
cursor = conn.cursor()
sql = """
      INSERT INTO 豫西大峡谷(name, evaluation, date)
      values (?, ?, ?)
      """
new_date = []
for i in range(len(all_user_name)):
    new_date.append(str(all_commentTime[i]).replace("[", '').replace("]", '').replace("'", ''))


for i in range(len(all_user_name)):
    if all_user_name[i] is None or all_user_name[i] == ' ':
        all_user_name[i] = "匿名用户"
    cursor.execute(sql, all_user_name[i], int(all_score[i][0]), all_commentDetail[i], new_date[i])

conn.commit()
cursor.close()
conn.close()