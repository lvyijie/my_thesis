import re
import csv
import os

def onRequest(context, request):
    return request

def onResponse(context, response):
    try:
        try:
            html_content = bytes(response.body).decode('utf-8', errors='ignore')
        except Exception:

            html_content = str(response.body)

        cards = html_content.split('<div class="feedbackCard">')[1:]
        
        if not cards:
            print("页面未找到评论数据")
            return response

        file_path = r"C:\jiasu\meituan_reviews.csv"
        file_exists = os.path.exists(file_path)

        with open(file_path, mode='a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['用户名', '时间', '评论内容'])

            count = 0
            for card in cards:
            
                u_match = re.search(r'<weak class="username">(.*?)</weak>', card)
                username = u_match.group(1).strip() if u_match else "未知用户"

                t_match = re.search(r'<weak class="time">(.*?)</weak>', card)
                time = t_match.group(1).strip() if t_match else ""

                c_match = re.search(r'<div class="comment">(.*?)</dd>', card, re.S)
                if c_match:
                    raw_content = c_match.group(1)
                    clean_text = re.sub(r'<.*?>', ' ', raw_content)
                    clean_content = re.sub(r'\s+', ' ', clean_text).strip()
                else:
                    clean_content = ""

                if username != "未知用户":
                    writer.writerow([username, time, clean_content])
                    count += 1

        print(f"成功从 HTML 提取 {count} 条评论 -> {file_path}")

    except Exception as e:
        print(f"解析 HTML 出错: {str(e)}")

    return response