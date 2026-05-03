import json
import csv
import os

def onRequest(context, request):
    return request

def onResponse(context, response):

    if "reviewlist" not in context.url:
        return response

    try:
        response.body.jsonify()
        data = response.body
        
        review_list = data['reviewInfo']['reviewListInfo']['reviewList']
        
        if not review_list:
            print("未发现评论数据")
            return response
            
        file_path = r"C:\jiasu\dp_reviews_result.csv"
        file_exists = os.path.exists(file_path)

        with open(file_path, mode='a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            
            if not file_exists:
                writer.writerow(['用户名', '时间', '评论内容'])

            count = 0
            for item in review_list:
                user = item.get('userNickName', '未知用户')
                add_time = item.get('addTime', '')
                
                review_text = ""
                review_body = item.get('reviewBody', {})
                body_children = review_body.get('children', [])
                
                if body_children:
                    inner_nodes = body_children[0].get('children', [])
                    text_parts = []
                    for node in inner_nodes:
                        if node.get('type') == 'text':
                            text_parts.append(node.get('text', ''))
                    review_text = "".join(text_parts).replace('\n', ' ').strip()

                writer.writerow([user, add_time, review_text])
                count += 1
        

    except Exception as e:
        print(f"脚本运行报错: {str(e)}")

    return response