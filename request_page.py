"""
发送请求,返回html
"""
from bs4 import BeautifulSoup as Bs
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options

class XieChengSpider:
    def __init__(self,url):
        self.time = 2
        print("启动！")
        edge_options = Options()
        edge_options.add_argument("--headless=new")
        self.browser = webdriver.Edge(options=edge_options)
        self.browser.get(url)

        print("休息2秒，等待页面加载")

        time.sleep(self.time)
        self.html = self.browser.page_source

        if self.html is None:
            print("获取失败！！！")
        else:
            print("获取完毕")

    def click_next_page(self):
        button = self.browser.find_element(By.XPATH, '//span[@class="ant-pagination-item-comment"]/a[text()="下一页"]')
        self.browser.execute_script("arguments[0].click();", button)
        print("点击下一页")

class AyHtml:
    def __init__(self, file_path):
        self.result = Bs(file_path, 'html.parser')
        if self.result is None:
            print("解析失败")
        else:
            print("解析成功")

    def finder(self,path,class_):
        fund = self.result.find(path,class_)
        return fund

if __name__ == "__main__":
    browser = XieChengSpider(r"https://you.ctrip.com/sight/lushi2273/61660.html")
    content = AyHtml(browser.html)
    browser.click_next_page()
