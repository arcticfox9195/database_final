import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
import sqlite3
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random
from random import randint
import pandas as pd
from datetime import datetime
import nltk.corpus
from nltk.corpus import stopwords
import pickle
import re
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.font_manager


def create_table(conn, table_name):
    try:
        table_name = table_name.replace(" ", "_")
        cur.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                商品名稱 TEXT,
                網址 TEXT,
                商品定價 REAL,
                星星評分 TEXT,
                全球評分數量 INTEGER,
                商品描述 TEXT,
                產品資訊 TEXT,
                全球排名 TEXT,
                留言網址 TEXT
            )
        ''')
        print(f"Table {table_name} created successfully.")
    except sqlite3.Error as e:
        print("Error creating table:", e)


def search_amazon(product_name):
    service = ChromeService(executable_path=ChromeDriverManager().install())
    chrome_options = webdriver.ChromeOptions()
    prefs = {"profile.default_content_setting_values.notifications" : 2}
    chrome_options.add_experimental_option("prefs",prefs)

    driver = webdriver.Chrome(service=service, options=chrome_options)
    time.sleep(5)

    theurl = []
    for i in range(5):
        # 去到你想要的網頁
        driver.get("https://www.amazon.com/s?k="+ product_name +"&page="+ str(i) +"ref=sr_pg_2")
        
        geturl = driver.find_elements(by=By.XPATH, value='//h2/a')

        for j in geturl:
            theurl.append(j.get_attribute('href'))
            
        time.sleep(5)
    title = []
    url = []
    price = []
    star = []
    starNum = []
    description = []
    productDscrp = []
    global_range = []
    view_url = []
    print(len(theurl))
    for page in range(0,len(theurl)):
        print('第 '+ str(page) + ' 個商品')
        #儲存網址
        url.append(theurl[page])
        
        # 去到你想要的網頁
        driver.get(theurl[page])
        time.sleep(randint( 7, 15))
        
        # 商品名稱
        title.append(driver.find_element(by=By.ID, value='title').text) 
        
        # 商品定價
        try:
            if len(driver.find_elements(by=By.ID, value='corePriceDisplay_desktop_feature_div'))==0:
                getprice = driver.find_element(by=By.ID, value='corePrice_desktop').text
            else:
                getprice = driver.find_element(by=By.ID, value='corePriceDisplay_desktop_feature_div').text
            
            getprice = getprice.replace('US$','') # 先把「US$」拿掉
            if '有了交易' in getprice:
                getprice = getprice[getprice.find('有了交易')+6:]
                getprice = getprice.split('\n')[0]
            elif '\n定價:\n' in getprice:
                getprice = getprice[getprice.find('\n')+1:getprice.find('\n定價:\n')]
                getprice = getprice.replace('\n','.')
            else:
                
                getprice = getprice.replace('定價：','') # 把「定價」拿掉
                if ' -' in getprice: # 利用「 - 」來切割兩個數字
                    getprice = getprice.replace('\n','') # 把「US$」拿掉
                    cutlist = getprice.split(' -')
                    getprice = (float(cutlist[0]) + float(cutlist[1]))/2 # 計算平均
                else:
                    getprice = getprice.replace('\n','.')
            price.append(getprice)
        except:
            price.append("can't get price")
        
        # 星星評分
        if len(driver.find_elements(by=By.ID, value='acrPopover'))==0:
            star.append('沒有星等')
        else:
            star.append(driver.find_element(by=By.ID, value='acrPopover').get_attribute("title").replace(' 顆星，最高 5 顆星',''))
        # 全球評分數量
        if len(driver.find_elements(by=By.ID, value='acrCustomerReviewText'))==0:
            starNum.append(0)
        else:
            getglobalNum = driver.find_element(by=By.ID, value='acrCustomerReviewText').text
            getglobalNum = getglobalNum.replace('等級','')
            getglobalNum = getglobalNum.replace(',','')
            starNum.append(getglobalNum)
        
        
        # 商品描述
        if len(driver.find_elements(by=By.ID, value='productDescription')) != 0:
            productDscrp.append(driver.find_element(by=By.ID, value='productDescription').text)
        else:
            productDscrp.append('')
        
        # 產品詳細資訊

        try:
            description.append(driver.find_element(by=By.ID, value='productDetails_feature_div').text)
        except:
            description.append("can't find description ")
        # 全球排名
            
        try:
            getdata = driver.find_element(by=By.XPATH, value='//div[@class = "a-section table-padding"]/table[@id = "productDetails_detailBullets_sections1"]/tbody/tr/td/span/span').text
            print(getdata)
            #getdata = getdata.replace('暢銷商品排名: ','')
            # getdata = getdata.replace('\n','')
            getdata = getdata.split('#')
            containar = {}
            for i in range(1,len(getdata)):
                rang = getdata[i].split(' 在 ')[0]
                item = getdata[i].split(' 在 ')[1]
                if ' (' in item:
                    item = item.split(' (')[0]
                containar[item] = int(rang.replace(',',''))
            global_range.append(str(containar))
        except:
            global_range.append("error")
        
        # 留言網址
        if len(driver.find_elements(by=By.XPATH, value='//a[@data-hook = "see-all-reviews-link-foot"]'))== 0 :
            view_url.append('沒有留言')
        else:
            view_url.append(driver.find_element(by=By.XPATH, value='//a[@data-hook = "see-all-reviews-link-foot"]').get_attribute('href'))
        
        dic = {
                '商品名稱' : title,
                '網址' : url,
                '商品定價' : price,
                '星星評分' : star,
                '全球評分數量' : starNum,
                '商品描述' : description,
                '產品資訊' : productDscrp,
                '全球排名' : global_range,
                '留言網址' : view_url
            }
            
        df = pd.DataFrame(dic)
        df.to_sql(product_name, conn, index=False, if_exists='append')
    return 0

# Step 1: 使用網路爬蟲獲取商品評論資料

def scrape_amazon_reviews_with_driver(product_name):
    query = f"SELECT 商品名稱, 留言網址 FROM {product_name} WHERE 留言網址 != '沒有留言'"
    df_from_db = pd.read_sql(query, conn)
    df_from_db = df_from_db.drop_duplicates(subset=['留言網址'])
    product_names = df_from_db['商品名稱'].tolist()
    comment_urls = df_from_db['留言網址'].tolist()

    theproduct = []
    theCommenturl = []
    who = []
    star = []
    thetime = []
    location = []
    comment = []
    helpful = []

    # 使用WebDriver
    service = ChromeService(executable_path=ChromeDriverManager().install())
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless")  # 在後台運行，無需打開實際瀏覽器
    driver = webdriver.Chrome(service=service, options=chrome_options)

    for data in range(len(comment_urls)):
        geturl = comment_urls[data]
        doit = True
        page = 0

        while doit:
            if page == 0:
                url = geturl
            else:
                url = geturl.split('/ref')[0] + '/ref=cm_cr_getr_d_paging_btm_next_' + str(
                    page) + '?ie=UTF8&reviewerType=all_reviews&pageNumber=' + str(page)

            driver.get(url)

            # 使用WebDriverWait等待評論區塊的出現
            try:
                WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '//div[@data-hook="review"]'))
                )
            except Exception as e:
                print(f"等待評論區塊時發生錯誤: {e}")

            # 此處繼續你的爬蟲操作
            soup = BeautifulSoup(driver.page_source, "html.parser")
            getdata = soup.find_all("div", attrs={"data-hook": "review"})

            if len(getdata) > 0:
                for i in getdata:
                    theproduct.append(product_names[data]) # 儲存商品名稱
                    theCommenturl.append(comment_urls[data]) # 儲存留言網址
                    
                    who.append(i.find('span', {'class':'a-profile-name'}).text) # 儲存留言者
                 
                    # 處理星星
                    getstart = i.find('span', {'class':'a-icon-alt'}).text
                    getstart = getstart.replace(' 顆星，最高 5 顆星','') # 中文網頁
                    getstart = getstart.replace(' out of 5 stars','') # 英文網頁
                    star.append(float(getstart))
              
                    
                    # 處理購買時間、地點
                    gettime = i.find('span', {'data-hook':'review-date'}).text
                    if 'Reviewed' in gettime: # 判斷是否為英文網頁
                        # 將英文月份換成數字，這樣待會才能給datetime辨別
                        gettime = gettime.replace('January','1')
                        gettime = gettime.replace('February','2')
                        gettime = gettime.replace('March','3')
                        gettime = gettime.replace('April','4')
                        gettime = gettime.replace('May','5')
                        gettime = gettime.replace('June','6')
                        gettime = gettime.replace('July','7')
                        gettime = gettime.replace('August','8')
                        gettime = gettime.replace('September','9')
                        gettime = gettime.replace('October','10')
                        gettime = gettime.replace('November','11')
                        gettime = gettime.replace('December','12')
                        
                        gettime_list = gettime.split(' on ')
                        thetime.append(datetime.strptime(gettime_list[1], "%m %d, %Y")) # 儲存留言時間
                        location.append(gettime_list[0].replace('Reviewed in the ','')) # 儲存留言地點
                    else:
                        if '於' in gettime: # 有時會出現不同呈現字串，範例:'在 2022年7月10日 於瑞典評論'
                            gettime = gettime.replace('在 ','')
                            gettime_list = gettime.split('於')
                        else:
                            gettime_list = gettime.split('在')
                        cuttime = gettime_list[0].replace(' ','')
                        thetime.append(datetime.strptime(cuttime, "%Y年%m月%d日")) # 儲存留言時間
                        location.append(gettime_list[1].replace('評論','')) # 儲存留言地點
                    
                    comment.append(i.find('span', {'data-hook':'review-body'}).text) # 儲存留言內容
           
                    # 處理覺得留言有用人數
                    gethelpful = i.findAll('span', {'data-hook':'helpful-vote-statement'}) # 儲存覺得留言有用人數
                    if len(gethelpful) != 0: # 判斷是否有資料
                        
                        gethelpful = gethelpful[0].text
                        gethelpful = gethelpful.replace(',','') # 把千分位的「,」拿掉
                        gethelpful = gethelpful.replace(' 個人覺得有用','') # 中文網頁
                        gethelpful = gethelpful.replace(' people found this helpful','') # 英文網頁
                        if '一人覺得有用' == gethelpful or 'One person found this helpful' == gethelpful: # 判斷是否只有一人
                            helpful.append(1)
                        else:
                            helpful.append(int(gethelpful))
                    else:
                        helpful.append(0)

            else:
                doit = False
            print('累計資料數量： ' + str(len(who)))
            page = page + 1
            time.sleep(randint(5, 20))

        print("第" + str(data) + '筆' + product_names[data] + '執行完畢')
        
    # 關閉WebDriver
    driver.quit()
    dic = {
        '商品名稱': theproduct,
        '留言網址': theCommenturl,
        '留言者': who,
        '星等': star,
        '留言時間': thetime,
        '留言地點': location,
        '留言內容': comment,
        '覺得留言有用人數': helpful,
        }
    df = pd.DataFrame(dic)
    df.to_sql(product_name + '_reviews', conn, index=False, if_exists='append')

    return 0


def preprocess_text(text, stop):
    words = nltk.word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop]
    return " ".join(filtered_words)

def generate_wordcloud(coefficients, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(coefficients)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()


# Step 2: 機器學習分析
def analyze_sentiment(product_name):
    nltk.download('punkt')
    nltk.download("stopwords")
    stop = stopwords.words("english")
    selected_columns = ['留言內容', '星等', '覺得留言有用人數']
    df = pd.read_sql(f"SELECT {', '.join(selected_columns)} FROM {product_name}_reviews", conn)
    df = df.drop_duplicates(subset='留言內容')
    df['processed_comment'] = df['留言內容'].apply(lambda x: preprocess_text(x, stop))
    df['星等_numeric'] = df['星等'].astype(float)

    tv = TfidfVectorizer(
        ngram_range=(1, 1),
        max_features=400
    )
    tfidf_matrix = tv.fit_transform(df['processed_comment'])

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tv.get_feature_names_out())
    print("TF-IDF Score:")
    print(tfidf_df)
    tfidf_matrix = tv.fit_transform(df['processed_comment'])
    
    # Create a tuple target variable (星等, 覺得留言有用人數)
    df['target_star'] = df['星等_numeric']
    df['target_votes'] = df['覺得留言有用人數']
    
    X = tfidf_matrix
    y_star = df['target_star']
    y_votes = df['target_votes']
    feature_names = tv.get_feature_names_out()
    # 存儲 TF-IDF 字典到 pickle 文件
    pickle.dump(tv.vocabulary_, open("TFIDF_feature.pkl", "wb"))

    param_dist = {
        'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }


    # 使用 Random Forest 進行星等的預測
    rf_star_model = RandomForestClassifier(random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rf_star_random = RandomizedSearchCV(
        estimator=rf_star_model,
        param_distributions=param_dist,
        n_iter=100,
        cv=kf,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    rf_star_random.fit(X, y_star)
    
    # 使用 Random Forest 進行覺得留言有用人數的預測
    rf_votes_model = RandomForestRegressor(random_state=42)
    rf_votes_random = RandomizedSearchCV(
        estimator=rf_votes_model,
        param_distributions=param_dist,
        n_iter=100,
        cv=kf,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    rf_votes_random.fit(X, y_votes)
    
    if hasattr(rf_star_random, 'best_estimator_') and hasattr(rf_votes_random, 'best_estimator_'):
        # 取得最佳模型
        best_rf_star_model = rf_star_random.best_estimator_
        best_rf_votes_model = rf_votes_random.best_estimator_

        # 預測星等
        df['predicted_star'] = best_rf_star_model.predict(tfidf_matrix)
        
        # 預測覺得留言有用人數
        df['predicted_votes'] = best_rf_votes_model.predict(tfidf_matrix)

        # 計算準確度（對於星等，使用準確度；對於覺得留言有用人數，使用 MSE）
        accuracy_star = accuracy_score(df['target_star'], df['predicted_star'])
        mse_votes = mean_squared_error(df['target_votes'], df['predicted_votes'])
        
        print(f"Accuracy for 星等: {accuracy_star * 100:.2f}%")
        print(f"Mean Squared Error for 覺得留言有用人數: {mse_votes}")

        # Feature importance for 星等
        feature_importance_star = best_rf_star_model.feature_importances_
        importances_df_star = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance_star})
        importances_df_star = importances_df_star.sort_values(by='Importance', ascending=False)
        print("Feature Importance for 星等:")
        print(importances_df_star)
        
        # Feature importance for 覺得留言有用人數
        feature_importance_votes = best_rf_votes_model.feature_importances_
        importances_df_votes = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance_votes})
        importances_df_votes = importances_df_votes.sort_values(by='Importance', ascending=False)
        print("Feature Importance for 覺得留言有用人數:")
        print(importances_df_votes)

        # Generate word cloud for 星等 feature importance
        wordcloud_weights_star = dict(zip(importances_df_star['Feature'], importances_df_star['Importance']))
        generate_wordcloud(wordcloud_weights_star, "Impact_Star")
        
        # Generate word cloud for 覺得留言有用人數 feature importance
        wordcloud_weights_votes = dict(zip(importances_df_votes['Feature'], importances_df_votes['Importance']))
        generate_wordcloud(wordcloud_weights_votes, "Impact_Votes")
        combined_feature_importance = feature_importance_star * feature_importance_votes
        wordcloud_weights = dict(zip(feature_names, combined_feature_importance))
        wordcloud_weights = {k: v * 100 for k, v in wordcloud_weights.items()}
        generate_wordcloud(wordcloud_weights, "Combined Impact")
    else:
        print("RandomizedSearchCV failed to fit the models.")
    return 0



if __name__ == "__main__":
    conn = sqlite3.connect(r"C:\Users\yaochiliao\Desktop\VSCODE\資料庫程式設計\期末專題\final.db")
    cur = conn.cursor()
    product_name = input("what kind of products are you searching for: ")
    table_name = product_name.replace(' ', '_').lower()
    command = int(input("what command you want to do: "))
    if command == 0:
        create_table(conn, table_name)
    elif command == 1:
        search_amazon(product_name)
    elif command == 2:
        scrape_amazon_reviews_with_driver(product_name)
    elif command == 3:
        analyze_sentiment(product_name)
    conn.commit()
    conn.close()