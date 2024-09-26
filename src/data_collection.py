import datetime
import tensorflow as tf
from newsapi import NewsApiClient
from transformers import pipeline
import re
import pandas as pd
import yfinance as yf
from src.data_preprocessing import preprocess_data
def clean_text(text):
    text = re.sub(r'\n', ' ', text)
    text=re.sub('https?:\/\/\S+','',text)

    return text

def find_sentiment(text):
    model_path='mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis'
    sentiment_pipeline = pipeline("sentiment-analysis",model=model_path, tokenizer=model_path, max_length=512, truncation=True,device=0)
    sent = sentiment_pipeline(text)[0]
    score = sent["score"]
    if sent["label"]== "positive":
      score=0.5* score + 0.5
    elif sent["label"] == "negative":
        score=-0.5 * score - 0.5
    elif sent["label"]=="neutral":
      score=score-0.5
    return score

def run(coin_name,coin_id,currency):
  newsapi = NewsApiClient(api_key='fac9c84381f64481ab2732c2a5fd1f88')
  from_date=datetime.date.today()
  to_date=from_date-datetime.timedelta(days=60)
  merged_articles=[]


  for i in range (1,5):
      req1 = newsapi.get_everything(q=coin_name,
                                      sources="the-hindu,the-times-of-india,",
                                        from_param=from_date,
                                        to=to_date,
                                        language='en',
                                        sort_by='publishedAt',
                                        page=i)
      req2 = newsapi.get_everything(q=coin_name,
                                      domains="coindesk.com",
                                        from_param=from_date,
                                        to=to_date,
                                        language='en',
                                        sort_by='publishedAt',
                                        page=i)
      req3 = newsapi.get_everything(q=coin_name,
                                      domains="cointelegraph.com",
                                        from_param=from_date,
                                        to=to_date,
                                        language='en',
                                        sort_by='publishedAt',
                                        page=1)
      req1=[{'date':datetime.datetime.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%SZ").strftime('%Y-%m-%d'),'news':article['description']} for article in req1['articles']]
      req2=[{'date':datetime.datetime.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%SZ").strftime('%Y-%m-%d'),'news':article['description']} for article in req2['articles']]
      req3=[{'date':datetime.datetime.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%SZ").strftime('%Y-%m-%d'),'news':article['description']} for article in req3['articles']]
      merged_articles.extend(req1)
      merged_articles.extend(req2)
      merged_articles.extend(req3)



  for articles in merged_articles:
      articles['news'] = clean_text(articles['news'])

  grouped_news = {}
  for obj in merged_articles:
      date = obj['date']
      news = obj['news']
      if date in grouped_news:
          grouped_news[date] += ' ' + news
      else:
          grouped_news[date] = news
  merged_articles = [{'date': date, 'news': news_list} for date, news_list in grouped_news.items()]

  existing_dates = {datetime.datetime.strptime(item['date'], "%Y-%m-%d").date() for item in merged_articles}
  date_range = [to_date + datetime.timedelta(days=i) for i in range((from_date - to_date).days )]
  for date in date_range:
      if date not in existing_dates:
          merged_articles.append({
              'date': date.strftime("%Y-%m-%d"),
              'news': 'No significant news for this day'
          })


  for article in merged_articles:
    content=article.get("news", "")
    article['sentiment']=find_sentiment(content)

  edits_df = pd.DataFrame.from_dict(merged_articles)
  edits_df=edits_df.sort_values(by=['date'], ascending=False)
  edits_df = edits_df.reset_index(drop=True)
  edits_df=edits_df.drop('news',axis=1)
  edits_df.index=pd.to_datetime(edits_df.index)
  edits_df = edits_df.set_index('date')
  edits_df.index=pd.to_datetime(edits_df.index)
  start_date = edits_df.index.min()
  end_date = edits_df.index.max()
  date_range = pd.date_range(start=start_date, end=end_date, freq='D')
  edits_df=edits_df.reindex(date_range, fill_value=pd.NA).ffill()
  #######################################################
  days="3mo"
  id=coin_id
  currency=currency
  btc_ticker = yf.Ticker(f"{id}-{currency}")
  btc = btc_ticker.history(period=days)
  btc.index=btc.index.strftime('%Y-%m-%d')
  btc.index = pd.to_datetime(btc.index)
  del btc["Stock Splits"]
  del btc["Dividends"]
  del btc['Volume']
  merged_df=pd.merge(edits_df, btc, how='inner', left_index=True, right_index=True)
  return merged_df

# ans=run('bitcoin','BTC','USD')
# ans,scaler=preprocess_data(ans)
# model = tf.keras.models.load_model("Bitcoin_prediction_model_better2/model.h5")
# output=model.predict(ans)
# output=output.reshape(30,5)
# output=scaler.inverse_transform(output)
# print(output)
