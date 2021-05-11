# 소설 작가 분류 AI 경진대회 코드분석

## 라이브러리 import 및 설정


```python
%reload_ext autoreload
%autoreload 2
%matplotlib inline
```


```python
from matplotlib import pyplot as plt
from matplotlib import rcParams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
import seaborn as sns
import warnings
```


```python
rcParams['figure.figsize'] = (16, 8)
plt.style.use('fivethirtyeight')
pd.set_option('max_columns', 100)
pd.set_option("display.precision", 4)
warnings.simplefilter('ignore')
```

### 사용할 데이터 로드


```python
trn_file = pd.read_csv('train.csv')
tst_file = pd.read_csv('test_x.csv') 
sample_file = pd.read_csv('sample_submission.csv') 

target_col = 'author'
n_fold = 5
n_class = 5
seed = 42
```


```python
algo_name = 'lr'
feature_name = 'tfidf'
model_name = f'{algo_name}_{feature_name}'
```


```python
feature_file =  f'{feature_name}.csv'
p_val_file =  f'{model_name}.val.csv'
p_tst_file =  f'{model_name}.tst.csv'
sub_file = f'{model_name}.csv'
```

### 불러온 데이터 확인하기 


```python
trn_file.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>text</th>
      <th>author</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>He was almost choking. There was so much, so m...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>“Your sister asked for it, I suppose?”</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>She was engaged one day as she walked, in per...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>The captain was in the porch, keeping himself ...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>“Have mercy, gentlemen!” odin flung up his han...</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(trn_file.shape)
```

    (54879, 3)
    


```python
tst_file.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>“Not at all. I think she is one of the most ch...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>"No," replied he, with sudden consciousness, "...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>As the lady had stated her intention of scream...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>“And then suddenly in the silence I heard a so...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>His conviction remained unchanged. So far as I...</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(tst_file.shape)
```

    (19617, 2)
    

### NLTK 예시


```python
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer 
from nltk.stem.snowball import SnowballStemmer
```


```python
s = trn_file.text[4] #문장 설정하기 
print(s)
```

    “Have mercy, gentlemen!” odin flung up his hands. “Don’t write that, anyway; have some shame. Here I’ve torn my heart asunder before you, and you seize the opportunity and are fingering the wounds in both halves.... Oh, my God!”
    


```python
tokens = word_tokenize(s) #저장한 문장에 대한 단어 토큰화 실시 
print(tokens)
```

    ['“', 'Have', 'mercy', ',', 'gentlemen', '!', '”', 'odin', 'flung', 'up', 'his', 'hands', '.', '“', 'Don', '’', 't', 'write', 'that', ',', 'anyway', ';', 'have', 'some', 'shame', '.', 'Here', 'I', '’', 've', 'torn', 'my', 'heart', 'asunder', 'before', 'you', ',', 'and', 'you', 'seize', 'the', 'opportunity', 'and', 'are', 'fingering', 'the', 'wounds', 'in', 'both', 'halves', '....', 'Oh', ',', 'my', 'God', '!', '”']
    


```python
lemmatizer = WordNetLemmatizer() #단어 기본형으로 변환하기 위해서 wordnetlemmatizer실시 
[lemmatizer.lemmatize(t) for t in tokens]
```




    ['“',
     'Have',
     'mercy',
     ',',
     'gentleman',
     '!',
     '”',
     'odin',
     'flung',
     'up',
     'his',
     'hand',
     '.',
     '“',
     'Don',
     '’',
     't',
     'write',
     'that',
     ',',
     'anyway',
     ';',
     'have',
     'some',
     'shame',
     '.',
     'Here',
     'I',
     '’',
     've',
     'torn',
     'my',
     'heart',
     'asunder',
     'before',
     'you',
     ',',
     'and',
     'you',
     'seize',
     'the',
     'opportunity',
     'and',
     'are',
     'fingering',
     'the',
     'wound',
     'in',
     'both',
     'half',
     '....',
     'Oh',
     ',',
     'my',
     'God',
     '!',
     '”']




```python
stemmer = SnowballStemmer("english") #어근(단어의 실제 의미부분) 추출 
[stemmer.stem(t) for t in tokens]
```




    ['“',
     'have',
     'merci',
     ',',
     'gentlemen',
     '!',
     '”',
     'odin',
     'flung',
     'up',
     'his',
     'hand',
     '.',
     '“',
     'don',
     '’',
     't',
     'write',
     'that',
     ',',
     'anyway',
     ';',
     'have',
     'some',
     'shame',
     '.',
     'here',
     'i',
     '’',
     've',
     'torn',
     'my',
     'heart',
     'asund',
     'befor',
     'you',
     ',',
     'and',
     'you',
     'seiz',
     'the',
     'opportun',
     'and',
     'are',
     'finger',
     'the',
     'wound',
     'in',
     'both',
     'halv',
     '....',
     'oh',
     ',',
     'my',
     'god',
     '!',
     '”']



### Bag-of-Words 피처 생성
BOW :특성이 되는 단어에 대해 빈도를 값으로 사용 
  
  이를 실시하기 위해 아래와 같은 전처리 사용

 **CountVectorizer실시**
  * word_tokenize 토크나이저 사용 
  * stopwords 라이브러리 사용해서 불용어 제거 
  * ngram 범위 지정  
  * min_df사용하여 문서에 나타난 빈도가 100보다 작으면 제외 


```python
vec = CountVectorizer(tokenizer=word_tokenize, stop_words=stopwords.words('english'), ngram_range=(1, 2), min_df=100)
X_cnt = vec.fit_transform(trn_file['text'])
print(X_cnt.shape)
```

    (54879, 2685)
    


```python
X_cnt[0, :50].todense()
```




    matrix([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0]], dtype=int64)



**TfidfVectorizer**실시
* CountVectorizer와 나머지는 같고 ngram 범위는 1,3으로 min_df=50으로 설정


```python
vec = TfidfVectorizer(tokenizer=word_tokenize, stop_words=stopwords.words('english'), ngram_range=(1, 3), min_df=50)
X = vec.fit_transform(trn_file['text'])
X_tst = vec.transform(tst_file['text'])
print(X.shape, X_tst.shape)
```

    (54879, 5897) (19617, 5897)
    


```python
X[0, :50].todense()
```




    matrix([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0.]])



## 로지스틱회귀 모델 학습


```python
cv = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
```


```python
y = trn_file.author.values
y.shape
```




    (54879,)




```python
#값이 모두 0인 배열 생성 
p = np.zeros((X.shape[0], n_class))
p_tst = np.zeros((X_tst.shape[0], n_class)) 
for i_cv, (i_trn, i_val) in enumerate(cv.split(X, y), 1):
    clf = LogisticRegression()
    clf.fit(X[i_trn], y[i_trn])
    p[i_val, :] = clf.predict_proba(X[i_val])
    p_tst += clf.predict_proba(X_tst) / n_class
```

카운터벡터 정확도와 손실함수 확인하기 


```python
print(f'Accuracy (CV): {accuracy_score(y, np.argmax(p, axis=1)) * 100:8.4f}%')
print(f'Log Loss (CV): {log_loss(pd.get_dummies(y), p):8.4f}')
```

    Accuracy (CV):  76.6158%
    Log Loss (CV):   0.6800
    

## 앞의 결과를 토대로 새롭게 파일 생성


```python
np.savetxt(p_val_file, p, fmt='%.6f', delimiter=',')
np.savetxt(p_tst_file, p_tst, fmt='%.6f', delimiter=',')
```


```python
sample_file = './sample_submission.csv'
```


```python
sub = pd.read_csv(sample_file, index_col=0)
print(sub.shape)
sub.head()
```

    (19617, 5)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
sub[sub.columns] = p_tst
sub.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0631</td>
      <td>0.5302</td>
      <td>0.3155</td>
      <td>0.0659</td>
      <td>0.0253</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0815</td>
      <td>0.8202</td>
      <td>0.0032</td>
      <td>0.0269</td>
      <td>0.0682</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.7208</td>
      <td>0.0319</td>
      <td>0.1174</td>
      <td>0.0381</td>
      <td>0.0918</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0392</td>
      <td>0.0036</td>
      <td>0.8465</td>
      <td>0.0058</td>
      <td>0.1049</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.3044</td>
      <td>0.2440</td>
      <td>0.1450</td>
      <td>0.1905</td>
      <td>0.1161</td>
    </tr>
  </tbody>
</table>
</div>




```python
sub.to_csv(sub_file)
```
