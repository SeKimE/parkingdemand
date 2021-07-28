#!/usr/bin/env python
# coding: utf-8

# ## 주차수요 예측 AI 경진대회

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

#폰트 설정          
from matplotlib import rc    
get_ipython().run_line_magic('matplotlib', 'inline')
rc('font', family='AppleGothic' )
plt.rcParams['axes.unicode_minus'] = False  


# In[2]:


#데이터 불러오기
dir='/Users/yuchanghee/Desktop/DACON/Parking_Demand/data/'
train= pd.read_csv(dir+'train.csv') 
test= pd.read_csv(dir+'test.csv') 
age_gender_info=pd.read_csv(dir+'age_gender_info.csv')
sample_submission=pd.read_csv(dir+'sample_submission.csv')


# In[3]:


print('train.shape: ', train.shape) 
print('test.shape: ', test.shape)     
print('age_gender_info.shape: ', age_gender_info.shape) 


# In[5]:


train.head() 


# In[6]:


test.head()


# In[10]:


age_gender_info.head()


# In[11]:


sample_submission.head()


# #### 변수 설명
# - 자격 유형: 입주민의 입주자격의 유형(비식별화)
# - 공가수: 비어있는 집
# - 전용면적별 세대수: 임대 아파트 대상만 집계
# - 전체 세대수 : '분양'아파트가 포함된 수치
# - 전용 면적 : 단순 면적이 아닌 별도의 비공개 기준

# In[7]:


#결측치 확인  
train.isnull().sum()


# In[11]:


test.isnull().sum()


# #### 추측 1: 임대보증금 - 임대료 상관관계 높아보임

# ### EDA 
# - 단변수 분석
# - 다변수 분석 (등록차량수)
# - 상관분석 

# In[4]:


train=train.rename({'도보 10분거리 내 지하철역 수(환승노선 수 반영)': '지하철', '도보 10분거리 내 버스정류장 수': '버스'},axis=1)
test=test.rename({'도보 10분거리 내 지하철역 수(환승노선 수 반영)': '지하철', '도보 10분거리 내 버스정류장 수': '버스'},axis=1)


# In[76]:


train.info()   


# In[77]:


train.describe().T  


# In[78]:


test.describe().T  


# In[5]:


age_gender_info=age_gender_info.set_index('지역')


# In[6]:


#인구 분포 파악 (평균)     
age_gender_info.loc['전체 분포 평균']=age_gender_info.mean()        
age_gender_info.loc['광역시 평균']=age_gender_info[age_gender_info.index.str.contains('시')].mean()
age_gender_info.loc['도 평균']=age_gender_info[age_gender_info.index.str.contains('도')].mean()


# In[13]:


age_gender_info 


# In[7]:


#세대 파악  
fig = plt.figure(figsize = (18, 8))  
sns.lineplot(data=age_gender_info.T) 
plt.title('지역별 세대(양성) 라인 차트')
plt.xticks(rotation=45)  
plt.ylim(top=0.13) 
plt.show() 


# 1. 대체로 남성보다 여성의 비율이 더 높다. 
# 2. 세종특별시 - 30대 비율 높다.
# 3. 서울특별시, 부산 광역시 - 60, 70대 여성 비율 높다.
# 4. 정규 분포의 형태
# 

# In[154]:


age_gender_info[age_gender_info.columns[age_gender_info.columns.str.contains('여자')]]  


# In[147]:


#성별 파악  
fig = plt.figure(figsize = (18, 8))  
sns.lineplot(data=age_gender_info[age_gender_info.columns[age_gender_info.columns.str.contains('여자')]].T)  
plt.title('지역별 성별(여성) 라인 차트')
plt.xticks(rotation=45)  
plt.show() 


# 1. 60대 여성까지 전체적으로 나이가 많아질수록 비율이 높아진다. 
# 2. 정규 분포의 형태

# In[150]:


#성별 파악  
fig = plt.figure(figsize = (18, 8))  
sns.lineplot(data=age_gender_info[age_gender_info.columns[age_gender_info.columns.str.contains('남자')]].T)  
plt.title('지역별 성별(남성) 라인 차트')  
plt.xticks(rotation=45)  
plt.show() 


# 1. 60대부터 비율이 낮아진다. 
# 2. 20대, 40대보다 30대 남자의 비율이 낮다. 
# 3. 정규 분포의 형태

# In[153]:


#히트맵
plt.figure(figsize=(14, 10))
sns.heatmap((age_gender_info*100).round(3),annot=True, linewidths=0.01)
plt.show() 


# 1. 세종특별시 - 30대 비율 높다. 
# 2. 부산광역시 - 60대 비율 높다. 
# 3. 서울특별시 - 60대 비율 높다. 

# In[165]:


len(train.dtypes[train.dtypes!='object'].index.tolist())     


# In[7]:


def float_plot(columns, df): #연속형 단변수 
    fig =plt.figure(figsize=(20,30))
    for i, col in enumerate(columns): 
        plt.subplot(5,2,i+1)            
        sns.distplot(df[col])   

def cate_plot(columns,df): #범주형 단변수
    fig =plt.figure(figsize=(20,35))      
    for i, col in enumerate(columns): 
        plt.subplot(5, 2, i+1) 
        chart=sns.countplot(x=col,data=df) 
        chart.set_xticklabels(chart.get_xticklabels(), rotation=65)


# In[15]:


float_plot(train.dtypes[train.dtypes!='object'].index.tolist(), train)  


# - 세대수가 많은 대단지 존재
# - 로그화 필요

# In[8]:


#총세대수가 가장 많은,적은 단지코드 
print(train.loc[train['단지코드'].drop_duplicates().index][['단지코드','총세대수','지역','공급유형','전용면적']].nlargest(3,'총세대수'))
print(train.loc[train['단지코드'].drop_duplicates().index][['단지코드','총세대수','지역','공급유형','전용면적']].nsmallest(3,'총세대수')) 


# In[13]:


#단지코드별 총세대수 = 전용면적별세대수
code_total=pd.DataFrame(train.groupby(['단지코드','총세대수'])['전용면적별세대수'].sum()).reset_index()  
code_total


# In[18]:


code_total[code_total['총세대수']!=code_total['전용면적별세대수']]


# - 분양 아파트 세대수 컬럼 필요 (if 총세대수 != 전용면적별세대수)
# - 실 세대수 칼럼 필요 (전용면적별세대수 sum - 공가수)

# In[96]:


train.columns


# In[21]:


cate_plot(['임대건물구분','지역','공급유형','자격유형'], train)         


# In[59]:


train[['임대료','임대보증금']].value_counts()


# In[9]:


#임대료, 임대보증금 null 처리
train.loc[train['임대료']=='-', '임대료'] = np.nan  
train.loc[train['임대보증금']=='-', '임대보증금'] = np.nan  


# In[10]:


train[['임대보증금','임대료']]=train[['임대보증금','임대료']].astype(np.float64)  


# In[23]:


fig,ax=plt.subplots(1,2)    
plt.figure(figsize=(15,10))      
sns.distplot(train['임대료'], ax=ax[0]) 
sns.distplot(train['임대보증금'] ,ax=ax[1]) 
print('평균 임대료',np.round(train['임대료'].mean(),2), '원')          
print('평균 임대보증금',np.round(train['임대보증금'].mean(),2), '원')   


# - 임대료 대부분이 40만원 이하, 평균 대략 20만원
# - 임대보증금 대부분 5천만원 이하, 평균 2천 6백만원

# 

# ## 2. 다변수 분석 - 종속변수와의 관계

# In[106]:


train.dtypes[train.dtypes=='object']


# In[48]:


train[train['임대건물구분']=='아파트']['등록차량수'].describe()


# In[49]:


train  


# In[11]:


def boxplot(df, col, y_col):
    fig = plt.figure(figsize=(20,50))
    for i, column in enumerate(df.dtypes[df.dtypes=='object'].index.tolist()):
        plt.subplot(7,2,i+1)  
        chart=sns.boxplot(x=df[column],y=y_col, data=df) 
        chart.set_xticklabels(chart.get_xticklabels(), rotation=65)


# In[101]:


boxplot(train, train.columns, '등록차량수')


# 1. 아파트가 상가보다 더 등록 차량수가 많다. 
# 2. 세종특별자치시가 다른 지역보다 월등하게 등록 차량수가 많다. 그 다음으로는 수도권  
# 3. 공공임대(분납)의 경우 등록 차량수가 많다.  

# ## 3. 상관분석

# In[50]:


train.corr()   


# In[51]:


mask = np.zeros_like(train.corr() ,dtype=bool)
mask[np.triu_indices_from(mask)] = True   
plt.figure(figsize=(20,10))
sns.heatmap(train.corr(),annot=True,fmt=".3f",annot_kws = {"size":20},cmap=sns.cubehelix_palette(),mask=mask) 


# ## 4. Preprocessing

# In[12]:


#데이터 중복 제거 
total=pd.concat([train,test],axis=0) 
total.reset_index(drop=True, inplace=True) 
total   


# In[13]:


total=total.drop_duplicates().reset_index(drop=True) 
total


# In[222]:


#null처리
total.isnull().sum()   


# In[241]:


#1.자격 유형 
total[total['자격유형'].isnull()]


# - 자격 유형은 단지코드, 지역, 공급유형과 연관이 있지 않을까

# In[242]:


total[(total['단지코드']=='C2411') | (total['단지코드']=='C2253')]  


# In[14]:


total.loc[(total['자격유형'].isnull()) & (total['단지코드']=='C2411'), '자격유형']='A'
total.loc[(total['자격유형'].isnull()) & (total['단지코드']=='C2253'), '자격유형']='C'


# In[15]:


total['임대료'].replace('-',np.nan,inplace=True)  
total['임대보증금'].replace('-',np.nan,inplace=True)
total['임대료']=total['임대료'].astype(np.float64)   
total['임대보증금']=total['임대보증금'].astype(np.float64) 


# In[245]:


total.isnull().sum() 


# In[16]:


# 2. 임대료, 임대보증금 
rent_null=total[total['임대료'].isnull() | total['임대보증금'].isnull()]    
rent_null


# In[17]:


rent_null.임대보증금.value_counts(dropna=False)    


# In[18]:


rent_null[rent_null['임대보증금'].notnull()]


# - 장기전세는 임대료 없으므로 0으로 대체

# In[19]:


rent_null.loc[rent_null['임대보증금'].notnull(), '임대료']=0  


# In[20]:


rent_null.임대료.value_counts(dropna=False) 


# In[21]:


total.loc[total['공급유형']=='장기전세', '임대료']=0
total.loc[total['공급유형']=='장기전세', '임대료']


# In[252]:


boxplot(total, total.columns, '임대보증금')


# In[109]:


boxplot(total, total.columns, '임대료') 


# - 공공분양, 상가의 경우 임대료와 임대 정보 X 
# - 세종특별자치시, 서울특별시의 경우 임대료, 임대보증금 높다. 
# - 장기 전세, 공공임대(분납)의 경우 임대보증금이 높다.  
# - 공공임대(10년)의 경우 임대료가 높다. 
# - 자격유형 D의 경우 임대보증금과 임대료 정보 X
# - 자격유형 E의 주민들은 임대보증금이 높게, 임대료가 낮게 측정

# In[253]:


total[total['자격유형']=='E'] 


# In[130]:


total['전용면적'].describe()   


# In[118]:


total[total['자격유형']=='D']['공급유형'].value_counts()  


# In[124]:


total[(total['자격유형']=='D')]['임대건물구분'].value_counts()  


# In[125]:


total[(total['자격유형']=='D') & (total['임대건물구분']=='아파트')] 


# 1. 자격 유형 D인 사람들 대부분 상가 건물, 아파트여도 2개를 제외하고 모두 공공분양

# In[128]:


total[(total['지역']=='대전광역시') & (total['공급유형']=='영구임대')]  


# In[22]:


total[(total['지역']=='대전광역시') & (total['공급유형']=='영구임대') & (total['전용면적']==52.74)] 


# In[23]:


#자격유형 C, D인 주민들 사이에 임대료, 임대보증금의 차이가 크게 나지 않을 것이라는 가정하에 전용면적에 따라 대체 
total.loc[2707,'임대보증금']=5787000.0 
total.loc[2707,'임대료']=79980.0 
total.loc[2709,'임대보증금']=11574000.0 
total.loc[2709,'임대료']=159960.0


# - LH 청약센터 자료 활용

# In[24]:


#공공분양의 경우 임대료 X  
total[(total['지역']=='대전광역시') & (total['공급유형']=='공공분양')] 


# In[25]:


total.loc[(total['지역']=='대전광역시') & (total['공급유형']=='공공분양'), ['임대료','임대보증금']]=0    
total[(total['지역']=='대전광역시') & (total['공급유형']=='공공분양')]  


# In[26]:


total['총분양가']=0.0   
total[(total['지역']=='대전광역시') & (total['공급유형']=='공공분양')]           


# 1. 대전광역시, 공공분양 : 대전광역시 서구 관저동 공공주택지구내 블록

# In[27]:


total.loc[(total['지역']=='대전광역시') & (total['공급유형']=='공공분양'), '총분양가']=[221640000, 217200000, 221640000, 251220000, 251220000, 251260000, 251320000]


# In[28]:


total[(total['지역']=='대전광역시') & (total['공급유형']=='공공분양')]     


# In[29]:


total[(total['임대건물구분']=='아파트') & (total['임대보증금'].isnull())].sort_values(by='지역', ascending=False)   


# 2. 부산광역시 - 국민임대 : 부산광역시 기장군 정관읍 모전로 41 (모전리 681번지) 부산정관 7단지 국민임대주택 

# In[30]:


total.loc[(total['임대건물구분']=='아파트') & (total['임대보증금'].isnull()) & (total['지역'] == '부산광역시'), '임대보증금'] = [7000000,7000000,7600000,14800000,23100000]  


# In[31]:


total.loc[(total['임대건물구분']=='아파트') & (total['임대료'].isnull()) & (total['지역'] == '부산광역시'), '임대료']= [135000, 135000, 142000, 198000, 259000]    


# 3. 대구광역시 - 국민임대 : 대구연경A2블록 국민임대주택

# In[32]:


total.loc[(total['임대건물구분']=='아파트') & (total['임대보증금'].isnull()) & (total['지역'] == '대구광역시'), '임대보증금'] = [10847000, 10847000, 17338000] 
total.loc[(total['임대건물구분']=='아파트') & (total['임대료'].isnull()) & (total['지역'] == '대구광역시'), '임대료'] = [138600, 138600, 197500]  


# In[33]:


total[(total['임대건물구분']=='아파트') & (total['임대보증금'].isnull())].sort_values(by='지역', ascending=False)


# 4. 경상남도 - 행복주택 : 정보 명확하지 않아서 경상남도, 행복주택인 경우에 따라 대치 
#     - 자격 유형이 L인 경우와 J인 경우 비슷
#     - 자격 유형과 전용 면적에 따라 처리

# In[34]:


total[(total['지역']=='경상남도') & (total['임대건물구분']=='아파트') & (total['공급유형']=='행복주택')].sort_values('전용면적')              


# In[35]:


pd.options.display.float_format='{:.3f}'.format  
total[(total['지역']=='경상남도') & (total['임대건물구분']=='아파트') & (total['공급유형']=='행복주택')].groupby(['자격유형','전용면적'])[['임대보증금','임대료']].mean()


# In[36]:


total.loc[(total['지역']=='경상남도') & (total['공급유형']=='행복주택') & (total['임대보증금'].isnull()) & (total['전용면적']==16.94) , '임대보증금']=14965000.0 
total.loc[(total['지역']=='경상남도') & (total['공급유형']=='행복주택') & (total['임대료'].isnull()) & (total['전용면적']==16.94) , '임대료']=70915.0 


# In[37]:


total.loc[(total['지역']=='경상남도') & (total['공급유형']=='행복주택') & (total['임대보증금'].isnull()) & (total['전용면적']==36.770) , '임대보증금']=34140000.000
total.loc[(total['지역']=='경상남도') & (total['공급유형']=='행복주택') & (total['임대료'].isnull()) & (total['전용면적']==36.770) , '임대료']=167000.000  


# In[38]:


total.loc[(total['지역']=='경상남도') & (total['공급유형']=='행복주택') & (total['임대보증금'].isnull()) & (total['전용면적']==26.850) , '임대보증금'] = 25000000.000 
total.loc[(total['지역']=='경상남도') & (total['공급유형']=='행복주택') & (total['임대료'].isnull()) & (total['전용면적']==26.850) , '임대료'] = 130000.000 


# In[39]:


#여기서부터 다시 
total[(total['지역']=='강원도') & (total['공급유형']=='행복주택')] 


# 5. 강원도 - 행복주택 : 춘천거두2 행복주택
# 6. 강원도 - 영구임대 : 화천신읍 공공실버주택 

# In[59]:


total.loc[(total['지역']=='강원도') & (total['공급유형']=='행복주택') & total['임대보증금'].isnull() , '임대보증금']= [13181000, 19700000, 19150000, 21679000]  


# In[55]:


total.loc[(total['지역']=='강원도') & (total['공급유형']=='행복주택') & total['임대료'].isnull() , '임대료'] = [65500, 96000, 94000, 105000]  


# In[71]:


total.loc[(total['지역']=='강원도') & (total['공급유형']=='영구임대') & total['임대보증금'].isnull(), '임대보증금']=[2129000, 2902000]
total.loc[(total['지역']=='강원도') & (total['공급유형']=='영구임대') & total['임대료'].isnull(), '임대료']=[42350, 57730]


# #### 2. 상가 null 채우기 

# In[79]:


['지역'].value_counts()total[total['임대보증금'].isnull()]


# In[85]:


#1. 부산광역시 상가 
total[(total['임대보증금'].isnull()) & (total['지역']=='부산광역시')].sort_values(by='전용면적') 


# In[96]:


total[(total['임대보증금'].isnull()) & (total['지역']=='부산광역시')].groupby('단지코드')['총세대수'].value_counts()   


# In[140]:


total[total['단지코드']=='C1109']


# - 부산금곡4
# 1. 21.98 -> 14000000, 189900
# 2. 28.38 -> 4992000 208000 
# 3. 21.85 -> 13350000, 181100
#              

# In[ ]:





# In[ ]:





# In[ ]:





# #### 3. 버스 null 채우기

# In[103]:


total.isnull().sum()


# In[104]:


total[total['버스'].isnull()] 


# In[105]:


boxplot(total, total.columns, '버스') #버스 이상치 존재 50


# In[107]:


total[total['버스']==50]  


# In[113]:


boxplot(total, total.columns, total[total['버스']<50]['버스']) #버스 이상치 삭제 


# In[124]:


boxplot(total, total.columns, '지하철') 


# In[133]:


total[(total['지역']=='경상남도') & (total['공급유형']=='공공임대(10년)')].sort_values(by=['임대보증금','등록차량수']) 


# 1. 버스는 임대보증금, 등록차량수와 상관성이 있으므로 그에 따라 대체 -> 3
# 2. 해당 row 지하철: 0으로 대체

# In[130]:


total.loc[(total['지역']=='경상남도') & (total['공급유형']=='공공임대(10년)') & (total['지하철'].isnull()) , '지하철']=0
total.loc[(total['지역']=='경상남도') & (total['공급유형']=='공공임대(10년)') &(total['버스'].isnull()), '버스'] = 3


# In[141]:


total.isnull().sum()


# 4. 지하철 null 채우기

# In[148]:


total[total['지하철'].isnull()]['지역'].value_counts() 


# - 충남, 대전 지하철 null

# In[149]:


total[total['지하철'].isnull()]


# In[221]:


total['지하철'].value_counts(dropna=False)


# In[162]:


for i in [0,1,2,3]:  
    print('지하철 개수: ',i, total[total['지하철']==i]['지역'].unique())


# In[168]:


total[total['지역']=='대전광역시']['지하철'].value_counts(dropna=False)


# In[216]:


total[(total['지역']=='대전광역시') & (total['지하철'].isnull())]['단지코드'].value_counts()


# In[217]:


total[(total['지역']=='대전광역시') & (total['지하철'].notnull())]['단지코드'].value_counts()    


# - 지하철은 그나마 총세대수와 상관성이 있다. 

# In[234]:


#모르겠다.. 그냥 최빈값 0으로 대치
total['지하철'].value_counts(dropna=False)


# In[235]:


total['지하철']=total['지하철'].fillna(0)  
total['지하철'].value_counts(dropna=False)


# In[236]:


total.isnull().sum()  


# In[239]:


age_gender_info.reset_index()


# In[267]:


set(age_gender_info.reset_index()['지역'].unique()) - set(total['지역'].unique()) 


# In[241]:


total  


# In[242]:


total_age=total.merge(age_gender_info.reset_index(), on='지역', how='inner') 
total_age


# In[272]:


total_age[total_age.columns[total_age.columns.str.contains('남자|여자')].tolist()].mean() 


# In[280]:


total_age.columns


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[92]:


for column in train.dtypes[train.dtypes!='object'].index.tolist():
    chart=sns.scatterplot(x=train[column],y='등록차량수', data=train)
    plt.title(column) 
    chart.set_xticklabels(chart.get_xticklabels(), rotation=65)
    plt.show()  


# 1. 총 세대수가 많을수록 당연히 등록차량수도 많다. 
# 2. 전용면적과 등록차량수는 상관관계가 적다. 
# 3. 단지내 주차 면수가 클수록 등록차량수가 많다. 
# 

# In[ ]:




