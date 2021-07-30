 
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')



train_df=pd.read_csv('train.csv')
test_df=pd.read_csv('test.csv')
age_gender_info=pd.read_csv('age_gender_info.csv')
sample_submission=pd.read_csv('sample_submission.csv')

train_df.loc[train_df.임대보증금=='-', '임대보증금'] = np.nan
test_df.loc[test_df.임대보증금=='-', '임대보증금'] = np.nan
train_df['임대보증금'] = train_df['임대보증금'].astype(float)
test_df['임대보증금'] = test_df['임대보증금'].astype(float)

train_df.loc[train_df.임대료=='-', '임대료'] = np.nan
test_df.loc[test_df.임대료=='-', '임대료'] = np.nan
train_df['임대료'] = train_df['임대료'].astype(float)
test_df['임대료'] = test_df['임대료'].astype(float)
 

# EDA - katie

train_df=train_df.rename({'도보 10분거리 내 지하철역 수(환승노선 수 반영)': '지하철', '도보 10분거리 내 버스정류장 수': '버스'},axis=1)
test_df=test_df.rename({'도보 10분거리 내 지하철역 수(환승노선 수 반영)': '지하철', '도보 10분거리 내 버스정류장 수': '버스'},axis=1)

age_gender_info=age_gender_info.set_index('지역')

code_total=pd.DataFrame(train_df.groupby(['단지코드','총세대수'])['전용면적별세대수'].sum()).reset_index()  
code_total

code_total[code_total['총세대수']!=code_total['전용면적별세대수']]

#임대료, 임대보증금 null 처리

total=pd.concat([train_df,test_df],axis=0) 
total.reset_index(drop=True, inplace=True) 
total   

total=total.drop_duplicates().reset_index(drop=True) 
total

total.isnull().sum()  

#1.자격 유형 
total[total['자격유형'].isnull()]
total[(total['단지코드']=='C2411') | (total['단지코드']=='C2253')]  
total.loc[(total['자격유형'].isnull()) & (total['단지코드']=='C2411'), '자격유형']='A'
total.loc[(total['자격유형'].isnull()) & (total['단지코드']=='C2253'), '자격유형']='C'
total['임대료'].replace('-',np.nan,inplace=True)  
total['임대보증금'].replace('-',np.nan,inplace=True)
total['임대료']=total['임대료'].astype(np.float64)   
total['임대보증금']=total['임대보증금'].astype(np.float64) 

# 2. 임대료, 임대보증금 
rent_null=total[total['임대료'].isnull() | total['임대보증금'].isnull()]    
rent_null
rent_null.임대보증금.value_counts(dropna=False)    
rent_null.loc[rent_null['임대보증금'].notnull(), '임대료']=0  
rent_null.임대료.value_counts(dropna=False) 
total.loc[total['공급유형']=='장기전세', '임대료']=0
total.loc[total['공급유형']=='장기전세', '임대료']
#자격유형 C, D인 주민들 사이에 임대료, 임대보증금의 차이가 크게 나지 않을 것이라는 가정하에 전용면적에 따라 대체 
total.loc[2707,'임대보증금']=5787000.0 
total.loc[2707,'임대료']=79980.0 
total.loc[2709,'임대보증금']=11574000.0 
total.loc[2709,'임대료']=159960.0
#공공분양의 경우 임대료 X  
total.loc[(total['지역']=='대전광역시') & (total['공급유형']=='공공분양'), ['임대료','임대보증금']]=0    
total[(total['지역']=='대전광역시') & (total['공급유형']=='공공분양')]  
total['총분양가']=0.0   
total.loc[(total['지역']=='대전광역시') & (total['공급유형']=='공공분양'), '총분양가']=[221640000, 217200000, 221640000, 251220000, 251220000, 251260000, 251320000]
total.loc[(total['임대건물구분']=='아파트') & (total['임대보증금'].isnull()) & (total['지역'] == '부산광역시'), '임대보증금'] = [7000000,7000000,7600000,14800000,23100000]
total.loc[(total['임대건물구분']=='아파트') & (total['임대료'].isnull()) & (total['지역'] == '부산광역시'), '임대료']= [135000, 135000, 142000, 198000, 259000]    
total.loc[(total['임대건물구분']=='아파트') & (total['임대보증금'].isnull()) & (total['지역'] == '대구광역시'), '임대보증금'] = [10847000, 10847000, 17338000] 
total.loc[(total['임대건물구분']=='아파트') & (total['임대료'].isnull()) & (total['지역'] == '대구광역시'), '임대료'] = [138600, 138600, 197500]  

pd.options.display.float_format='{:.3f}'.format  
total[(total['지역']=='경상남도') & (total['임대건물구분']=='아파트') & (total['공급유형']=='행복주택')].groupby(['자격유형','전용면적'])[['임대보증금','임대료']].mean()

total.loc[(total['지역']=='경상남도') & (total['공급유형']=='행복주택') & (total['임대보증금'].isnull()) & (total['전용면적']==16.94) , '임대보증금']=14965000.0 
total.loc[(total['지역']=='경상남도') & (total['공급유형']=='행복주택') & (total['임대료'].isnull()) & (total['전용면적']==16.94) , '임대료']=70915.0

total.loc[(total['지역']=='경상남도') & (total['공급유형']=='행복주택') & (total['임대보증금'].isnull()) & (total['전용면적']==36.770) , '임대보증금']=34140000.000
total.loc[(total['지역']=='경상남도') & (total['공급유형']=='행복주택') & (total['임대료'].isnull()) & (total['전용면적']==36.770) , '임대료']=167000.000  


total.loc[(total['지역']=='경상남도') & (total['공급유형']=='행복주택') & (total['임대보증금'].isnull()) & (total['전용면적']==26.850) , '임대보증금'] = 25000000.000 
total.loc[(total['지역']=='경상남도') & (total['공급유형']=='행복주택') & (total['임대료'].isnull()) & (total['전용면적']==26.850) , '임대료'] = 130000.000 

total.loc[(total['지역']=='강원도') & (total['공급유형']=='행복주택') & total['임대보증금'].isnull() , '임대보증금']= [13181000, 19700000, 19150000, 21679000]  
total.loc[(total['지역']=='강원도') & (total['공급유형']=='행복주택') & total['임대료'].isnull() , '임대료'] = [65500, 96000, 94000, 105000]  
total.loc[(total['지역']=='강원도') & (total['공급유형']=='영구임대') & total['임대보증금'].isnull(), '임대보증금']=[2129000, 2902000]
total.loc[(total['지역']=='강원도') & (total['공급유형']=='영구임대') & total['임대료'].isnull(), '임대료']=[42350, 57730]

#상가null
total[(total['임대보증금'].isnull()) & (total['지역']=='부산광역시')].groupby('단지코드')['총세대수'].value_counts()   

#버스null
total.loc[(total['지역']=='경상남도') & (total['공급유형']=='공공임대(10년)') & (total['지하철'].isnull()) , '지하철']=0
total.loc[(total['지역']=='경상남도') & (total['공급유형']=='공공임대(10년)') &(total['버스'].isnull()), '버스'] = 3
total['지하철']=total['지하철'].fillna(0)  
total['지하철'].value_counts(dropna=False)

age_gender_info.reset_index()
total_age=total.merge(age_gender_info.reset_index(), on='지역', how='inner') 
total_age




 
test_df.loc[test_df.단지코드.isin(['C2411']) & test_df.자격유형.isnull(), '자격유형'] = 'A'
test_df.loc[test_df.단지코드.isin(['C2253']) & test_df.자격유형.isnull(), '자격유형'] = 'C'

 
unique_cols = ['총세대수', '지역', '공가수', 
               '지하철',
               '버스',
               '단지내주차면수', '등록차량수'
               ]
train_agg = train_df.set_index('단지코드')[unique_cols].drop_duplicates()
test_agg = test_df.set_index('단지코드')[[col for col in unique_cols if col!='등록차량수']].drop_duplicates()

 

# 이걸 보고 순서대로 0,1,2,3,4, 입력해보기
df=pd.DataFrame(train_agg.groupby('지역')['등록차량수'].mean().sort_values())
for i,v in enumerate(df.index):
    train_agg.loc[train_agg['지역']==v,'지역']=i
    test_agg.loc[test_agg['지역']==v,'지역']=i

 
for i in train_df['자격유형'].unique():
    train_df['자격유형_{}'.format(i)]=0

for i in train_df['공급유형'].unique():
    train_df['공급유형_{}'.format(i)]=0

for i in train_df['단지코드'].unique():
    df=train_df[train_df['단지코드']==i]
    qual_columns=df['자격유형'].unique()
    sup_columns=df['공급유형'].unique()
    
    for z in qual_columns:
        train_df.loc[train_df['단지코드']==i,'자격유형_{}'.format(z)]=df[df['자격유형']==z]['전용면적별세대수'].sum()/df['전용면적별세대수'].sum()
    for z in sup_columns:
        train_df.loc[train_df['단지코드']==i,'공급유형_{}'.format(z)]=df[df['공급유형']==z]['전용면적별세대수'].sum()/df['전용면적별세대수'].sum()

train_df_2=train_df[['단지코드','자격유형_A', '자격유형_B', '자격유형_C', '자격유형_D',
       '자격유형_E', '자격유형_F', '자격유형_G', '자격유형_H', '자격유형_I', '자격유형_J', '자격유형_K',
       '자격유형_L', '자격유형_M', '자격유형_N', '자격유형_O', '공급유형_국민임대', '공급유형_공공임대(50년)',
       '공급유형_영구임대', '공급유형_임대상가', '공급유형_공공임대(10년)', '공급유형_공공임대(분납)',
       '공급유형_장기전세', '공급유형_공공분양', '공급유형_행복주택', '공급유형_공공임대(5년)']].drop_duplicates()
 

for i in test_df['자격유형'].unique():
    test_df['자격유형_{}'.format(i)]=0

for i in test_df['공급유형'].unique():
    test_df['공급유형_{}'.format(i)]=0

for i in test_df['단지코드'].unique():
    df=test_df[test_df['단지코드']==i]
    qual_columns=df['자격유형'].unique()
    sup_columns=df['공급유형'].unique()
    
    for z in qual_columns:
        test_df.loc[test_df['단지코드']==i,'자격유형_{}'.format(z)]=df[df['자격유형']==z]['전용면적별세대수'].sum()/df['전용면적별세대수'].sum()
    for z in sup_columns:
        test_df.loc[test_df['단지코드']==i,'공급유형_{}'.format(z)]=df[df['공급유형']==z]['전용면적별세대수'].sum()/df['전용면적별세대수'].sum()

 
test_df_2=test_df[['단지코드','자격유형_H', '자격유형_A', '자격유형_E', '자격유형_C', '자격유형_D',
       '자격유형_G', '자격유형_I', '자격유형_J', '자격유형_K', '자격유형_L', '자격유형_M', '자격유형_N',
       '공급유형_국민임대', '공급유형_영구임대', '공급유형_임대상가', '공급유형_공공임대(50년)',
       '공급유형_공공임대(10년)', '공급유형_공공임대(분납)', '공급유형_행복주택']].drop_duplicates()

 
train_agg=pd.merge(train_agg,train_df_2,on='단지코드')
test_agg=pd.merge(test_agg,test_df_2,on='단지코드')

train_agg=train_agg.fillna(0)
test_agg=test_agg.fillna(0)
 
train=train_agg.drop(['단지코드'],axis=1)
train=train_agg[['총세대수','지역', '공가수', '지하철', '버스',
       '단지내주차면수', '자격유형_H', '자격유형_A', '자격유형_E',
       '자격유형_C', '자격유형_D', '자격유형_G', '자격유형_I', '자격유형_J', '자격유형_K', '자격유형_L',
       '자격유형_M', '자격유형_N', '공급유형_국민임대', '공급유형_영구임대', '공급유형_임대상가',
       '공급유형_공공임대(50년)', '공급유형_공공임대(10년)', '공급유형_공공임대(분납)', '공급유형_행복주택', '등록차량수']]

test=test_agg.drop(['단지코드'],axis=1)


 
import xgboost as xgb
from sklearn.metrics import mean_absolute_error


 



X, y = np.array(train.iloc[:,:-1]),np.array(train.iloc[:,-1])
data_dmatrix = xgb.DMatrix(data=X,label=y)



 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10)

xg_reg = xgb.XGBRegressor(objective ='reg:linear',eval_metric='mae', colsample_bytree = 1, learning_rate = 0.5, max_depth = 4, alpha = 5, n_estimators = 9)
#
xg_reg.fit(X_train,y_train)

#preds = xg_reg.predict(np.array(test))
preds = xg_reg.predict(X_test)


mae = np.sqrt(mean_absolute_error(y_test, preds))
print("MAE: %f" % (mae))


 


xg_reg.fit(X,y)

preds = xg_reg.predict(np.array(test))

sub_df=test_agg[['단지코드']]
sub_df['Y']=preds
sub_df.columns=['code','num']
sub_df.to_csv('submission.csv',index=False)






 
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

xg_reg = xgb.XGBRegressor(objective ='reg:linear',eval_metric='mae', colsample_bytree = 1, learning_rate = 0.4, max_depth = 3,  n_estimators = 10)

avg=0
for i in range(20):
    scores = cross_val_score(xg_reg, X, y, cv=KFold(n_splits=5, shuffle=True), scoring='r2')

    avg+=scores.mean()
    

print(avg/20)

 
