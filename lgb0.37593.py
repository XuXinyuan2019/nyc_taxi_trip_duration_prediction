import pandas as pd
import numpy as np
import lightgbm as lgb

train = pd.read_csv('../input/train_add.csv')
test = pd.read_csv('../input/test_add.csv')

X = train.drop(['id','log_trip_duration'],axis=1)
y = train['log_trip_duration']

test_id = test['id']
test_X = test.drop(['id'],axis=1)

lgb_params = {
    'metric' : 'rmse',
    'learning_rate': 0.1,
    'max_depth': 25,
    'num_leaves': 1000, 
    'objective': 'regression',
    'feature_fraction': 0.9,
    'bagging_fraction': 0.5,
    'max_bin': 1000 }

lgb_df = lgb.Dataset(X, y, categorical_feature=['vendor_id','store_and_fwd_flag','passenger_count',
                                                               'Month','dayofweek',
                                                               'is_jfk','is_lg','is_holiday',
                                                               'pickup_district','dropoff_district',
                                                               
                                                               'number_of_steps'],free_raw_data=False)

lgb_model = lgb.train(lgb_params, lgb_df, num_boost_round=1500)

### feature selection
df = pd.DataFrame(X.columns.tolist(), columns=['feature'])
df['importance']=list(lgb_model.feature_importance())                           # 特征分数
df = df.sort_values(by='importance',ascending=False)                      # 特征排序

predictions = lgb_model.predict(test_X)

#Create a data frame designed a submission on Kaggle
submission = pd.DataFrame({'id': test_id, 'trip_duration': np.exp(predictions)-1})
submission.head()

#Create a csv out of the submission data frame
submission.to_csv("sub.csv", index=False)
