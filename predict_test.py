import luigi
import pandas as pd
from datetime import date
import pickle


class Predict_test(luigi.Task):

    def output(self):
        return luigi.LocalTarget('answers_test.csv')

    def run(self):
        data_test = pd.read_csv('data_test.csv').drop(['Unnamed: 0'], axis=1)
        data_test['buy_time'] = data_test['buy_time'].apply(lambda data: date.fromtimestamp(data))
        data_test['year'] = data_test['buy_time'].apply(lambda data: data.year)
        data_test['month'] = data_test['buy_time'].apply(lambda data: data.month)
        data_test['day'] = data_test['buy_time'].apply(lambda data: data.day)
                
        features = pd.read_csv('features.csv', sep='\t').drop(['Unnamed: 0'], axis=1)
        features['buy_time'] = features['buy_time'].apply(lambda data: date.fromtimestamp(data))
        features = features.drop(['85', '75', '139', '81', '203'],axis=1)
        
        X_test_nearest = pd.merge_asof(data_test.sort_values(by=['id']), 
                       features.sort_values(by=['id']), 
                       on='id', 
                       by='buy_time', 
                       direction='nearest')
        
        important_features_top = ['vas_id', 'id', 'day', '226', '52', '164', 'month', '128', '115', '247',
                                   '145', '53', '58', '144', '61', '1', '207', '188', '241', '169',
                                   '39', '5', '51', '143', '243', '168', '191', '171', '250', '63',
                                   '62', '3', '185', '60', '224', '246', '222', '127', '37', '244',
                                   '230', '40', '126', '0', '187', '146', '239', '148', '59', '50',
                                   '237', '2', '111', '4', '210', '49', '151', '234', '150', '110',
                                   '147', '223', '7', '133', '134', '25', '160', '227', '228', '9',
                                   '68', '112', '181', '156', '249', '6', '136', '18', '135', '8',
                                   '21', '102', '66', '159', '183', '34', '138', '113', '137', '158',
                                   '233', '67', '100', '10', '96', '41', '107', '152', '140', '101',
                                   '209', '215', '97', '42', '123', '214', '124', '174', '71', '141',
                                   '219', '142', '178', '86', '92', '91', '87', '173', '65', '94',
                                   '16', '83', '79', '118', '177', '220']
        
        with open('final_model.pkl', 'rb') as file:
            final_model = pickle.load(file)
        
        test_preds_proba = final_model.predict_proba(X_test_nearest[important_features_top])[:,1]
                
        result = pd.DataFrame({
                               "id" : X_test_nearest.id.tolist(),
                               "vas_id" : X_test_nearest.vas_id.tolist(),
                               "buy_time" : X_test_nearest.buy_time.tolist(),
                               "target" : test_preds_proba.tolist()
                              })
        
        with self.output().open('w') as f:
            result.to_csv(f, index=False)

if __name__ == '__main__':
    luigi.build([Predict_test()])