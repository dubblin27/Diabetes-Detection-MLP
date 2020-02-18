
import numpy as np 
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split,KFold
from subprocess import check_output
from sklearn.metrics import confusion_matrix,classification_report,r2_score

np.random.seed(16)
class Pima:
    def load_data(self):
        self.df=pd.read_csv('diabetes.csv')
        self.scores={}
        self.skin_scores={}
        self.missing_values={}
        self.all_skin=(self.df.SkinThickness!=0)&(self.df.Insulin!=0)&(self.df.BMI!=0)
    def std_scale(self):
        scale=StandardScaler()
        scale_skin=StandardScaler()
        scale.fit(self.X)
        scale_skin.fit(self.X_skin)
        self.X=scale.transform(self.X)
        self.X_skin=scale_skin.transform(self.X_skin)
        pass
    def mm_scale(self):
        scale=MinMaxScaler()
        scale_skin=MinMaxScaler()
        scale.fit(self.X)
        scale_skin.fit(self.X_skin)
        self.X=scale.transform(self.X)
        self.X_skin=scale_skin.transform(self.X_skin)
        pass
    def reload_data(self):
        self.X=self.df.iloc[:,:8]
        self.Y=self.df.iloc[:,8]
        self.X_skin=self.df.loc[self.all_skin,['Insulin','BMI','DiabetesPedigreeFunction','Age']]
        self.Y_skin=np.array(self.df.loc[self.all_skin,'SkinThickness'])
        self.std_scale()
        #self.train,self.test,self.train_y,self.test_y=train_test_split(self.X,self.Y,random_state=7)
        self.describe=self.df.describe()
        self.corr_mat=self.df.corr()
        for col in self.df:
            self.missing_values[col]=len(self.df.loc[self.df[col]==0,:])
        pass
    def more_vars(self):
        self.df['Insulin_Log']=np.log(self.df['Insulin']+1)
        self.df.loc[self.df.Age<=60,'temp_age']=self.df.loc[self.df.Age<=60,'Age']-13
        self.df.loc[self.df.Age>60,'temp_age']=60
        self.df.loc[:,'Preg/Age']=self.df.loc[:,'Pregnancies']/self.df.loc[:,'temp_age']
        self.reload_data()
        pass
    def variable_distributions(self):
        for col_ind in range(len(self.df.columns)):
            plt.figure(col_ind)
            sns.boxplot(self.df.iloc[:,col_ind])
        pass
    def adjust_data(self):
        #Glucose
        all_g=self.df.Glucose!=0
        self.df.loc[self.df.Glucose==0,'Glucose']=np.mean(self.df.Glucose[all_g])
        #BloodPressure
        all_bp=self.df.BloodPressure>=40
        self.df.loc[(self.df.BloodPressure<40)&(self.df.BloodPressure!=0),'BloodPressure']=40
        self.df.loc[self.df.BloodPressure==0,'BloodPressure']=np.mean(self.df.BloodPressure[all_bp])
        
        
    def evaluate(self):
        self.class_report=classification_report(self.test_y,self.predictions)
        self.conf=confusion_matrix(self.test_y,self.predictions)
        #print('Confusion mat:')
        #print(conf)
        #print('Classification report:')
        #print(class_report)
        pass
    def MLP(self,layer_size):
        cv_scores=[]
        kfold=KFold(n_splits=3)
        self.kfolds=kfold.split(self.X)
        for train_index,test_index in self.kfolds:
            #print('Run with layers:',str(layer_size))
            cf=MLPClassifier(hidden_layer_sizes=layer_size,random_state=16,max_iter=1000)
            cf.fit(self.X[train_index],self.Y[train_index])
            #print(cf.score(self.test,self.test_y))
            score=100*cf.score(self.X[test_index,:],self.Y[test_index])
            cv_scores.append(score)
            #self.predictions=cf.predict(self.test)
            #self.evaluate()
        self.scores[str(layer_size)]=np.mean(cv_scores)
        pass
    def SkinMLP(self,layer_size):
        cv_scores=[]
        kfold=KFold(n_splits=10)
        self.skfolds=kfold.split(self.X_skin)
        for train_index,test_index in self.skfolds:
            print('Run with layers:',str(layer_size))
            cf=MLPRegressor(hidden_layer_sizes=layer_size,random_state=3,max_iter=1000)
            cf.fit(self.X_skin[train_index],self.Y_skin[train_index])
            prediction=cf.predict(self.X_skin[test_index])
            #print(cf.score(self.X_skin,self.Y_skin))
            score=100*r2_score(self.Y_skin[test_index],prediction)
            cv_scores.append(score)
            self.predictions=cf.predict(self.test)
            self.evaluate()
            print(self.predictions)
        self.skin_scores[str(layer_size)]=np.mean(cv_scores)
        pass
pm=Pima()
pm.load_data()
pm.adjust_data()
pm.reload_data()
pm.df.loc[pm.df['Glucose']==0,:]
# pm.SkinMLP(3)
sns.distplot(pm.X_skin)
plt.show()