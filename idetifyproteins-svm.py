import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import normalize
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pickle
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
import joblib
data_cv=[]
traffic_feature_cv=[]
traffic_target_cv=[]
csv_file_cv = csv.reader(open('pssm_cv_350.csv'))
for content in csv_file_cv:
    content=list(map(float,content))
    if len(content)!=0:
        data_cv.append(content)
        traffic_feature_cv.append(content[0:-1])
        traffic_target_cv.append(content[-1])

data_ind=[]
traffic_feature_ind=[]
traffic_target_ind=[]
csv_file_ind = csv.reader(open('pssm_ind_350.csv'))
for content in csv_file_ind:
    content=list(map(float,content))
    if len(content)!=0:
        data_ind.append(content)
        traffic_feature_ind.append(content[0:-1])
        traffic_target_ind.append(content[-1])
        
'''
scaler = StandardScaler() # 标准化转换
scaler.fit(traffic_feature_cv)  # 训练标准化对象 标准化归一化
traffic_feature_cv = scaler.transform(traffic_feature_cv)   # 转换数据集
#scaler.fit(traffic_feature_ind)
traffic_feature_ind = scaler.transform(traffic_feature_ind)
#feature_train, feature_test, target_train, target_test = train_test_split(traffic_feature, traffic_target, test_size=0.3,random_state=0)
#clf = RandomForestClassifier(criterion='entropy')
'''

#线性归一化
scaler = MinMaxScaler() # 线性归一化转换
#scaler.fit(traffic_feature_cv)  
traffic_feature_cv = scaler.fit_transform(traffic_feature_cv)   # 转换数据集
#scaler.fit(traffic_feature_ind)
traffic_feature_ind = scaler.transform(traffic_feature_ind)

'''
#L2正则化
normalizer = Normalizer()
Normalizer(copy=True, norm='l1')
normalizer.fit(traffic_feature_cv)
traffic_feature_cv = normalizer.transform(traffic_feature_cv)
traffic_feature_ind = normalizer.transform(traffic_feature_ind)
'''

clf = svm.SVC(kernel='rbf', C=11, gamma=0.1, probability=True)
clf.fit(traffic_feature_cv,traffic_target_cv)
cv_score_pre = cross_val_score(clf,traffic_feature_cv,traffic_target_cv,cv=10).mean()
print(cv_score_pre)
cv_val_predict = cross_val_predict(clf,traffic_feature_cv,traffic_target_cv,cv=10)
print(confusion_matrix(traffic_target_cv, cv_val_predict))
predict_results=clf.predict(traffic_feature_ind)
print(accuracy_score(predict_results, traffic_target_ind))


conf_mat = confusion_matrix(traffic_target_ind, predict_results)
print(conf_mat)
print(classification_report(traffic_target_ind, predict_results))

print('ACC',accuracy_score(predict_results, traffic_target_ind))

#SN
print('SN:',conf_mat[1][1]/(conf_mat[1][0]+conf_mat[1][1]))
#SP
print('SP:',conf_mat[0][0]/(conf_mat[0][0]+conf_mat[0][1]))
#MCC
mcc1 = matthews_corrcoef(traffic_target_ind, predict_results)
print('MCC',mcc1)
#F1
f11=f1_score(traffic_target_ind, predict_results, pos_label=1)
print('f1',f11)
#Precision
pre1 =  precision_score(traffic_target_ind, predict_results, average='macro')
print('precision',pre1)
#ROC
prob_predict_target_ind = clf.predict_proba(traffic_feature_ind)
predictions_ind = prob_predict_target_ind[:,1]
fpr,tpr,_ = roc_curve(traffic_target_ind,predictions_ind)
roc_auc = auc(fpr,tpr)
plt.title('ROC ind')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
