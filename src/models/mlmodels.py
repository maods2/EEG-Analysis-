from utils.feature_extraction import dtw_fast
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def KNNDWN_model_predict(X_train,X_test,y_train,y_test, scoring=None):
  KNN = KNeighborsClassifier(metric=dtw_fast)
  param_grid = {'n_neighbors': [9], 'weights':['distance']}
  knn_model = GridSearchCV(KNN, param_grid, cv=5,scoring=scoring)
  knn_model.fit(X_train, y_train)
  y_pred = knn_model.predict(X_test)
  best_score = round(knn_model.best_score_,3)
  return y_test, y_pred, best_score

def KNN_model_predict(X_train,X_test,y_train,y_test, scoring=None):
  KNN = KNeighborsClassifier()
  param_grid = {'n_neighbors': [9]}
  knn_model = GridSearchCV(KNN, param_grid, cv=5,scoring=scoring)
  knn_model.fit(X_train, y_train)
  y_pred = knn_model.predict(X_test)
  best_score = round(knn_model.best_score_,3)
  return y_test, y_pred, best_score

def SVM_model_predict(X_train,X_test,y_train,y_test,scoring=None):
  svm = SVC(decision_function_shape='ovo')
  param_grid = { 'kernel':['linear']}
  svm_model = GridSearchCV(svm, param_grid, cv=5,scoring=scoring)  
  svm_model.fit(X_train, y_train)
  y_pred = svm_model.predict(X_test)
  best_score = round(svm_model.best_score_,3)
  return y_test, y_pred, best_score

def XGB_model_predict(X_train,X_test,y_train,y_test,scoring=None):
  xgb = XGBClassifier(objective='multi:softmax',random_state=0)
  param_grid = {
        # 'min_child_weight': [1, 5, 10],
        # 'gamma': [0.5, 1, 1.5, 2, 5],
        # 'subsample': [0.6, 0.8, 1.0],
        # 'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [ 5],
        # 'n_estimators': [50,100,200]
        }
  xgb_model = GridSearchCV(xgb, param_grid, cv=5,scoring=scoring)  
  xgb_model.fit(X_train, y_train)
  y_pred = xgb_model.predict(X_test)
  best_score = round(xgb_model.best_score_,3)
  return y_test, y_pred, best_score


def LDA_model_predict(X_train,X_test,y_train,y_test,scoring=None):
  lda = LDA()
  param_grid = {'solver':['svd', 'eigen'] }
  lda_model = GridSearchCV(lda, cv=5,param_grid=param_grid, scoring=scoring)  
  lda_model.fit(X_train, y_train)
  y_pred = lda_model.predict(X_test)
  best_score = round(lda_model.best_score_,3)
  return y_test, y_pred, best_score