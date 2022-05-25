import numpy as np
from sklearn.model_selection import train_test_split
import timeit
from datetime import timedelta
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics
import matplotlib.pyplot as plt

class PipelineBuilder:
  def __init__(self, multidimentional_data=False, params=dict()):

    self._pipes = []
    self._pipe_names = []
    self._models = []
    self._model_names = []
    self._multidimentional_data = multidimentional_data
    self.params = params
    
  def add_pipe(self, pipe_object, pipe_name:str):
     self._pipes.append( pipe_object)
     self._pipe_names.append( pipe_name)

  def add_model(self, model_object, model_name:str):
     self._models.append(model_object)
     self._model_names.append(model_name)

  def process(self, x, y):
    data = x
    if self._pipes:
      for pipe_transform, pipe_name in zip(self._pipes, self._pipe_names):
        print(f"\nProcessing step {pipe_name}...")
        print(f"Data Shape: {np.shape(data)}...")
        if self._multidimentional_data and  (len(np.shape(data)) == 3):
          data = [pipe_transform(d, self.params) for d in data]
        else:
          data = pipe_transform(data, self.params)

    if (self._multidimentional_data ) and (len(np.shape(data)) == 3):
      data = self._flatten(data)

    x_train, x_test,y_train,y_test = train_test_split(data, y, test_size=0.20, stratify=y, random_state=28)
    models_metrics = []

    if self._models and (len(self._models) == len(self._model_names)):

      for model_predict, model_name in zip(self._models, self._model_names):
        
        print(f"\nTraining {model_name} model...")
        print(f"Data Shape: {np.shape(data)}...")
        start = timeit.default_timer()
        y_test, y_pred, val_score = model_predict(x_train,x_test,y_train,y_test)

        stop = timeit.default_timer()
        model_runtime = stop - start

        metrics = self._run_metrics(y_test, y_pred, val_score, model_name, model_runtime)
        models_metrics.append(metrics)

    return data, models_metrics


  def get_pipes(self):
    return ' -> '.join(self._pipe_names)

  def _flatten(self, data):
    samples, channels, data_points = np.shape(data)
    new_array = np.array(data).reshape(samples, channels * data_points)
    return new_array

  def _run_metrics(self, y_true, y_pred, cross_val_score, model_name, model_runtime):
    # print("\n")
    print("-"*100)
    print(f"PIPELINE: {self.get_pipes()}")
    print(f"MODEL: {model_name}")
    print(f"Runtime: {str(timedelta(seconds=model_runtime))}")
    print("-"*50)
    print("\n")

    print(classification_report(y_true, y_pred),'\n')


    accucaracy = metrics.accuracy_score(y_true, y_pred)
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_true, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_true, y_pred)
    mathew_coef = matthews_corrcoef(y_true, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)   
    auc = metrics.auc(fpr, tpr)


    print(f'Train Data:\n')
    print(f"Cross validation score {'ACC'}: {cross_val_score} \n")
    print(f'Test Data:\n')
    print(f'Accuracy: {accucaracy} \n')
    print(f'Precision: {precision} \n')
    print(f'Recall: {recall} \n')
    print(f'F1 Score: {fscore} \n')
    print(f'Area Under Curve: {auc} \n')
    print(f'Cohen_kappa: {kappa} \n')
    print(f'Matthews Coef: {mathew_coef} \n')

    full_pipeline = self.get_pipes() + ' => ' + model_name


    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(15,7))
    fig.suptitle(f'Métricas: {model_name}', fontsize=16)

    axs[0].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
    axs[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axs[0].set_ylim([0.0, 1.05])
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[0].set_title('Curva ROC')
    axs[0].legend(loc="lower right")

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,ax=axs[1],values_format='d')
    axs[1].set_title('Matriz de Confusão');
    axs[1].grid(False)

    return { 
      "model_name":full_pipeline, 
      "cross_val_score":cross_val_score, 
      "Accuracy":accucaracy, 
      "Precision":precision, 
      "Recall":recall, 
      "F1_score":fscore, 
      "auc":auc, "kappa": kappa, 
      "mathew_coef":mathew_coef, 
      "model_runtime":str(timedelta(seconds=model_runtime)),
      "dataset": self.params['dataset'], 
      "subject": self.params['subject'], 
      }
