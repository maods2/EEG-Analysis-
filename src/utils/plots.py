
from xmlrpc.client import FastParser
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics

def plot_surface(eeg_signal, channel_ticks, channel_string, title):
    fig = go.Figure(data=[go.Surface(z=eeg_signal)])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                    highlightcolor="limegreen", project_z=True))
    fig.update_layout(title=title, autosize=False,
                    scene_camera_eye=dict(x=1.87, y=0.88, z=-0.64),
                    width=800, height=600,
                    margin=dict(l=20, r=20, b=50, t=50),
                    scene=dict(
                            xaxis=dict(
                                title='Time (sample num)',
                            ),
                            yaxis=dict(
                                title='Channel',
                                tickvals=channel_ticks,
                                ticktext=channel_string,
                            )
                           
                        )
                    )
    fig.show()

def plot_specgram(signal, fs, title ):
    plt.title(title)
    plt.grid(False)
    plt.specgram(signal,  NFFT=200, Fs=fs, noverlap=30, cmap='inferno')
    plt.colorbar()
    plt.show()

def plot_fft_welch(signal, fs, title ):
    f, pxx_den = welch(signal, fs=fs)
    plt.title(title)
    plt.grid(True, alpha=0.5)
    plt.plot(pxx_den)
    plt.show()


def plot_acc_loss_keras(fitted_model, y_true, y_pred):
    fig, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(15,5))
    fig.suptitle(f'Métrics ', fontsize=16)

    axs[0].plot(fitted_model.history['accuracy'],  lw=2)
    axs[0].plot(fitted_model.history['val_accuracy'],  lw=2)
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('accuracy')
    axs[0].set_title('model accuracy')
    

    axs[1].plot(fitted_model.history['loss'],  lw=2)
    axs[1].plot(fitted_model.history['val_loss'],  lw=2)
    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('loss')
    axs[1].set_title('model loss');   
    
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,ax=axs[2],values_format='d')
    axs[2].set_title('Matriz de Confusão');
    axs[2].grid(False)



def get_metrics(y_true, y_pred, cross_val_score, model_name, params, print_scores=False):
    # print("\n")

    

    accucaracy = metrics.accuracy_score(y_true, y_pred)
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_true, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_true, y_pred)
    mathew_coef = matthews_corrcoef(y_true, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)   
    auc = metrics.auc(fpr, tpr)

    if print_scores:
        print("-"*100)
        print(f"MODEL: {model_name}")

        print("-"*50)
        print("\n")
        print(classification_report(y_true, y_pred),'\n')

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
      "cross_val_score":cross_val_score, 
      "Accuracy":accucaracy, 
      "Precision":precision, 
      "Recall":recall, 
      "F1_score":fscore, 
      "auc":auc, "kappa": kappa, 
      "mathew_coef":mathew_coef, 
      "dataset": params['dataset'], 
      "subject": params['subject'], 
      "pipeline": params['pipeline'], 
      }
    