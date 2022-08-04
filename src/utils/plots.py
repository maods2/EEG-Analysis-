
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


def plot_specgram(signal, fs, title):
    plt.title(title)
    plt.grid(False)
    plt.specgram(signal,  NFFT=200, Fs=fs, noverlap=30, cmap='inferno')
    plt.colorbar()
    plt.show()


def plot_fft_welch(signal, fs, title):
    f, pxx_den = welch(signal, fs=fs)
    plt.title(title)
    plt.grid(True, alpha=0.5)
    plt.plot(pxx_den)
    plt.show()


def plot_acc_loss_keras(fitted_model, y_true, y_pred, model_name, subject):

    accucaracy = metrics.accuracy_score(y_true, y_pred)
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(
        y_true, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_true, y_pred)
    mathew_coef = matthews_corrcoef(y_true, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(9, 5))
    train_sub_title = model_name if subject == "All" else model_name + " | " + subject
    fig.suptitle(f'Training Accuracy - {train_sub_title}', fontsize=16)

    axs.plot(fitted_model.history['accuracy'],  lw=2)
    axs.plot(fitted_model.history['val_accuracy'],  lw=2)
    axs.set_xlabel('epoch')
    axs.set_ylabel('accuracy')
    # axs.set_title('model accuracy')
    file_sub_title = f'src/artifacts/acc_{model_name.replace(" ","_").replace("->","")}.png' if subject == "All" else f'src/artifacts/acc_{model_name.replace(" ","_").replace("->","")}_{subject}.png'
    plt.savefig(file_sub_title, bbox_inches='tight')
    plt.close(fig)

    fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(9, 5))
    fig.suptitle(f'Training Loss - {train_sub_title}', fontsize=16)
    axs.plot(fitted_model.history['loss'],  lw=2)
    axs.plot(fitted_model.history['val_loss'],  lw=2)
    axs.set_xlabel('epoch')
    axs.set_ylabel('loss')
    # axs.set_title('model loss')
    file_sub_title = f'src/artifacts/loss_{model_name.replace(" ","_").replace("->","")}.png' if subject == "All" else f'src/artifacts/loss_{model_name.replace(" ","_").replace("->","")}_{subject}.png'
    plt.savefig(file_sub_title, bbox_inches='tight')
    plt.close(fig)

    fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(9, 5))
    fig.suptitle(f'Matriz de Confusão - {train_sub_title}', fontsize=16)
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=axs, values_format='d')
    # axs.set_title('Matriz de Confusão')
    axs.grid(False)
    file_sub_title = f'src/artifacts/cm_{model_name.replace(" ","_").replace("->","")}.png' if subject == "All" else f'src/artifacts/cm_{model_name.replace(" ","_").replace("->","")}_{subject}.png'
    plt.savefig(file_sub_title, bbox_inches='tight')
    plt.close(fig)

    return {
        "acc_test": accucaracy,
        "Precision": precision,
        "Recall": recall,
        "F1_score": fscore,
        "auc": auc, "kappa": kappa,
        "mathew_coef": mathew_coef,
        "model_name": model_name
    }


def get_metrics(y_true, y_pred, cross_val_score, model_name, params, print_scores=False):
    # print("\n")

    accucaracy = metrics.accuracy_score(y_true, y_pred)
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(
        y_true, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_true, y_pred)
    mathew_coef = matthews_corrcoef(y_true, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    if print_scores:
        print("-"*100)
        print(f"MODEL: {model_name}")

        print("-"*50)
        print("\n")
        print(classification_report(y_true, y_pred), '\n')

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

    fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(9, 5))

    train_sub_title = params["pipeline"] + " + " + \
        model_name if params['subject'] == "All" else params["pipeline"] + \
        " + "+model_name+" | " + params['subject']
    fig.suptitle(
        f'Métricas: {train_sub_title }', fontsize=16)

    axs.plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC curve (area = %0.2f)' % auc)
    axs.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axs.set_ylim([0.0, 1.05])
    axs.set_xlabel('False Positive Rate')
    axs.set_ylabel('True Positive Rate')
    axs.set_title('Curva ROC')
    axs.legend(loc="lower right")
    file_sub_title = f'src/artifacts/roc_{params["pipeline"].replace(" ","_").replace("->","") + "_"+model_name}.png' if params[
        'subject'] == "All" else f'src/artifacts/roc_{params["pipeline"].replace(" ","_").replace("->","") + "_"+model_name}_{params["subject"]}.png'

    plt.savefig(file_sub_title, bbox_inches='tight')
    plt.close(fig)

    fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(9, 5))
    fig.suptitle(
        f'Métricas: {train_sub_title }', fontsize=16)

    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=axs, values_format='d')
    axs.set_title('Matriz de Confusão')
    axs.grid(False)

    file_sub_title = f'src/artifacts/cm_{params["pipeline"].replace(" ","_").replace("->","") + "_"+model_name}.png' if params[
        'subject'] == "All" else f'src/artifacts/cm_{params["pipeline"].replace(" ","_").replace("->","") + "_"+model_name}_{params["subject"]}.png'
    plt.savefig(file_sub_title, bbox_inches='tight')
    plt.close(fig)

    return {
        "cross_val_score": cross_val_score,
        "Accuracy": accucaracy,
        "Precision": precision,
        "Recall": recall,
        "F1_score": fscore,
        "auc": auc, "kappa": kappa,
        "mathew_coef": mathew_coef,
        "dataset": params['dataset'],
        "subject": params['subject'],
        "pipeline": params['pipeline'],
        "model_name": model_name
    }
