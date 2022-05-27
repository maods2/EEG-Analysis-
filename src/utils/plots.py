
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.metrics import ConfusionMatrixDisplay

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
    