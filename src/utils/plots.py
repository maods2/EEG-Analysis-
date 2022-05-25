
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.signal import welch

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