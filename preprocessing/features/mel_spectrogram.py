import librosa
import numpy as np


def extract_mel_spectrogram(
    audio,
    sample_rate=16000,
    n_mels=64,
    n_fft=512,
    hop_length=160
):
    """
    Extract a log-Mel spectrogram.

    Parameters
    ----------
    audio : np.ndarray
        Audio waveform.

    sample_rate : int
        Sampling rate.

    n_mels : int
        Number of Mel filters.

    n_fft : int
        FFT size.

    hop_length : int
        Hop length.

    Returns
    -------
    np.ndarray
        Log-Mel spectrogram.
    """

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return log_mel
