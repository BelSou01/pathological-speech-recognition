import librosa
import numpy as np


def extract_mfcc(
    audio,
    sample_rate=16000,
    n_mfcc=13,
    n_fft=512,
    hop_length=160
):
    """
    Extract MFCC features from an audio signal.

    Parameters
    ----------
    audio : np.ndarray
        Audio waveform.

    sample_rate : int
        Sampling rate.

    n_mfcc : int
        Number of MFCC coefficients.

    n_fft : int
        FFT size.

    hop_length : int
        Hop length.

    Returns
    -------
    np.ndarray
        MFCC feature matrix.
    """

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )

    return mfcc
