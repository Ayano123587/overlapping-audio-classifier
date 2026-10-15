import math
import wave
from pathlib import Path
import numpy as np
import torch
from scipy.signal import resample_poly

SAMPLE_RATE = 8000
SAMPLES = 8000

def load_wave(path, sample_rate=SAMPLE_RATE, seconds=1.):
    with wave.open(str(path), "rb") as f:
        width, channels, rate = f.getsampwidth(), f.getnchannels(), f.getframerate()
        if width != 2:
            raise ValueError("Only uncompressed 16-bit PCM WAV is supported")
        raw = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32)/32768
    raw = raw.reshape(-1, channels).mean(axis=1)
    if rate != sample_rate:
        gcd=math.gcd(rate,sample_rate)
        raw=resample_poly(raw,sample_rate//gcd,rate//gcd).astype(np.float32)
    length=round(sample_rate*seconds)
    # Center crop gives a deterministic inference/evaluation window.
    start=max(0,(len(raw)-length)//2)
    return torch.from_numpy(np.pad(raw[start:start+length],(0,max(0,length-len(raw)))).copy())

def write_wave(path, samples, rate=SAMPLE_RATE):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with wave.open(str(path),"wb") as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(rate)
        f.writeframes((np.clip(np.asarray(samples),-1,1)*32767).astype("<i2").tobytes())

def spectrogram(waveform):
    if waveform.ndim == 1: waveform=waveform[None]
    if waveform.shape[-1] < 256: waveform=torch.nn.functional.pad(waveform,(0,256-waveform.shape[-1]))
    spectrum=torch.stft(waveform,n_fft=256,hop_length=128,window=torch.hann_window(256,device=waveform.device),return_complex=True)
    result=torch.log1p(spectrum.abs().square())
    return result[:,None,:,:]

def augment_wave(waveform, generator):
    gain=.7+.6*torch.rand((),generator=generator)
    shift=int(torch.randint(-800,801,(),generator=generator))
    noise=torch.randn(waveform.shape,generator=generator)*.01
    return torch.roll(waveform,shift,-1)*gain+noise

def augment_spec(spec, generator):
    result=spec.clone()
    for axis,width in [(-2,8),(-1,5)]:
        size=result.shape[axis]
        length=int(torch.randint(0,min(width,size)+1,(),generator=generator))
        start=int(torch.randint(0,size-length+1,(),generator=generator))
        index=[slice(None)]*result.ndim;index[axis]=slice(start,start+length)
        result[tuple(index)]=0
    return result
