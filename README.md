# E2E_TTS

Plans
----------

1. [01/14/2022] 
E2E-TTS model (StyleSpeech + HiFi-GAN) fine-tuning process is completed. 
    - Finetuning model: lmel_hifi:lmel_ss = 1:45, lr = 2e-6, steps: 5000
    - Quantitative results
        | Data | Model | WER | SIM | 
        | :-------------: | :---------------: | :---------------: | :---------------: |
        | - | gt | 4.969626777 | 1 |
        | val | no finetuning | 6.233312315 | 0.8410108297 |
        | val | finetuning | 6.123760538 | 0.8775588885 |
        | unseen | no finetuning | 6.969311073 | 0.8302679062 |
        | unseen | finetuning | 6.448718703 | 0.8699259364 |
    
2. [07/18/2022]
We are currently developing E2E-TTS StyleSpeech architecture. 
    - Changes
        - Originally, StyleSpeech extracted style vectors from Mel-spectrograms and generated Mel-spectrograms. For waveform quality, E2E-TTS uses lin-spectrograms. 
        - HiFi-GAN generator gets text encoded values and hidden states of spectrogram decoder as inputs. For hidden state processing, we adopt the Gaussian interpolation used in EATS.
        - We use both losses of StyleSpeech and HiFi-GAN. In addition, the r1 penalty is used for the discriminator training stage. 
        - DWT Discriminator based on Fre-GAN is used as the alternative to the HiFi-GAN discriminator. 
    - Model details
        - `models/StyleSpeech.py`
            - `StyleSpeech`: Original StyleSpeech model. Return values have been changed.
            - `StyleSpeech_attn`: Added multi-head attention of style attention and text encoded values right after the variance adaptor. 
            - `StyleSpeech_transformer`: Added multi-head attention of style attention and hidden state to the StyleSpeech's spectrogram decoder (Tranformer-like decoder).
            - `StyleSpeech_ali`: Alignment-learning StyleSpeech.
        - `models/Hifigan.py`
            - `Generator_intpol`: Current best model with interpolation issue resolved. Originally, we divided hidden states before interpolation. However, since the nearby information is critical to waveform generation, we changed the order.
            - `Generator_intpol_conv`: Current best model (HiFi-GAN + Hidden state interpolation).
            - `ConstantExpandFrame`: Interpolation frame s.t. copy & paste.
            - `ExpandFrame`: Gaussian interpolation frame which was used in EATS.
            - `DiscriminatorP_dwt, MultiPeriodDiscriminator_dwt, DiscriminatorS_dwt, MultiScaleDiscriminator_dwt`: DWT discriminator proposed in Fre-GAN.
    - best WER: 10.92 (Data: val)
    - Train & Evaluation commands
        ```console
        python3 train_e2e_lin.py
        python3 wer_evaluate_e2e_all.py
        ```