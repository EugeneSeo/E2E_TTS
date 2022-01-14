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
    
2. Time for improvement!