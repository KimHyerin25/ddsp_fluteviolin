# DDSP_SoundSynthesis

The dataset consists only of solo flute and violin instrument samples from MusicNet. The license for the full dataset is held by MusicNet, and you can find more details on the [MusicNet record page on Zenodo](https://zenodo.org/records/5120004).

```bash
!tar -xvf musicnet_fluteviolin.tar.gz
```

## Prepare your audio data to convert audio's sampling rate to 16k

```bash
ffmpeg -y -loglevel fatal -i $input_file -ac 1 -ar 16000 $output_file
```

## Use CREPE to precalculate the (time, frequency, confidence) from your audio

```bash
crepe directory-to-audio/ --output directory-to-audio/f0_0.004/  --viterbi --step-size 4
# if it doesn't work please do with just crepe and your audio file path
```
Edit the `config.yaml` file to fit your needs (audio location, preprocess folder, sampling rate, model parameters...), then preprocess your data using 

```bash
python preprocess.py
python train.py
python inference.py --input_wav {your input wave file path}
```

finally you will have 'recon_audio.wav' in your directory.
If you succeed to timbre transfer your audio file, keep moving on for sound synthesis of the audio.
keep going with your generation!

# Sound Synthesis with Autoencoder

Please check if the audio file has been successfully converted using CREPE. If the conversion is complete, you are ready to begin.


```bash
python autoencoder/train.py --data_dir {your dataset path}
python autoencoder/inference.py --pt_dir {your model pt file path} --test_dir {your test dataset directory} 
```

The result will be saved as a wave file named 'mixed_audio_{current time}'. Enjoy the transformation!





