# Leap Myo

This project correlates hand landmarks from a Leap Motion Controller with EMG data from a Myo Armband.

## Requirements

- Python
- [Leap Motion Controller](https://en.wikipedia.org/wiki/Leap_Motion#Technology)
- [Myo Armband](https://github.com/thalmiclabs)

## Getting started

- Install [Ultraleap Gemini](https://leap2.ultraleap.com/gemini-downloads/)
- Install the necessary python packages `pip3 install -m requirements.txt`
- Charge your Myo Armband and have your Leap Motion Controlller on hand

## Usage

Both EMG data and hand landmarks are PII, so you'll need to train the model on your own data.

### Collecting data

1. Connect your Leap Motion Controller and have your charged Myo Armband on hand
2. Run `collect.py` for 15 minutes, moving your hands within the frame of the Leap Motion Controller

When you quit the python process, collected data will be stored in a pandas `DataFrame` with each row containing a timestamp, 8 channels of EMG information and hand landmarks.

### Training and testing the model

1. Run `train.py` to train a model on the collected data and plot results
2. With your Myo Armband on, predict hand landmarks with `predict.py`

## Resources

- [NeuroPose](https://par.nsf.gov/servlets/purl/10295971)
- [Keras](https://keras.io/getting_started/intro_to_keras_for_engineers/)
- [Leap API](https://github.com/ultraleap/leapc-python-bindings/tree/main)
- [Myo API](https://github.com/PerlinWarp/pyomyo/blob/main/src/pyomyo/pyomyo.py)
