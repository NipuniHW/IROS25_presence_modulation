# IROS25_presence_modulation
Work for IROS 2025

# Presence Modulation
This project covers a context identification model based on audio inputs for presence modulation of robots. It has 3 modules:
1. An ambient sound classification using a Convolutional Neural Network (model.h5)
2. Speech to text keyword detection
3. Speech to text sentiment analysis

These three modules have been statistically merged using a Naive Bayes classifier (NB_model.joblib). It finally gives the context as one of five predefined states:
- Alarmed (Emergency situation)
- Alert (Cautious on whether the situation is emergency or not)
- Social (Everyday casual interactions)
- Passive (Awake and aware on the situation, with little to no movements/ itneractions)
- Disengaged (Silent/ Off mode)

This proect is implemented for Pepper robot (Softbank Robotics) and uses Choreographe software to define behaviours for each state. Robot will also speak state relevantly.

# Installation

1. git clone https://github.com/NHWCode/presence_modulation.git
2. cd presence_modulation
3. pip install -r requirements.txt    # For dependencies

# File Details
1. Folder: context_identification
    - Contains context identification and presence modulation related files
2. Folder:presence_actions
    - Contains Pepper's predefined behaviours for each state

# Usage

Follow below steps to run the Pepper robot with context identification and presence modulation (speech & behaviour):
1. Complete installation
2. Connect to physical Pepper in Choreographe
3. Open "presence_modulation.py"
4. Add ip address and port for physical Pepper
5. Include relevant paths for CNN model (model.h5), NB model (NB_model.joblib).
6. Add whisper API key
7. Run command in the terminal: python pepper_presence_modulation.py

Follow below steps to run the context identification module separately:
1. Complete installation
2. Open "context_classifier.py"
3. Add whisper API key
4. Include relevant paths for CNN model (model.h5), NB model (NB_model.joblib).
5. Run command in the terminal: python context_classifier.py

# License

This project is licensed under the MIT License.

