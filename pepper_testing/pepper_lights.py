# from connection import Connection
# # import qi
# # from dotenv import load_dotenv
# # from openai import OpenAI

# # Load the environment variables from the .env file
# # load_dotenv()

# # Initialize OpenAI client
# # client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # Connect Pepper robot
# pepper = Connection()
# session = pepper.connect('pepper.local', '9559')

# # Create a proxy to the AL services
# # behavior_mng_service = session.service("ALBehaviorManager")
# # tts = session.service("ALTextToSpeech")
# leds = session.service("ALLeds")

# # print(leds.listGroup("AllLeds"))

# # def update_lights(light):
# #     if light == 0:
# #         light_n = 0.1
# #     else:
# #         light_n = round(max(0, light/10), 1)
        
# # # print(f"Light_n: {light, light_n}")
# for i in range(10):
#     i = i/10
#     leds.setIntensity("AllLeds", i)
# #     leds.setIntensity("Ears/Led/Left/108Deg/Actuator/Value", i)            
# #     leds.setIntensity("Ears/Led/Left/144Deg/Actuator/Value", i)
# #     leds.setIntensity("Ears/Led/Left/180Deg/Actuator/Value", i)
# #     leds.setIntensity("Ears/Led/Left/216Deg/Actuator/Value", i)
# #     leds.setIntensity("Ears/Led/Left/252Deg/Actuator/Value", i)
# #     leds.setIntensity("Ears/Led/Left/288Deg/Actuator/Value", i)            
# #     leds.setIntensity("Ears/Led/Left/324Deg/Actuator/Value", i)
# #     leds.setIntensity("Ears/Led/Left/36Deg/Actuator/Value", i)
# #     leds.setIntensity("Ears/Led/Left/72Deg/Actuator/Value", i)
# #     leds.setIntensity("Ears/Led/Right/0Deg/Actuator/Value", i)
# #     leds.setIntensity("Ears/Led/Right/108Deg/Actuator/Value", i)            
# #     leds.setIntensity("Ears/Led/Right/144Deg/Actuator/Value", i)
# #     leds.setIntensity("Ears/Led/Right/180Deg/Actuator/Value", i)
# #     leds.setIntensity("Ears/Led/Right/216Deg/Actuator/Value", i)
# #     leds.setIntensity("Ears/Led/Right/252Deg/Actuator/Value", i)
# #     leds.setIntensity("Ears/Led/Right/288Deg/Actuator/Value", i)            
# #     leds.setIntensity("Ears/Led/Right/324Deg/Actuator/Value", i)
# #     leds.setIntensity("Ears/Led/Right/36Deg/Actuator/Value", i)
# #     leds.setIntensity("Ears/Led/Right/72Deg/Actuator/Value", i)

# print("Completed")


# # ['Ears/Led/Left/0Deg/Actuator/Value', 'Ears/Led/Left/108Deg/Actuator/Value', 
# #  'Ears/Led/Left/144Deg/Actuator/Value', 'Ears/Led/Left/180Deg/Actuator/Value', 
# #  'Ears/Led/Left/216Deg/Actuator/Value', 'Ears/Led/Left/252Deg/Actuator/Value', 
# #  'Ears/Led/Left/288Deg/Actuator/Value', 'Ears/Led/Left/324Deg/Actuator/Value', 
# #  'Ears/Led/Left/36Deg/Actuator/Value', 'Ears/Led/Left/72Deg/Actuator/Value', 
# #  'Ears/Led/Right/0Deg/Actuator/Value', 'Ears/Led/Right/108Deg/Actuator/Value',
# #  'Ears/Led/Right/144Deg/Actuator/Value', 'Ears/Led/Right/180Deg/Actuator/Value', 
# #  'Ears/Led/Right/216Deg/Actuator/Value', 'Ears/Led/Right/252Deg/Actuator/Value', 
# #  'Ears/Led/Right/288Deg/Actuator/Value', 'Ears/Led/Right/324Deg/Actuator/Value', 
# #  'Ears/Led/Right/36Deg/Actuator/Value', 'Ears/Led/Right/72Deg/Actuator/Value']

# # ['Face/Led/Blue/Left/0Deg/Actuator/Value', 'Face/Led/Blue/Left/135Deg/Actuator/Value', 
# #  'Face/Led/Blue/Left/180Deg/Actuator/Value', 'Face/Led/Blue/Left/225Deg/Actuator/Value', 
# #  'Face/Led/Blue/Left/270Deg/Actuator/Value', 'Face/Led/Blue/Left/315Deg/Actuator/Value', 
# #  'Face/Led/Blue/Left/45Deg/Actuator/Value', 'Face/Led/Blue/Left/90Deg/Actuator/Value', 
# #  'Face/Led/Blue/Right/0Deg/Actuator/Value', 'Face/Led/Blue/Right/135Deg/Actuator/Value', 
# #  'Face/Led/Blue/Right/180Deg/Actuator/Value', 'Face/Led/Blue/Right/225Deg/Actuator/Value', 
# #  'Face/Led/Blue/Right/270Deg/Actuator/Value', 'Face/Led/Blue/Right/315Deg/Actuator/Value', 
# #  'Face/Led/Blue/Right/45Deg/Actuator/Value', 'Face/Led/Blue/Right/90Deg/Actuator/Value', 
# #  'Face/Led/Green/Left/0Deg/Actuator/Value', 'Face/Led/Green/Left/135Deg/Actuator/Value', 
# #  'Face/Led/Green/Left/180Deg/Actuator/Value', 'Face/Led/Green/Left/225Deg/Actuator/Value',
# #  'Face/Led/Green/Left/270Deg/Actuator/Value', 'Face/Led/Green/Left/315Deg/Actuator/Value', 
# #  'Face/Led/Green/Left/45Deg/Actuator/Value', 'Face/Led/Green/Left/90Deg/Actuator/Value', 
# #  'Face/Led/Green/Right/0Deg/Actuator/Value', 'Face/Led/Green/Right/135Deg/Actuator/Value', 
# #  'Face/Led/Green/Right/180Deg/Actuator/Value', 'Face/Led/Green/Right/225Deg/Actuator/Value', 
# #  'Face/Led/Green/Right/270Deg/Actuator/Value', 'Face/Led/Green/Right/315Deg/Actuator/Value', 
# #  'Face/Led/Green/Right/45Deg/Actuator/Value', 'Face/Led/Green/Right/90Deg/Actuator/Value', 
# #  'Face/Led/Red/Left/0Deg/Actuator/Value', 'Face/Led/Red/Left/135Deg/Actuator/Value', 
# #  'Face/Led/Red/Left/180Deg/Actuator/Value', 'Face/Led/Red/Left/225Deg/Actuator/Value', 
# #  'Face/Led/Red/Left/270Deg/Actuator/Value', 'Face/Led/Red/Left/315Deg/Actuator/Value', 
# #  'Face/Led/Red/Left/45Deg/Actuator/Value', 'Face/Led/Red/Left/90Deg/Actuator/Value', 
# #  'Face/Led/Red/Right/0Deg/Actuator/Value', 'Face/Led/Red/Right/135Deg/Actuator/Value', 
# #  'Face/Led/Red/Right/180Deg/Actuator/Value', 'Face/Led/Red/Right/225Deg/Actuator/Value', 
# #  'Face/Led/Red/Right/270Deg/Actuator/Value', 'Face/Led/Red/Right/315Deg/Actuator/Value', 
# #  'Face/Led/Red/Right/45Deg/Actuator/Value', 'Face/Led/Red/Right/90Deg/Actuator/Value']

# # ['ChestBoard/Led/Blue/Actuator/Value', 'ChestBoard/Led/Green/Actuator/Value', 
# #  'ChestBoard/Led/Red/Actuator/Value']

# # def update_volume(volume):    
# #     volume_n = round(max(0, volume/10), 1)
# #     print(f"Volume_n: {volume, volume_n}")
# #     tts.setVolume(volume_n)
# #     tts.say("Beep boop beep boop")

# # def update_movements(movement):
# #     behavior_mng_service.stopAllBehaviors()
# #     behavior_mng_service.startBehavior("modulated_actions/" + str(movement)) 


# ### ALL LEDS AVAILABLE

# # ['ChestBoard/Led/Blue/Actuator/Value', 'ChestBoard/Led/Green/Actuator/Value', 'ChestBoard/Led/Red/Actuator/Value', 'Ears/Led/Left/0Deg/Actuator/Value', 'Ears/Led/Left/108Deg/Actuator/Value', 'Ears/Led/Left/144Deg/Actuator/Value', 'Ears/Led/Left/180Deg/Actuator/Value', 'Ears/Led/Left/216Deg/Actuator/Value', 'Ears/Led/Left/252Deg/Actuator/Value', 'Ears/Led/Left/288Deg/Actuator/Value', 'Ears/Led/Left/324Deg/Actuator/Value', 'Ears/Led/Left/36Deg/Actuator/Value', 'Ears/Led/Left/72Deg/Actuator/Value', 'Ears/Led/Right/0Deg/Actuator/Value', 'Ears/Led/Right/108Deg/Actuator/Value', 'Ears/Led/Right/144Deg/Actuator/Value', 'Ears/Led/Right/180Deg/Actuator/Value', 'Ears/Led/Right/216Deg/Actuator/Value', 'Ears/Led/Right/252Deg/Actuator/Value', 'Ears/Led/Right/288Deg/Actuator/Value', 'Ears/Led/Right/324Deg/Actuator/Value', 'Ears/Led/Right/36Deg/Actuator/Value', 'Ears/Led/Right/72Deg/Actuator/Value', 'Face/Led/Blue/Left/0Deg/Actuator/Value', 'Face/Led/Blue/Left/135Deg/Actuator/Value', 'Face/Led/Blue/Left/180Deg/Actuator/Value', 'Face/Led/Blue/Left/225Deg/Actuator/Value', 'Face/Led/Blue/Left/270Deg/Actuator/Value', 'Face/Led/Blue/Left/315Deg/Actuator/Value', 'Face/Led/Blue/Left/45Deg/Actuator/Value', 'Face/Led/Blue/Left/90Deg/Actuator/Value', 'Face/Led/Blue/Right/0Deg/Actuator/Value', 'Face/Led/Blue/Right/135Deg/Actuator/Value', 'Face/Led/Blue/Right/180Deg/Actuator/Value', 'Face/Led/Blue/Right/225Deg/Actuator/Value', 'Face/Led/Blue/Right/270Deg/Actuator/Value', 'Face/Led/Blue/Right/315Deg/Actuator/Value', 'Face/Led/Blue/Right/45Deg/Actuator/Value', 'Face/Led/Blue/Right/90Deg/Actuator/Value', 'Face/Led/Green/Left/0Deg/Actuator/Value', 'Face/Led/Green/Left/135Deg/Actuator/Value', 'Face/Led/Green/Left/180Deg/Actuator/Value', 'Face/Led/Green/Left/225Deg/Actuator/Value', 'Face/Led/Green/Left/270Deg/Actuator/Value', 'Face/Led/Green/Left/315Deg/Actuator/Value', 'Face/Led/Green/Left/45Deg/Actuator/Value', 'Face/Led/Green/Left/90Deg/Actuator/Value', 'Face/Led/Green/Right/0Deg/Actuator/Value', 'Face/Led/Green/Right/135Deg/Actuator/Value', 'Face/Led/Green/Right/180Deg/Actuator/Value', 'Face/Led/Green/Right/225Deg/Actuator/Value', 'Face/Led/Green/Right/270Deg/Actuator/Value', 'Face/Led/Green/Right/315Deg/Actuator/Value', 'Face/Led/Green/Right/45Deg/Actuator/Value', 'Face/Led/Green/Right/90Deg/Actuator/Value', 'Face/Led/Red/Left/0Deg/Actuator/Value', 'Face/Led/Red/Left/135Deg/Actuator/Value', 'Face/Led/Red/Left/180Deg/Actuator/Value', 'Face/Led/Red/Left/225Deg/Actuator/Value', 'Face/Led/Red/Left/270Deg/Actuator/Value', 'Face/Led/Red/Left/315Deg/Actuator/Value', 'Face/Led/Red/Left/45Deg/Actuator/Value', 'Face/Led/Red/Left/90Deg/Actuator/Value', 'Face/Led/Red/Right/0Deg/Actuator/Value', 'Face/Led/Red/Right/135Deg/Actuator/Value', 'Face/Led/Red/Right/180Deg/Actuator/Value', 'Face/Led/Red/Right/225Deg/Actuator/Value', 'Face/Led/Red/Right/270Deg/Actuator/Value', 'Face/Led/Red/Right/315Deg/Actuator/Value', 'Face/Led/Red/Right/45Deg/Actuator/Value', 'Face/Led/Red/Right/90Deg/Actuator/Value']

import cv2

for i in range(5):  # Try indexes 0 to 4
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} is available")
        cap.release()
