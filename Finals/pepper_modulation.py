from connection import Connection

# Create a proxy to the AL services
behavior_mng_service = session.service("ALBehaviorManager")
tts = session.service("ALTextToSpeech")
leds = session.service("ALLeds")

# To update lights
def update_lights(light):
    if light == 0:
        light_n = 0.1
    else:
        light_n = round(max(0, light/10), 1)
        
    # print(f"Light_n: {light, light_n}")
    leds.setIntensity("Face/Led/Blue/Left/225Deg/Actuator/Value", light_n)
    leds.setIntensity("Face/Led/Blue/Left/270Deg/Actuator/Value", light_n)            
    leds.setIntensity("Face/Led/Green/Left/225Deg/Actuator/Value", light_n)
    leds.setIntensity("Face/Led/Green/Left/270Deg/Actuator/Value", light_n)
    leds.setIntensity("Face/Led/Red/Left/270Deg/Actuator/Value", light_n)
    
# To update volume
def update_volume(volume):    
    volume_n = round(max(0, volume/10), 1)
    print(f"Volume_n: {volume, volume_n}")
    tts.setVolume(volume_n)
    tts.say("Beep boop beep boop")

# To update movements
def update_movements(movement):
    behavior_mng_service.stopAllBehaviors()
    behavior_mng_service.startBehavior("attention_actions/" + str(movement)) 