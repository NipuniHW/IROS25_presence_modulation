import random
import qi
from mdp_formulation import *

# Camera resolution constants
kQQVGA = 160  # 160x120
kQVGA = 320  # 320x240
kVGA = 640  # 640x480
k4VGA = 1280  # 1280x960
k16VGA = 2560  # 2560x1920

# Color spaces constants
kYUV422ColorSpace = 9
kRGBColorSpace = 11
kBGRColorSpace = 13
kHSVColorSpace = 15

config = high_gaze_config

class Pepper:
    def __init__(self):
        self.session = qi.Session()
        self.is_connected = False
        
    def __del__(self):
        # Cleanup actions when the object is destroyed
        self.behavior_mng_service.stopAllBehaviors()        
        if self.is_connected:
            print("Disconnecting from the robot...")
            self.video_proxy.unsubscribe(self.subscriber_id)
            self.session.close()
            print("Session closed.")
    
    def connect(self, ip, port):
        # Connect to the robot
        print("Connect to the robot...")
        try:
            self.session.connect("tcp://{0}:{1}".format(ip, port))
            print("Session Connected....!")
            self.is_connected = True
        except Exception as e:
            print("Could not connect to Pepper:", e)
            self.is_connected = False
            return
        
        self.tts = self.session.service("ALTextToSpeech")
        self.leds = self.session.service("ALLeds")
        
        self.video_proxy = self.session.service("ALVideoDevice") #, PEPPER_IP, PORT)
        
        self.camera_id = 0  # 0 = Top Camera, 1 = Bottom Camera
        self.resolution = k4VGA #--switched#kQVGA  # 320x240 resolution
        self.color_space = kBGRColorSpace  # OpenCV expects BGR format
        self.fps = 5  # Frames per second
        
        # Subscribe to Pepper’s camera
        self.subscriber_id = self.video_proxy.subscribeCamera("pepper_cam", 
                                                              self.camera_id, 
                                                              self.resolution, 
                                                              self.color_space, 
                                                              self.fps)

        self.posture_service = self.session.service("ALRobotPosture")
        self.posture_service.goToPosture("StandInit", 1.0)
        
        self.behavior_mng_service = self.session.service("ALBehaviorManager")

    # To update volume
    def update_volume(self, volume):    
        volume_n = round(max(0, volume/10), 1)
        # print(f"Volume_n: {volume, volume_n}")
        self.tts.setVolume(volume_n)
        
        # List of random greetings or catchphrases
        greetings = [
            "Hello there!",
            "How's it going?",
            "Nice to see you!",
            "What's up?",
            "Greetings!",
            "Hey, how are you?",
            "Good day!",
            "Hi there!",
            "Howdy!",
            "Welcome!",
            "beep boop beep",
            "I am here!",
            "Hello, human!",
            "beep beep beep" # just for Damith
        ]    
        # Randomly pick a greeting
        random_greeting = random.choice(greetings)
        self.tts.say(random_greeting)
        
    # To update movements
    def update_movements(self, movement):
        self.behavior_mng_service.stopAllBehaviors()
        self.behavior_mng_service.startBehavior("modulated_actions/" + str(movement)) 
    
    # To update lights
    def update_lights(self, light):
        if light == 0:
            light_n = 0.1
        else:
            light_n = round(max(0, light/10), 1)
        leds = self.leds
        # pdb.set_trace()
        self.set_all_leds(leds, light_n)    
        
    def set_all_leds(self, leds, light_n):
        for led in config.led_actuators:
            leds.setIntensity(led, light_n)
            
    # Function to execute an action
    def execute_action(self, light, movement, volume):
        self.update_lights(light)
        self.update_movements(movement)
        self.update_volume(volume)
        
    def update_behavior(self, action, light, movement, volume):
        # pdb.set_trace()
        l_action, m_action, v_action = action.split(', ')
        
        if l_action == "Increase L":
            light = min(10, light + 1)
        elif l_action == "Decrease L":
            light = max(0, light - 1)
            
        if m_action == "Increase M":
            movement = min(5, movement + 1)
        elif m_action == "Decrease M":
            movement = max(0, movement - 1)
                
        if v_action == "Increase V":
            volume = min(10, volume + 1)
        elif v_action == "Decrease V":
            volume = max(0, volume - 1)
                
        # Keep Same
        if l_action == "Keep L":
            light = light
        elif m_action == "Keep M":
            movement = movement
        elif v_action == "Keep V":
            volume = volume
        
        print(f"###Action###: {action}")
        print(f"Light: {light}, Movement: {movement}, Volume: {volume}")
        
        # Perform the action
        self.execute_action(light, movement, volume)
        return light, movement, volume     
        

        
