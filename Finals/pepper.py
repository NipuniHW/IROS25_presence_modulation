import qi

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


# Create a proxy to access Pepper's camera


# Camera Settings



            