#Import library
import qi

class Connection:
    def __init__(self):
        self.session = qi.Session()

    def connect(self, ip, port):
        # Connect to the robot
        print("Connect to the robot...")
        try:
            self.session.connect("tcp://{0}:{1}".format(ip, port))
            print("Session Connected....!")
            self.behavior_mng_service = self.session.service("ALBehaviorManager")
            print("Stopping all the actions")
            self.behavior_mng_service.stopAllBehaviors() 
            self.session.close()
    
        except Exception as e:
            print("Could not connect to Pepper:", e)
            exit(1)
            
if __name__=="__main__":
    # peper = Connection
    pepper = Connection()
    # pepper.connect("localhost", 36227)
    pepper.connect('pepper.local', 9559)