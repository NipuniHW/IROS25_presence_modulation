import cv2

def check_camera(camera_id=2):
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Error: Camera {camera_id} could not be opened.")
        return

    print(f"Camera {camera_id} is working. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        cv2.imshow(f"Camera {camera_id} Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera test completed.")

if __name__ == "__main__":
    check_camera(2)
