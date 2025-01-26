import cv2

def view_rtsp_stream(rtsp_url):
    # Create a VideoCapture object
    cap = cv2.VideoCapture(rtsp_url)

    # Check if the connection is opened successfully
    if not cap.isOpened():
        print("Error: Cannot open RTSP stream.")
        return

    print("Press 'q' to exit.")

    while True:
        # Read frames from the stream
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to fetch frame from the stream.")
            break

        # Display the frame
        cv2.imshow('RTSP Stream', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Replace with your RTSP URL
rtsp_url = "rtsp://admin:HPGVFX@192.168.1.9:554/H.264"
view_rtsp_stream(rtsp_url)
