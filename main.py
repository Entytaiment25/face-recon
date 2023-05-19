import cv2
import dlib
import numpy as np
import psutil

print(cv2.cuda.getCudaEnabledDeviceCount())

# Create a video capture object for the webcam
cap = cv2.VideoCapture(0)

# Load the face detection model
face_detector = dlib.get_frontal_face_detector()

# Initialize the correlation tracker and green square toggle
face_tracker = None
show_green_square = True

# Toggle flag for night vision mode
night_vision_mode = False

# Toggle flag for face pixelation
pixelation_enabled = False
use_new_pixelation = False

# Variables for FPS calculation
start_time = cv2.getTickCount()
frame_count = 0

# Toggle flag for additional information display
show_additional_info = True

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if night_vision_mode:
        # Create a custom greyish color map
        custom_color_map = cv2.COLORMAP_BONE

        # Apply the custom color map to simulate night vision effect
        frame = cv2.applyColorMap(frame, custom_color_map)

    # Perform face detection every 10 frames or when face tracker is None
    if frame_count % 10 == 0 or face_tracker is None:
        # Detect faces in the grayscale frame
        faces = face_detector(gray, 1)

        if len(faces) > 0:
            # Initialize the correlation tracker with the first detected face
            face_tracker = dlib.correlation_tracker()
            face_tracker.start_track(frame, faces[0])

    if face_tracker is not None:
        try:
            # Update the correlation tracker and get the new bounding box
            face_tracker.update(frame)
            pos = face_tracker.get_position()

            # Extract the bounding box coordinates
            x = int(pos.left())
            y = int(pos.top())
            w = int(pos.width())
            h = int(pos.height())

            if w > 0 and h > 0:
                # Check if the face region is valid and within frame bounds
                if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1] and y + h <= frame.shape[0] and x + w <= frame.shape[1]:
                    if pixelation_enabled:
                        if use_new_pixelation:
                            # New Pixelation
                            pixelated_face = cv2.resize(frame[y:y + h, x:x + w], (w // 20, h // 20), interpolation=cv2.INTER_NEAREST)
                            pixelated_face = cv2.resize(pixelated_face, (w, h), interpolation=cv2.INTER_NEAREST)
                        else:
                            # Old Pixelation
                            pixelated_face = cv2.resize(frame[y:y + h, x:x + w], (w // 10, h // 10), interpolation=cv2.INTER_NEAREST)
                            pixelated_face = cv2.resize(pixelated_face, (w, h), interpolation=cv2.INTER_NEAREST)

                        if night_vision_mode:
                            # Apply the night vision effect to the pixelated face
                            pixelated_face = cv2.applyColorMap(pixelated_face, custom_color_map)

                        frame[y:y + h, x:x + w] = pixelated_face

                    # Draw the bounding box on the frame
                    if show_green_square:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Display the size and skin color of the detection
                        text = f"Size: {w}x{h}"
                        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                        skin_region = frame[y:y + h, x:x + w]
                        skin_color = skin_region.mean(axis=0).mean(axis=0).astype(int)
                        skin_color_text = f"Skin Color: {skin_color[2]}, {skin_color[1]}, {skin_color[0]}"
                        cv2.putText(frame, skin_color_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        except Exception as e:
            print(e)
            face_tracker = None

    # Calculate and display the FPS if additional info is enabled
    if show_additional_info:
        frame_count += 1
        elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        fps = frame_count / elapsed_time
        fps_text = f'FPS: {fps:.2f}'
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Get CPU and RAM load
        cpu_load = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent

        system_info_text = f'CPU: {cpu_load:.2f}%   RAM: {ram_usage:.2f}%'
        cv2.putText(frame, system_info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Display options on the frame
        options_text = [
            'Options:',
            '[N] Toggle Night Vision',
            '[H] Toggle Green Square',
            '[R] Reset Face Detection',
            '[F] Toggle Pixelation',
            '[T] Toggle Pixelation Method',
            '[I] Toggle Additional Info',
            '[Q] Quit'
        ]
        for i, text in enumerate(options_text):
            cv2.putText(frame, text, (10, 90 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    else:
        # Display "Press 'I' for menu" when additional info is hidden
        menu_text = "Press 'I' for menu"
        menu_text_size = cv2.getTextSize(menu_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        menu_text_pos = (frame.shape[1] - menu_text_size[0][0] - 10, frame.shape[0] - menu_text_size[0][1] - 10)
        cv2.putText(frame, menu_text, menu_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Display the frame
    cv2.imshow('Video Stream', frame)

    # Check for key press events
    key = cv2.waitKey(1) & 0xFF

    # Toggle night vision mode on 'N' key press
    if key == ord('N') or key == ord('n'):
        night_vision_mode = not night_vision_mode

    # Toggle green square on 'H' key press
    if key == ord('H') or key == ord('h'):
        show_green_square = not show_green_square

    # Reset face detection on 'R' key press
    if key == ord('R') or key == ord('r'):
        face_tracker = None

    # Toggle pixelation on 'F' key press
    if key == ord('F') or key == ord('f'):
        pixelation_enabled = not pixelation_enabled

    # Toggle pixelation method on 'T' key press
    if key == ord('T') or key == ord('t'):
        use_new_pixelation = not use_new_pixelation

    # Toggle additional info display on 'I' key press
    if key == ord('I') or key == ord('i'):
        show_additional_info = not show_additional_info

    # Break the loop if the 'Q' key is pressed
    if key == ord('Q') or key == ord('q'):
        break

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()
