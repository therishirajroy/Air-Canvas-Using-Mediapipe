import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.9)

# Canvas dimensions
canvas_width, canvas_height = 640, 480
canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255  # White canvas

# Drawing parameters
draw_color = (0, 0, 0)  # Default Black
brush_thickness = 5
colors = [(0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Black, Red, Green, Blue
color_names = ['Black', 'Red', 'Green', 'Blue']
color_boxes = [(50, 0, 150, 50), (200, 0, 300, 50), (350, 0, 450, 50), (500, 0, 600, 50)]

# Brush size selector
brush_sizes = [2, 4, 6, 8, 10]
size_boxes = [(550, 50 + i * 50, 600, 100 + i * 50) for i in range(len(brush_sizes))]

# Flags
drawing = False  # Tracks if the user is drawing
last_position = None  # Tracks the last position of the fingertip

# Video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip horizontally for natural interaction
    h, w, _ = frame.shape

    # Resize the canvas to match the frame dimensions (optional)
    canvas = cv2.resize(canvas, (w, h))

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract the index fingertip and thumb tip positions
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            x, y = int(index_tip.x * w), int(index_tip.y * h)
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Change color if finger is in a color box
            for i, (x1, y1, x2, y2) in enumerate(color_boxes):
                if x1 < x < x2 and y1 < y < y2:
                    draw_color = colors[i]

            # Change brush size if finger is in a size box
            for i, (x1, y1, x2, y2) in enumerate(size_boxes):
                if x1 < x < x2 and y1 < y < y2:
                    brush_thickness = brush_sizes[i]

            # Calculate the distance between the index fingertip and thumb tip
            distance = np.sqrt((x - thumb_x)**2 + (y - thumb_y)**2)

            if distance < 30:  # Threshold to detect if thumb and index finger are close
                drawing = False
            elif distance >= 30:
                drawing = True    
            
            if drawing:  # If drawing is enabled
                if last_position is not None:
                    # Draw a line from the last position to the current fingertip position
                    cv2.line(canvas, last_position, (x, y), draw_color, brush_thickness)
                last_position = (x, y)
            else:
                last_position = None

            # Display landmarks for debugging purposes (optional)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Draw color selection boxes
    for i, (x1, y1, x2, y2) in enumerate(color_boxes):
        cv2.rectangle(frame, (x1, y1), (x2, y2), colors[i], -1)
        cv2.putText(frame, color_names[i], (x1 + 5, y1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw brush size selector
    for i, (x1, y1, x2, y2) in enumerate(size_boxes):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv2.putText(frame, str(brush_sizes[i]), (x1 + 15, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Display instructions
    instructions = "Move your finger to a color box to change color. Move to size box to change thickness."
    cv2.putText(frame, instructions, (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the canvas and camera feed
    cv2.imshow("Canvas", canvas)
    cv2.imshow("Camera Feed", frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):  # Clear the canvas
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    elif key == ord('q'):  # Quit
        break

cap.release()
cv2.destroyAllWindows()
