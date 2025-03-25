from ultralytics import YOLO
import cv2
import os

def main():
    # Load the trained YOLO model
    model = YOLO("yolo11n.pt")

    # Open a connection to the camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    frame_skip = 5  # Number of frames to skip between detections
    frame_count = 0
    output_dir = "runs/detect/exp1"  # Directory to save detection results
    os.makedirs(output_dir, exist_ok=True)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Process every nth frame
        if frame_count % frame_skip == 0:
            # Run detection on the frame
            results = model(frame)

            # Check if there are any detections
            has_detections = False

            # Draw bounding boxes and labels on the frame
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
                confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                labels = result.boxes.cls.cpu().numpy()  # Class labels

                if len(boxes) > 0:
                    has_detections = True

                for box, confidence, label in zip(boxes, confidences, labels):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f"{int(label)}: {confidence:.2f}"
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Save the annotated frame only if there are detections
            if has_detections:
                output_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(output_path, frame)

        # Display the resulting frame
        cv2.imshow('Real-Time Vehicle Detection', frame)

        # Break the loop if the window is closed or 'q' key is pressed
        if cv2.getWindowProperty('Real-Time Vehicle Detection', cv2.WND_PROP_VISIBLE) < 1 or cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
