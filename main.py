import cv2
from mediapipe_processor import MediapipeProcessor


if __name__ == "__main__":
    processor = MediapipeProcessor()

    cap = cv2.VideoCapture(0)
    with processor.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         processor.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, \
         processor.mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:

        while cap.isOpened():
            _, frame = cap.read()
            if not _:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False

            results = processor.process_frame(frame, pose, holistic, face_detection)

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            processor.draw_landmarks(frame, results)
            annotated_frame = processor.annotate_frame(frame, results)

            cv2.imshow('Mediapipe Recognition', annotated_frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
