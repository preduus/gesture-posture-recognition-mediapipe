import mediapipe as mp

class MediapipeProcessor:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.mp_holistic = mp.solutions.holistic
        self.mp_face_detection = mp.solutions.face_detection

    def recognize_gesture(self, landmarks):
        thumb_up = (
            landmarks[self.mp_holistic.HandLandmark.THUMB_TIP].y < landmarks[self.mp_holistic.HandLandmark.WRIST].y and
            landmarks[self.mp_holistic.HandLandmark.INDEX_FINGER_TIP].y < landmarks[self.mp_holistic.HandLandmark.THUMB_TIP].y
        )
        peace_sign = (
            landmarks[self.mp_holistic.HandLandmark.INDEX_FINGER_TIP].y > landmarks[self.mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y and
            landmarks[self.mp_holistic.HandLandmark.THUMB_TIP].y > landmarks[self.mp_holistic.HandLandmark.INDEX_FINGER_TIP].y
        )
        pointing = landmarks[self.mp_holistic.HandLandmark.INDEX_FINGER_TIP].visibility > 0.9

        if thumb_up:
            return "Thumbs Up"
        elif peace_sign:
            return "Peace Sign"
        elif pointing:
            return "Pointing"
        elif landmarks[self.mp_holistic.HandLandmark.WRIST].visibility < 0.9:
            return "Hand Waving"
        elif landmarks[self.mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[self.mp_holistic.HandLandmark.WRIST].y:
            return "Middle Finger Raised"
        else:
            visible_fingers = sum(1 for lm in landmarks if lm.visibility > 0.9)
            return f"{visible_fingers} Finger(s)"

    def recognize_posture(self, pose_landmarks):
        hips = [pose_landmarks.landmark[i] for i in [self.mp_pose.PoseLandmark.LEFT_HIP.value, self.mp_pose.PoseLandmark.RIGHT_HIP.value]]
        shoulders = [pose_landmarks.landmark[i] for i in [self.mp_pose.PoseLandmark.LEFT_SHOULDER.value, self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]]

        if all(hip.y > 0.7 for hip in hips):
            return "Sitting"
        elif all(shoulder.y < 0.5 for shoulder in shoulders):
            return "Jumping"
        elif all(hip.x - shoulders[i].x < 0.1 for i, hip in enumerate(hips)):
            return "Straight Posture"
        else:
            return "Standing"

    def process_frame(self, image, pose, holistic, face_detection):
        results = {
            "pose": pose.process(image),
            "holistic": holistic.process(image),
            "face": face_detection.process(image)
        }
        return results

    def draw_landmarks(self, image, results):
        if results["pose"].pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results["pose"].pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        if results["holistic"].left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results["holistic"].left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
            )
        if results["holistic"].right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results["holistic"].right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
            )

    def annotate_frame(self, image, results):
        if results["holistic"].right_hand_landmarks:
            gesture = self.recognize_gesture(results["holistic"].right_hand_landmarks.landmark)
            cv2.putText(image, gesture, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if results["pose"].pose_landmarks:
            posture = self.recognize_posture(results["pose"].pose_landmarks)
            cv2.putText(image, posture, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if results["face"].detections:
            num_faces = len(results["face"].detections)
            cv2.putText(image, f"Detected {num_faces} Face(s)", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return image
