import numpy as np
import cv2

class PIDController:
    def __init__(self, target_pos):
        self.target_pos = target_pos
        self.Kp = 3500.0
        self.Ki = 82.0
        self.Kd = 45000.0
        self.bias = 0.0
        self.errors=[0.0]
        return

    def reset(self):
        return
    def detect_ball(self, frame):
        bgr_color = 85, 65, 155
        hsv_threshold = [60, 80, 100]
        radius_threshold = 8

        hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
        HSV_LOWER = np.array([hsv_color[0] - hsv_threshold[0], hsv_color[1] - hsv_threshold[1], hsv_color[2] - hsv_threshold[2]])
        HSV_UPPER = np.array([hsv_color[0] + hsv_threshold[0], hsv_color[1] + hsv_threshold[1], hsv_color[2] + hsv_threshold[2]])
        x, y, radius = -1, -1, -1
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv_frame, HSV_LOWER, HSV_UPPER)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=10)

        im2, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = (-1, -1)

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            m = cv2.moments(mask)
            center = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))

        if radius > radius_threshold:
            cv2.circle(frame, (int(x), int(y)), int(radius), (255, 0, 0), 2)
            cv2.circle(frame, center, 5, (0, 255, 0), -1)

        

        return (-center[1] + 482) / 482.0

    def get_fan_rpm(self, image_frame=None):
        output = 0.0
        vertical_ball_position = self.detect_ball(image_frame)
        interations = len(self.errors)
        self.errors.append(self.target_pos - vertical_ball_position)
        output = self.Kp*(self.errors[interations]) + self.Ki*sum(self.errors)
        if self.errors[interations-1] != 0:
            output += self.Kd*(self.errors[interations]-self.errors[interations-1]) 
        

        return output, vertical_ball_position
