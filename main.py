import cv2
import mediapipe as mp
import pygame
import numpy as np

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

mp_drawing = mp.solutions.drawing_utils

pygame.init()
screen = pygame.display.set_mode((1600, 800))
pygame.display.set_caption("Hand Recognition and Drawing")

clock = pygame.time.Clock()

WHITE = (255, 255, 255)

BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
colors = [BLACK, RED, GREEN, BLUE]
current_color = colors[0]

drawing = []
drawing_screen = pygame.Surface((800, 800))
drawing_screen.fill(WHITE)
paused = False

def draw_color_palette(screen, colors, current_color):


    for i, color in enumerate(colors):
        pygame.draw.rect(screen, color, (1550, 50 + i * 50, 50, 50))


        if color == current_color:
            pygame.draw.rect(screen, WHITE, (1550, 50 + i * 50, 50, 50), 3)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    exit()

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    screen.fill(WHITE)


    draw_color_palette(screen, colors, current_color)

    if results.multi_hand_landmarks:


        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            index_finger_tip = landmarks[8]
            thumb_tip = landmarks[4]
            wrist = landmarks[mp_hands.HandLandmark.WRIST]



            h, w, _ = frame.shape
            index_finger_tip_pos = (int(index_finger_tip.x * 800), int(index_finger_tip.y * 800))
            thumb_tip_pos = (int(thumb_tip.x * 800), int(thumb_tip.y * 800))

            if np.linalg.norm(np.array(index_finger_tip_pos) - np.array(thumb_tip_pos)) < 30:
                current_color = colors[(colors.index(current_color) + 1) % len(colors)]

            if not paused and index_finger_tip.y < wrist.y - 0.1:
                drawing.append((index_finger_tip_pos, current_color))




    for i in range(1, len(drawing)):
        pygame.draw.line(drawing_screen, drawing[i - 1][1], drawing[i - 1][0], drawing[i][0], 5)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.rot90(frame)
    frame_surface = pygame.surfarray.make_surface(frame)
    frame_surface = pygame.transform.scale(frame_surface, (800, 800))



    screen.blit(frame_surface, (0, 0))
    screen.blit(drawing_screen, (800, 0))




    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            cv2.destroyAllWindows()
            exit()
            
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                paused = not paused
            elif event.key == pygame.K_c:
                drawing = []
                drawing_screen.fill(WHITE)

    pygame.display.update()
    clock.tick(60)

cap.release()
cv2.destroyAllWindows()
pygame.quit()
