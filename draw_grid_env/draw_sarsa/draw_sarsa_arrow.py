import pygame
import sys
import math
# Initialize Pygame
pygame.init()

# Define constants
GRID_SIZE = 40  # Size of grid cells
GRID_WIDTH = 10  # Number of columns
GRID_HEIGHT = 10  # Number of rows
SCREEN_WIDTH = GRID_SIZE * GRID_WIDTH
SCREEN_HEIGHT = GRID_SIZE * GRID_HEIGHT
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
ARROW_COLOR = (0, 0, 255)

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Sarsa Lambda Visualization")

# Define the path taken by the agent
path_taken = [(0, 9), (1, 9), (2, 9), (3, 9), (4, 9), (4, 8), (4, 7), (5, 7), (6, 7), (7, 7), (8, 7), (9, 7), (9, 6)]
goal_position = (9, 6)

# Define the action values (arrows)
arrows = {
    (2, 9): (1, 0),  # Example: move up
    (3, 9): (1, 0),
    (4, 9): (0, -1),
    (4, 8): (0, -1),  # Example: move right
    (4, 7): (1, 0),
    (5, 7): (1, 0),
    (6, 7): (1, 0),
    (7, 7): (1, 0),
    (8, 7): (1, 0),  # Example: move up
    (9, 7): (0, -1),
}


# Function to draw the grid
def draw_grid():
    for x in range(0, SCREEN_WIDTH, GRID_SIZE):
        for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
            rect = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)


# Function to draw the path
def draw_path():
    for i in range(len(path_taken) - 1):
        start_pos = (path_taken[i][0] * GRID_SIZE + GRID_SIZE // 2, path_taken[i][1] * GRID_SIZE + GRID_SIZE // 2)
        end_pos = (path_taken[i + 1][0] * GRID_SIZE + GRID_SIZE // 2, path_taken[i + 1][1] * GRID_SIZE + GRID_SIZE // 2)
        pygame.draw.line(screen, BLACK, start_pos, end_pos, 5)


# Function to draw the goal
def draw_goal():
    goal_rect = pygame.Rect(goal_position[0] * GRID_SIZE, goal_position[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
    pygame.draw.rect(screen, RED, goal_rect)


# Function to draw arrows indicating action values
def draw_arrows():
    max_arrow_size = 20  # Maximum arrow size
    min_arrow_size = 5  # Minimum arrow size

    target_x, target_y = goal_position

    for position, direction in arrows.items():
        x = position[0] * GRID_SIZE + GRID_SIZE // 2
        y = position[1] * GRID_SIZE + GRID_SIZE // 2
        dx = direction[0] * GRID_SIZE // 2
        dy = direction[1] * GRID_SIZE // 2
        end_pos = (x + dx, y + dy)

        # Calculate the distance from the current position to the target
        distance_to_target = math.sqrt((target_x - position[0]) ** 2 + (target_y - position[1]) ** 2)

        # Normalize the distance to determine the arrow size
        max_distance = math.sqrt((target_x - 0) ** 2 + (target_y - 0) ** 2)  # Max possible distance on the grid
        arrow_size = max_arrow_size - ((distance_to_target / max_distance) * (max_arrow_size - min_arrow_size))

        # Draw the arrow line
        pygame.draw.line(screen, ARROW_COLOR, (x, y), end_pos, 3)

        # Draw the arrowhead
        if direction[0] != 0:  # horizontal arrow
            pygame.draw.polygon(screen, ARROW_COLOR, [(end_pos[0], end_pos[1]),
                                                      (end_pos[0] - direction[0] * arrow_size, end_pos[1] - arrow_size),
                                                      (
                                                      end_pos[0] - direction[0] * arrow_size, end_pos[1] + arrow_size)])
        elif direction[1] != 0:  # vertical arrow
            pygame.draw.polygon(screen, ARROW_COLOR, [(end_pos[0], end_pos[1]),
                                                      (end_pos[0] - arrow_size, end_pos[1] - direction[1] * arrow_size),
                                                      (
                                                      end_pos[0] + arrow_size, end_pos[1] - direction[1] * arrow_size)])
# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)
    draw_grid()
    # draw_path()
    draw_goal()
    draw_arrows()
    pygame.display.flip()
    # if press s then save:
    keys = pygame.key.get_pressed()
    if keys[pygame.K_s]:
        import time
        time_format = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        pygame.image.save(screen, f"grid_env_{time_format}.png")
        print("saved")

pygame.quit()
sys.exit()
