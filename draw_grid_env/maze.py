import pygame
import sys

# Initialize Pygame
pygame.init()

# Define constants
GRID_SIZE = 40  # Size of grid cells
GRID_WIDTH = 11  # Number of columns
GRID_HEIGHT = 11  # Number of rows
SCREEN_WIDTH = GRID_SIZE * GRID_WIDTH
SCREEN_HEIGHT = GRID_SIZE * GRID_HEIGHT
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (165, 42, 42)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Maze Environment")


# Function to draw the grid
def draw_grid():
    for x in range(0, SCREEN_WIDTH, GRID_SIZE):
        for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
            rect = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)


# Function to draw the walls (obstacles)
def draw_walls():
    # List of wall positions based on the image
    walls = [
        # (2, 2), (3, 2), (4, 2),
        (1, 8),
        (2, 8), (2, 9),
        (4, 2), (4, 4),
        (5, 3),
        (8, 7),
        (9, 0), (9, 1), (9, 7),
        (10, 8)
        # (8, 4), (9, 4),
        # (1, 6), (2, 6),
        # (5, 6),
        # (7, 7),
        # (3, 8)
    ]

    # Draw the walls
    for wall in walls:
        wall_rect = pygame.Rect(wall[0] * GRID_SIZE, wall[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(screen, BLACK, wall_rect)


# Function to draw the goal
def draw_goal():
    goal = (10, 0)  # Based on the green square in the image
    goal_rect = pygame.Rect(goal[0] * GRID_SIZE, goal[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
    # pygame.draw.rect(screen, GREEN, goal_rect)
    # red:
    pygame.draw.rect(screen, RED, goal_rect)

# Function to draw the start point
def draw_start_point():
    start = (0, 10)  # Based on the 'S' position in the image
    start_rect = pygame.Rect(start[0] * GRID_SIZE, start[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
    # pygame.draw.rect(screen, RED, start_rect)
    # green:
    pygame.draw.rect(screen, GREEN, start_rect)


# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)
    draw_grid()
    draw_walls()
    draw_goal()
    draw_start_point()
    pygame.display.flip()

    # Save if 's' is pressed
    keys = pygame.key.get_pressed()
    if keys[pygame.K_s]:
        import time

        time_format = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        pygame.image.save(screen, f"maze_{time_format}.png")
        print("Image saved as maze_{time_format}.png")

pygame.quit()
sys.exit()
