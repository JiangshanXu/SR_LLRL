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
ORANGE = (255, 165, 0)  # Lava color

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Four Rooms Environment")

# Function to draw the grid
def draw_grid():
    for x in range(0, SCREEN_WIDTH, GRID_SIZE):
        for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
            rect = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)

# Function to draw the walls (obstacles) to create the four rooms
def draw_walls():
    # List of wall positions for the four rooms (adjusted for 11x11 grid)
    walls = [
        # Horizontal walls
        (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5,5),(6, 5), (7, 5), (8, 5), (9, 5), (10, 5),
        # Vertical walls
        (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10)
    ]

    # Add doors (remove specific wall tiles to create doors)
    doors = [(5, 3), (3, 5), (7, 5), (5, 7)]
    walls = [wall for wall in walls if wall not in doors]

    # Draw the walls
    for wall in walls:
        wall_rect = pygame.Rect(wall[0] * GRID_SIZE, wall[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(screen, BLACK, wall_rect)

# Function to draw the lava path
def draw_lava():
    # Predefined lava path for this environment
    lava = [
        (3, 10), (5, 9), (7, 7), (8, 5), (9, 3)  # Diagonal lava path
    ]
    for lava_tile in lava:
        lava_rect = pygame.Rect(lava_tile[0] * GRID_SIZE, lava_tile[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(screen, ORANGE, lava_rect)
    return lava

# Function to draw goals
def draw_goals():
    # Predefined goals for this environment
    goals = [
        (10, 0)  # Top-right corner
    ]
    for goal in goals:
        goal_rect = pygame.Rect(goal[0] * GRID_SIZE, goal[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
        # pygame.draw.rect(screen, GREEN, goal_rect)
        # red:
        pygame.draw.rect(screen, RED, goal_rect)
    return goals

# Function to draw the start point
def draw_start_point():
    start = (0, 10)  # Bottom-left corner
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
    # draw_walls()
    draw_lava()
    draw_goals()
    draw_start_point()
    pygame.display.flip()
    # save if s is pressed:
    keys = pygame.key.get_pressed()
    if keys[pygame.K_s]:
        import time
        time_format = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        pygame.image.save(screen, f"lava_{time_format}.png")
        print("saved")

pygame.quit()
sys.exit()
