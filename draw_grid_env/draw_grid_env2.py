import pygame
import sys
import random
# Initialize Pygame
pygame.init()

# Define constants
GRID_SIZE = 20  # Size of grid cells
GRID_WIDTH = 20  # Number of columns
GRID_HEIGHT = 20  # Number of rows
SCREEN_WIDTH = GRID_SIZE * GRID_WIDTH
SCREEN_HEIGHT = GRID_SIZE * GRID_HEIGHT
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (165, 42, 42)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Grid Environment")


start_points = [
    (0,19),(12,19),(15,19),(4,18),(17,18)
]
# Create a list of obstacle positions

# at row 16, the first 14 column:
# obstacle2= [(i, 16) for i in range(14)]
# append:
# obstacles.extend(obstacle2)
# Create a list of goal positions
goals = [
    # (18, 2), (17, 3), (16, 4)
    # more sparse:
    (18,2) , (2,3), (14,1)
]
# random generate three goals between 18,1 to 2,5:
# random_goals = [(random.randint(2, 18), random.randint(1, 5)) for _ in range(100)]


# Function to draw the grid
def draw_grid():
    for x in range(0, SCREEN_WIDTH, GRID_SIZE):
        for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
            rect = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)

# Function to draw obstacles
def draw_obstacles(given_goals):
    # obstacles = [
    #     (5, 5), (5, 6), (5, 7), (6, 7), (7, 7),
    #     (10, 10), (11, 10), (12, 10), (12, 11), (12, 12),
    #     (3, 4), (4, 4), (3, 3), (4, 3), (15, 15), (16, 15), (17, 15)
    #
    # ]
    # # random generate more obstacles in (x,y) where y should be smaller than 17.
    # random_obs = [(random.randint(0, 19), random.randint(0, 16)) for _ in range(12)]
    # # can not comflict with goals:
    # for obs in random_obs:
    #     if obs in given_goals:
    #         random_obs.remove(obs)
    # for obs in obstacles:
    #     if obs in given_goals:
    #         obstacles.remove(obs)
    # obstacles.extend(random_obs)
    predefined_obstacles = {
        (5, 5), (5, 6), (5, 7), (6, 7), (7, 7),
        (10, 10), (11, 10), (12, 10), (12, 11), (12, 12),
        (3, 4), (4, 4), (3, 3), (4, 3), (15, 15), (16, 15), (17, 15)
    }

    # Convert given_goals to a set for O(1) lookups
    given_goals_set = set(given_goals)

    # Remove any obstacles that overlap with goals
    obstacles = predefined_obstacles - given_goals_set

    # Generate random obstacles
    while len(obstacles) < 100:
        new_obs = (random.randint(0, 19), random.randint(0, 16))
        if new_obs not in obstacles and new_obs not in given_goals_set:
            obstacles.add(new_obs)
    for obs in obstacles:
        obs_rect = pygame.Rect(obs[0] * GRID_SIZE, obs[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(screen, BLACK, obs_rect)

# Function to draw goals
def draw_goals():
    random_goals = [(random.randint(2, 18), random.randint(1, 5)) for _ in range(3)]

    for goal in random_goals:
        goal_rect = pygame.Rect(goal[0] * GRID_SIZE, goal[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(screen, RED, goal_rect)
    return random_goals

# Function to draw 5 random start points in the last two rows
def draw_start_points():
    '''
    draw using start point
    '''
    for start in start_points:
        start_rect = pygame.Rect(start[0] * GRID_SIZE, start[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(screen, GREEN, start_rect)




# Main loop
running = True
should_render = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # wait for space to continue rendering:
    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE]:
        should_render = not should_render
    # if r then save the current screen:
    if keys[pygame.K_r]:
        # pygame.image.save(screen, "grid_env.png")
        # save with time:
        import time
        time_format = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        # pygame.image.save(screen, f"grid_env_{time.time()}.png")
        pygame.image.save(screen, f"grid_env_{time_format}.png")
        print("saved")

    if not should_render:
        # print("paused")
        pygame.display.flip()
        continue
    screen.fill(WHITE)
    draw_grid()
    random_goals = draw_goals()
    draw_obstacles(random_goals)

    draw_start_points()
    pygame.display.flip()





pygame.quit()
sys.exit()
