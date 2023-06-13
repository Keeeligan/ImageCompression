import pygame



def run_pygame():
    pygame.init()

    # Create canvas
    canvas = pygame.display.set_mode((500, 500))

    # Title of canvas
    pygame.display.set_caption("My Board")
    exit = False

    # Start running
    while not exit:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit = True
        pygame.display.update()



if __name__ == "__main__":
    run_pygame()