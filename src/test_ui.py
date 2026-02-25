# test_ui.py
import pygame
import sys

pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Test de stabilité")
font = pygame.font.SysFont("monospace", 20)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    screen.fill((0, 0, 255))
    label = font.render("Si vous voyez ce bleu, Pygame fonctionne !", True, (255, 255, 255))
    screen.blit(label, (20, 140))
    pygame.display.update()