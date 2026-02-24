import pygame
import os

# --- Constantes visuelles en français ---
TAILLE_CASE = 90
LARGEUR_BARRE_LATERALE = 220  # Un peu plus large pour l'élégance
HAUTEUR_ENTETE = 110
HAUTEUR_PIED_PAGE = 50

BLEU_PLATEAU = (0, 90, 190)
NOIR_FOND = (10, 10, 10)
GRIS_SIDEBAR = (25, 25, 25)
ROUGE = (230, 50, 50)
JAUNE = (245, 210, 60)
BLANC = (245, 245, 245)
GRIS_TEXTE = (140, 140, 140)


class GameRenderer:
    def __init__(self, plateau):
        pygame.init()
        self.plateau = plateau
        self.largeur = (plateau.COLS * TAILLE_CASE) + LARGEUR_BARRE_LATERALE
        self.hauteur = (plateau.ROWS * TAILLE_CASE) + HAUTEUR_ENTETE + HAUTEUR_PIED_PAGE
        self.screen = pygame.display.set_mode((self.largeur, self.hauteur))
        pygame.display.set_caption("Puissance 4 - Intelligence Artificielle")

        # Chargement du Logo
        self.logo = None
        if os.path.exists("logo.jpg"):
            self.logo = pygame.image.load("logo.jpg")
            self.logo = pygame.transform.smoothscale(self.logo, (150, 100))

        # Polices
        self.police_titre = pygame.font.SysFont("georgia", 38, bold=True)
        self.police_infos = pygame.font.SysFont("segoe ui", 20, bold=True)
        self.police_aide = pygame.font.SysFont("segoe ui", 15)

    def dessiner_interface(self, nom_p1, nom_p2, scores):
        # 1. Barre Latérale (Sidebar)
        pygame.draw.rect(self.screen, GRIS_SIDEBAR, (0, 0, LARGEUR_BARRE_LATERALE, self.hauteur))

        # Logo ou Cercle Design
        if self.logo:
            self.screen.blit(self.logo, (LARGEUR_BARRE_LATERALE // 2 - 50, 20))
        else:
            pygame.draw.circle(self.screen, BLANC, (LARGEUR_BARRE_LATERALE // 2, 60), 40, 2)

        # Joueur 1
        self._dessiner_avatar(160, ROUGE, "JOUEUR 1", nom_p1)
        # Joueur 2
        self._dessiner_avatar(360, JAUNE, "JOUEUR 2", nom_p2)

        # 2. Entête (Header)
        pygame.draw.rect(self.screen, NOIR_FOND,
                         (LARGEUR_BARRE_LATERALE, 0, self.largeur - LARGEUR_BARRE_LATERALE, HAUTEUR_ENTETE))

        score_txt = f"Jaune {scores[-1]}  |  {scores[1]} Rouge"
        label_score = self.police_infos.render(score_txt, True, BLANC)
        self.screen.blit(label_score, label_score.get_rect(
            center=(LARGEUR_BARRE_LATERALE + (self.largeur - LARGEUR_BARRE_LATERALE) // 2, HAUTEUR_ENTETE - 30)))

        # 3. Pied de page
        pygame.draw.rect(self.screen, NOIR_FOND, (LARGEUR_BARRE_LATERALE, self.hauteur - HAUTEUR_PIED_PAGE,
                                                  self.largeur - LARGEUR_BARRE_LATERALE, HAUTEUR_PIED_PAGE))
        aide = self.police_aide.render("[R] Recommencer  •  [Q] Quitter", True, GRIS_TEXTE)
        self.screen.blit(aide, aide.get_rect(
            center=(LARGEUR_BARRE_LATERALE + (self.largeur - LARGEUR_BARRE_LATERALE) // 2,
                    self.hauteur - HAUTEUR_PIED_PAGE // 2)))

    def _dessiner_avatar(self, y_pos, couleur, label, sous_label):
        pygame.draw.circle(self.screen, couleur, (LARGEUR_BARRE_LATERALE // 2, y_pos), 35)
        txt = self.police_infos.render(label, True, BLANC)
        stxt = self.police_aide.render(f"({sous_label})", True, GRIS_TEXTE)
        self.screen.blit(txt, txt.get_rect(center=(LARGEUR_BARRE_LATERALE // 2, y_pos + 55)))
        self.screen.blit(stxt, stxt.get_rect(center=(LARGEUR_BARRE_LATERALE // 2, y_pos + 75)))

    def dessiner_plateau(self):
        for c in range(self.plateau.COLS):
            for r in range(self.plateau.ROWS):
                x = LARGEUR_BARRE_LATERALE + c * TAILLE_CASE
                y = HAUTEUR_ENTETE + r * TAILLE_CASE
                pygame.draw.rect(self.screen, BLEU_PLATEAU, (x, y, TAILLE_CASE, TAILLE_CASE))
                val = self.plateau.grid[r][c]
                color = NOIR_FOND
                if val == 1:
                    color = ROUGE
                elif val == -1:
                    color = JAUNE
                pygame.draw.circle(self.screen, color, (x + TAILLE_CASE // 2, y + TAILLE_CASE // 2),
                                   int(TAILLE_CASE / 2 - 8))

    def afficher_victoire(self, msg, couleur):
        label = self.police_titre.render(msg, True, couleur)
        rect = label.get_rect(
            center=(LARGEUR_BARRE_LATERALE + (self.largeur - LARGEUR_BARRE_LATERALE) // 2, HAUTEUR_ENTETE // 2 - 10))
        self.screen.blit(label, rect)