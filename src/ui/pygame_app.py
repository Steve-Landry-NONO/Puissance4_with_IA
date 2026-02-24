import sys
import pygame
from src.ui.render import GameRenderer, TAILLE_CASE, LARGEUR_BARRE_LATERALE, ROUGE, JAUNE, BLANC, NOIR_FOND

class PygameApp:
    def __init__(self, board, p1, p2, delay_ms=300):
        self.plateau = board
        self.p1 = p1
        self.p2 = p2
        self.delay_ms = delay_ms
        self.scores = {1: 0, -1: 0}
        self.renderer = GameRenderer(self.plateau)
        self.reinitialiser()

    def reinitialiser(self):
        self.plateau.grid.fill(0)
        self.tour = 1
        self.partie_finie = False

    def executer(self):
        horloge = pygame.time.Clock()
        nom_j1 = self.p1.__class__.__name__.replace("Agent", "")
        nom_j2 = self.p2.__class__.__name__.replace("Agent", "")

        while True:
            agent_actuel = self.p1 if self.tour == 1 else self.p2
            est_humain = "human" in agent_actuel.__class__.__name__.lower()

            for evenement in pygame.event.get():
                if evenement.type == pygame.QUIT:
                    pygame.quit();
                    sys.exit()
                if evenement.type == pygame.KEYDOWN:
                    if evenement.key == pygame.K_r: self.reinitialiser()
                    if evenement.key == pygame.K_q: pygame.quit(); sys.exit()

                if not self.partie_finie and est_humain and evenement.type == pygame.MOUSEBUTTONDOWN:
                    pos_x, _ = evenement.pos
                    if pos_x > LARGEUR_BARRE_LATERALE:
                        colonne = (pos_x - LARGEUR_BARRE_LATERALE) // TAILLE_CASE
                        if self.plateau.is_valid_action(colonne):
                            self._jouer_coup(colonne)

            self.renderer.screen.fill(NOIR_FOND)
            self.renderer.dessiner_interface(nom_j1, nom_j2, self.scores)
            self.renderer.dessiner_plateau()

            if not self.partie_finie and not est_humain:
                pygame.time.wait(self.delay_ms)
                self._jouer_coup(agent_actuel.act(self.plateau))

            if self.partie_finie:
                vainqueur = self.plateau.check_winner()
                msg = "ROUGE GAGNE !" if vainqueur == 1 else "JAUNE GAGNE !" if vainqueur == -1 else "MATCH NUL"
                self.renderer.afficher_victoire(msg, ROUGE if vainqueur == 1 else JAUNE if vainqueur == -1 else BLANC)

            pygame.display.flip()
            horloge.tick(60)

    def _jouer_coup(self, colonne):
        self.plateau.drop_piece_inplace(colonne, self.tour)
        res = self.plateau.check_winner()
        if res is not None and res != 0:
            self.scores[res] += 1
            self.partie_finie = True
        elif self.plateau.is_draw():
            self.partie_finie = True
        else:
            self.tour *= -1