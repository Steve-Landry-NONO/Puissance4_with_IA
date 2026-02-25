# humain

class HumanAgent:
    """
    Agent représentant un joueur humain.

    Dans cette architecture, la logique de décision n'est pas traitée
    par la méthode act() de l'agent, mais interceptée directement
    par la boucle d'événements de Pygame dans PygameApp.
    """

    def __init__(self):
        pass

    def act(self, *args, **kwargs):
        """
        Cette méthode est requise par le polymorphisme du projet,
        mais elle n'est pas utilisée pour l'humain en mode graphique.
        """
        pass


# Alias pour assurer la compatibilité avec les différents imports du main.py
HumainAgent = HumanAgent