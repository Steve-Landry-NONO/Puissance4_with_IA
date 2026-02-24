Puissance 4 Intelligent : Deep Q-Learning & Minimax

Ce projet est réalisé dans le cadre de ma formation IATIC-5 à l'ISTY (UVSQ/Paris-Saclay). Il propose une implémentation complète du jeu Puissance 4, intégrant une interface graphique moderne et une Intelligence Artificielle basée sur l'apprentissage par renforcement profond (DQN).
🌟 Fonctionnalités

    Moteur de jeu robuste : Gestion complète des règles, détection de victoire et d'égalité.

    Interface Graphique (GUI) : Développée avec Pygame, incluant une sidebar informative et une fluidité de jeu.

    Intelligence Artificielle multi-niveaux :

        Agent Aléatoire : Pour les tests de base.

        Agent Minimax : Algorithme de recherche avec élagage Alpha-Bêta.

        Agent DQN : Réseau de neurones convolutif (CNN) entraîné avec TensorFlow.

    Mode "Headless" : Entraînement massif de l'IA sans interface graphique pour une performance accrue.

🏗️ Architecture du Réseau DQN

L'agent DQN utilise un réseau de neurones conçu pour interpréter la grille comme une image à deux canaux (6×7×2), permettant de distinguer ses propres pions de ceux de l'adversaire.
🚀 Installation

    Cloner le dépôt :
    Bash

    git clone https://github.com/votre-username/puissance4-ia.git
    cd puissance4-ia

    Installer les dépendances :
    Bash

    pip install -r requirements.txt

🎮 Utilisation

Le script principal main.py permet de lancer des parties en configurant les deux joueurs via la ligne de commande.
Jouer contre l'IA (Minimax)
Bash

python src/main.py --p1 human --p2 minimax --depth 4

Observer un duel DQN vs Minimax
Bash

python src/main.py --p1 dqn --p2 minimax --headless

Entraîner l'IA
Bash

python src/main.py --p1 dqn --p2 random --train --episodes 1000 --headless

📂 Structure du projet

    src/core/ : Logique métier (Board, Rules).

    src/ui/ : Interface graphique et gestion des événements Pygame.

    src/agents/ : Implémentations des différents types d'IA.

    assets/ : Ressources graphiques (logos ISTY/UVSQ, images).

    models/ : Sauvegardes des modèles TensorFlow entraînés (.h5).

📚 Références & Remerciements

    M. Franck Talbart (ISTY) : Pour la coordination du projet.

    M. Hakim Horairy (HETIC) : Pour les supports de cours fondamentaux sur les CNN et l'apprentissage supervisé.

    Cursus IATIC 3-5 : Pour l'ensemble des compétences en génie logiciel et IA acquises à l'ISTY.

📄 Licence

Ce projet est réalisé dans un but pédagogique pour l'obtention du diplôme d'ingénieur ISTY.
