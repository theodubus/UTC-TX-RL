# UTC-TX-RL

[![CI](https://github.com/theodubus/UTC-TX-RL/actions/workflows/ci.yml/badge.svg)](https://github.com/theodubus/UTC-TX-RL/actions/workflows/ci.yml)

Apprentissage par renforcement — notre **TX** (projet de recherche en autonomie) à l'UTC, menée en binôme avec [sacha-sz](https://github.com/sacha-sz) : un agent **Q-learning** apprend seul à traverser des labyrinthes de difficulté croissante, récupérer une clé puis ouvrir un coffre.

<p align="center">
  <img src="images/laby.gif" alt="Labyrinthe simple" width="350"/>
  <img src="images/lave.gif" alt="Labyrinthe avec lave" width="350"/>
</p>

## Le principe

Chaque niveau est une grille chargée depuis [`levels/`](levels) (8 niveaux fournis, du couloir simple aux labyrinthes piégés) :

| | Dans le labyrinthe |
|---|---|
| `#` / `.` | mur infranchissable / case libre |
| `P` | position de départ de l'agent |
| `K` puis `C` | la clé à récupérer, puis le coffre à ouvrir |
| `L` | lave, fortement pénalisée |
| `B` | mur cassable : un raccourci qui coûte |

L'agent ne connaît rien du niveau : il explore (politique ε-greedy), encaisse les récompenses — +100 pour la clé et le coffre, −100 pour la lave, −10 contre un mur, −5 pour en casser un, −1 par pas — et remplit sa **Q-table** jusqu'à converger vers une politique qui va chercher la clé par le chemin le plus rentable avant d'ouvrir le coffre.

L'interface (tkinter) permet de choisir le niveau, de lancer l'entraînement et de rejouer la politique apprise sous forme de GIF :

<p align="center">
  <img src="images/agent.gif" alt="Politique apprise rejouée" width="350"/>
</p>

## Lancer le projet

```bash
make install        # dépendances (tkinter requis : sudo apt install python3-tk)
make run            # interface graphique
make test           # tests unitaires de l'environnement
```

La méthodologie, les choix de modélisation et les résultats détaillés sont dans le [rapport de TX](docs/Rapport_TX.pdf).

## Licence

[MIT](LICENSE) — [theodubus](https://github.com/theodubus/) & [sacha-sz](https://github.com/sacha-sz)
