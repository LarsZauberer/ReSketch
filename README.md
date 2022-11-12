# ReSketch AI

## Abstrakt
ReSketch ist eine künstliche Intelligenz, die
versucht, Strichbilder auf eine physische Weise nachzuzeichnen. Strichbilder
sind in diesem Fall beispielsweise Ziffern oder Buchstaben. Um die Frage zu
beantworten, inwiefern das möglich ist, sind definierende Kriterien des
Nachzeichnens festgelegt. So soll die künstliche Intelligenz zum Beispiel nur
Bewegungen ausführen können, die auch mit einem Stift möglich wären. Die
künstliche Intelligenz erlernt das Nachzeichnen nach diesen Kriterien durch Deep
Q-Learning, einem Reinforcement Learning Modell. Das Modell basiert auf der
Arbeit hinter Doodle-SDQ \cite{zhou_learning_2018}, erfährt aber konzeptuelle
Variationen, wie die Integration einer Physiksimulation. Die künstliche
Intelligenz ist auf das Nachzeichnen von Ziffern trainiert. Ein Test dieser
trainierten künstlichen Intelligenz auf Buchstaben und andere Arten von
Strichbildern führt zu der Antwort auf die Frage, ob eine künstliche Intelligenz
das Nachzeichnen im Allgemeinen erlernen kann.

[Vollständige Dokumentation](https://github.com/LarsZauberer/Nachzeichner-KI/releases/download/1.0/Maturarbeit_IanWasser_RobinSteiner.pdf)

## Resultate
### Bilder
![AI Generated Images (Base-Base)](/Documentation/images/resultate/base-base.png)

### Tabellen
| ~                  | Übereinstimmung \% | Erkennbarkeit \% | Geschwindigkeit |
|--------------------|--------------------|------------------|-----------------|
| Grund-Basis        | 86.5               | 86.6             | 24.5            |
| Grund-MNIST        | 66.8               | 64.3             | 51.2            |
| Grund-Speed        | 85.7               | 82.3             | 23.3            |
| Grund-MNIST-Speed  | 61.4               | 55.1             | 56.8            |
| Physik-Basis       | 56.4               | 46.4             | 62.5            |
| Physik-MNIST       | 38.4               | 35.7             | 63.9            |
| Physik-Speed       | 63.0               | 58.2             | 61.2            |
| Physik-MNIST-Speed | 29.2               | 27.3             | 63.7            |

| ~                  | Übereinstimmung \% | Erkennbarkeit \% | Geschwindigkeit |
|--------------------|--------------------|------------------|-----------------|
| Grund-Basis        | 86.8               | 74.5             | 38.2            |
| Grund-MNIST        | 65.2               | 45.0             | 57.4            |
| Grund-Speed        | 88.1               | 73.5             | 36.1            |
| Grund-MNIST-Speed  | 62.2               | 40.0             | 60.9            |
| Physik-Basis       | 57.6               | 32.4             | 63.5            |
| Physik-MNIST       | 43.3               | 23.6             | 63.9            |
| Physik-Speed       | 56.3               | 35.0             | 63.6            |
| Physik-MNIST-Speed | 30.2               | 13.9             | 64.0            |

| ~                  | Übereinstimmung \% | Erkennbarkeit \% | Geschwindigkeit |
|--------------------|--------------------|------------------|-----------------|
| Grund-Basis        | 79.1               | 80.5             | 39.1            |
| Grund-MNIST        | 57.3               | 62.5             | 59.9            |
| Grund-Speed        | 80.0               | 80.3             | 35.0            |
| Grund-MNIST-Speed  | 54.9               | 58.9             | 62.5            |
| Physik-Basis       | 48.1               | 55.7             | 63.8            |
| Physik-MNIST       | 30.5               | 38.9             | 64.0            |
| Physik-Speed       | 50.0               | 58.3             | 63.6            |
| Physik-MNIST-Speed | 22.4               | 31.1             | 64.0            |

## Benutzungsanleitung

### Requirements

Tested Python Version 3.10.8 (Sollte auch auf tieferen Versionen laufen, aber nicht zu tief. :D)

Run:
```bash
pip install -r requirements.txt
```

um die Python Requirements zu installieren.


### Ausführen
Die Basis-Version ist im Ordner `src` zu finden. Die Physik basierte Version ist
im Ordner `src_physics` zu finden.

Zum **Trainieren** kann man die Datei `nn_train.py` ausführen. Der Parameter
`-h` zeigt noch weitere Konfigurationsmöglichkeiten an.

Zum **Testen** kann man die Datei `nn_test.py` ausführen. Auch hier kann `-h`
weitere Konfigurationsmöglichkeiten anzeigen.

## Zitierung
```biblatex
@software{,
    title = "ReSketch AI",
    author = {Ian Wasser, Robin Steiner}
    date = {2022-10-11}
    url = {https://github.com/LarsZauberer/Nachzeichner-KI}
}
```
