# Bayesian Image Restoration using Unadjusted Langevin Algorithm(ULA) 

Ce projet implémente une méthode de restauration d'images (défloutage/débruitage) basée sur l'algorithme **ULA (Unadjusted Langevin Algorithm)**, en utilisant la bibliothèque **DeepInverse**. L'approche combine des modèles de physique inverse avec des débruiteurs agissant comme des priors.

## Fonctionnalités
* **ULA Iterator Custom** : Implémentation itérative de l'algorithme de Langevin.
* **Priors Profonds** : Support de `DRUNet` et `DnCNN` comme opérateurs de régularisation.
* **Calcul de Métriques** : Script utilitaire pour comparer des images après traitement.

---

## Installation

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/alexandre-martel/Bayesian-imaging-PnP-ULA-algorithm.git
   cd Bayesian-imaging-PnP-ULA-algorithm
   ```

2. Installez les dépendances nécessaires :
    ```bash
    pip install torch torchvision numpy matplotlib pillow deepinv
    ```

## Utilisation

### Exécution du test principal

Pour lancer la restauration sur l'image de test (camera_man.jpg), exécutez votre script principal :
    ```bash
    python test.py
    ```

### Calcul de métriques indépendant

Un script est disponible pour calculer le PSNR et le SSIM entre deux images sauvegardées :
    ```bash
    python src/calculate_metrics.py -path1 "data/original.png" -path2 --path-to-img-2
    ```

## Configuration des Paramètres

Les performances de l'algorithme dépendent fortement du réglage des hyperparamètres dans le dictionnaire `algo_params_default` :

### 1. Choix du Débruiteur
* **DRUNet** : Très performant pour le flou complexe. 
    * `burn_in` conseillé : ~50.
    * `n_iter` conseillé : ~300.
* **DnCNN** : Plus rapide par itération mais nécessite souvent plus d'étapes.
    * `burn_in` conseillé : ~500.
    * `n_iter` conseillé : ~4000.

### 2. Paramètres Physiques et de Convergence
* **sigma_destruction** : Définit le niveau de bruit ajouté par le modèle physique (ex: 1/255**2).
* **denoiser_param** : La force du débruitage. Pour DRUNet, une valeur autour de 25/255 est un bon point de départ.
* **delta (Step size)** : Le pas de descente. Il est calculé automatiquement via la constante de Lipschitz de la physique ($L_y$) :

$$\delta = \frac{0.5}{\frac{L}{\text{denoiser\_param}} + L_y}$$

## Algorithme et Équations

L'échantillonnage de Langevin mis en œuvre suit la mise à jour :

$$x_{k+1} = x_k - \delta \nabla \log p(y|x_k) - \delta \nabla \log p(x_k) + \sqrt{2\delta} \epsilon$$

Où :
* $\nabla \log p(y|x_k)$ est le gradient du terme de fidélité aux données (fourni par `physics`).
* $\nabla \log p(x_k)$ est approximé par le score via le débruiteur choisi.
* $\epsilon \sim \mathcal{N}(0, I)$ est un bruit gaussien injecté pour explorer la distribution a posteriori.

L'image finale est obtenue en calculant la moyenne des échantillons après la période de **Burn-in** pour réduire la variance.