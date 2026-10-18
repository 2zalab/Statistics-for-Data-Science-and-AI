# Réponse au rapport d'évaluation (révision majeure)

Ce document répond point par point au rapport et indique où se trouvent les
modifications dans le manuscrit (`paper/main.pdf`) et dans le code.

## Point central : DA-HAC ne contrôlait pas le risque de première espèce

Le rapport avait raison. La version précédente (DA-HAC, devenue **DA-K**)
rejetait 9,9 % et 12,8 % d'hypothèses nulles vraies sous dépendance modérée et
forte (expérience 1), et environ 15 % dans l'expérience 2.

**Diagnostic.** Une étude pilote, réalisée sur des graines aléatoires distinctes
de celles des études principales, montre que ces faux positifs venaient presque
tous des réplications où τ̂² est tronqué à zéro :
- en dépendance modérée, 24 % de rejets quand τ̂² est tronqué, contre 4 % sinon ;
- le biais négatif de l'estimateur à noyau centré par pli n'était alors plus compensé.

**Correction.** La variance d'échantillonnage du test n'est plus estimée par un
noyau de Bartlett. Elle est calculée à partir du modèle de corrélation
exponentiel ajusté aux variogrammes des différences de pertes centrées par pli.
C'est la nouvelle procédure **DA-P** (section 4). Le choix a été fait sur les
graines pilotes, avant les études principales, et le manuscrit le dit
explicitement (section 5.1). L'ancienne version reste dans le manuscrit sous le
nom DA-K, comme élément de comparaison.

**Résultats sous H0.**

| Cadre | DA-P | DA-K (ancien) |
|---|---|---|
| Expérience 1 (aucune / faible / modérée / forte) | 3,1 / 4,2 / 3,4 / 2,4 % | 4,8 / 7,2 / 9,9 / 12,8 % |
| Expérience 2, CV par blocs | 2,3 / 4,7 / 10,0 / 11,0 % | 3,3 / 7,0 / 15,0 / 14,3 % |

- **Couverture des intervalles de DA-P** : 95 à 99 % dans l'expérience 1, rapportée
  séparément pour chaque δ et chaque niveau de dépendance (Fig. 3), comme demandé.
- **Limite restante** : un sur-rejet d'environ 10 % persiste dans l'expérience 2
  sous dépendance modérée et forte. Il est expliqué, et non masqué (voir plus bas).

## 1. Vérification de la dérivation de la variance

- **Proposition 2 (nouvelle, section 4.5).** Elle donne le biais exact de
  V̂ = V̂_u + ζ_K τ̂² sous le modèle complet, covariances entre moyennes de plis
  comprises. Ce biais se décompose en trois termes :
  - une partie liée à l'échantillonnage du test : biais des estimateurs et
    covariance c_u entre plis ;
  - une partie liée à l'entraînement, τ²[(1−ρ)ζ_K(ρ₀) − 1/K − ρ(1−1/K)].

  L'estimateur est sans biais si et seulement si ρ₀ = ρ, c_u = 0 et les
  estimateurs de variance sont sans biais. La variance est sous-estimée dès que
  ρ > ρ₀.
- **Corrélation des effets d'entraînement.** ρ₀ est désormais un paramètre
  explicite. L'analyse de sensibilité (section 5.3, tableau 4) croise le ρ réel
  ∈ {0 ; 0,25 ; 0,5 ; 0,75} avec le ρ₀ supposé ∈ {0 ; 1/K ; 0,5} :
  - DA-P est valide pour ρ ≤ 0,25, et sous dépendance modérée quel que soit ρ ;
  - sans dépendance spatio-temporelle, il rejette 10 % des nulles vraies si
    ρ = 0,5 et 15 % si ρ = 0,75 ;
  - toutes les autres méthodes font pire dans ces cas (14 à 29 %).

  C'est une limite affichée comme telle.
- **Cas limite τ̂² = 0** : tableau 2. La troncature est fréquente (63 à 76 % des
  réplications), mais le taux de rejet conditionnel reste de 2,5 à 6,1 %.

## 2. Règle de décision « indéterminée », évaluée empiriquement

Comme demandé, le seuil de 0,25 sur le rapport portée/domaine a été **évalué**
au lieu d'être postulé :
- grille de 7 portées, 800 réplications chacune (section 5.5, tableau 6) ;
- analyse de toutes les réplications nulles des deux expériences (Fig. 5).

Le résultat est négatif, et il est rapporté tel quel. Avec DA-P, les grands
rapports de portée réduisent la puissance mais ne causent pas de sur-rejet, et
les échecs de l'expérience 2 se situent à des rapports intermédiaires. Ce seuil
ne protégerait donc pas la validité, et il a été abandonné.

La règle retenue (équation 12) rend explicite la dépendance de la conclusion à
l'hypothèse non vérifiable sur ρ :
- **amélioration déclarée** si elle est significative avec ρ₀ = 1/K **et** avec ρ₀ = 0,5 ;
- **« indéterminée »** si elle n'est significative qu'avec ρ₀ = 1/K ;
- **non rejet** sinon.

Évaluation (tableau 7) :
- faux rejets ≤ 4 % dans l'expérience 1 ;
- 8,7 % dans le cas le plus difficile (expérience 2, dépendance modérée et forte) ;
- 0 à 11 % des vraies différences classées « indéterminées ».

Le rapport de portée et la taille effective d'échantillon restent fournis comme
diagnostics de l'information disponible.

## 3. Positionnement et méthodes concurrentes

- **État de l'art réécrit (section 2).** Il cite Hering & Genton (2011),
  Gilleland et al. (2018) et Watson et al. (2024). Un nouveau paragraphe
  « Positioning of this work » précise que la nouveauté ne réside pas dans la
  prise en compte de la dépendance en soi. Elle tient en quatre éléments :
  - la comparaison de procédures réentraînées par pli ;
  - la combinaison de la variance d'échantillonnage du test et de la variance
    d'entraînement ;
  - un variogramme estimé sur les différences centrées par pli ;
  - une règle de décision à trois issues.
- **Test de Hering-Genton ajouté** (version spatio-temporelle) dans toutes les
  expériences :
  - dans l'expérience 1, il est valide et un peu plus puissant que DA-P, ce qui
    est reconnu dans le texte ;
  - dans l'expérience 2, avec des modèles réentraînés, il rejette 17 à 23 % des
    nulles vraies, contre 10 à 11 % pour DA-P ;
  - sur données réelles, il déclare deux graines de XGBoost significativement
    différentes dans 20 % des paires pour les piétons, contre 3 % pour DA-P.

## 4. Expériences, couverture, multiplicité, validation temporelle

- **Robustesse (section 5.4, tableau 5).** Six scénarios : anisotropie 4:1,
  covariance gaussienne, hétéroscédasticité, portée non stationnaire, sites
  irréguliers. DA-P reste à 0,8–4,6 % de faux positifs, avec 95 à 99 % de
  couverture. La mauvaise spécification coûte de la puissance, pas de la validité.
- **Couverture** : rapportée séparément pour chaque δ et chaque niveau de dépendance (Fig. 3).
- **Interpolation et prévision (section 7.1)** : validation chronologique
  (forward-chaining) ajoutée pour les deux jeux horaires. Sur la météo de NYC,
  le classement s'inverse selon la tâche :
  - CV aléatoire : XGBoost apparaît meilleur ;
  - blocs temporels : les modèles sont équivalents ;
  - prévision : la forêt aléatoire est meilleure (p = 0,008, p de Holm = 0,023,
    robuste à ρ₀ = 0,5).
- **Multiplicité** : trois modèles comparés deux à deux, p-values ajustées par
  Holm, et études de cas présentées comme exploratoires.

## 5. Références et reproductibilité

- **Référence [14]** (Bouckaert & Frank) complétée. Toutes les références ont
  un DOI ou une URL.
- **Reproductibilité** :
  - code public (`src/dacomp`, 10 tests unitaires) ;
  - scripts des études pilotes et des études principales, résultats bruts en CSV ;
  - tous les tableaux et figures sont générés par `experiments/analyze.py` ;
  - la relance de l'expérience 1 reproduit exactement les résultats stockés
    (écart maximal entre p-values nul).

## Ce qui n'a pas été résolu

- **Calibration dans le cas le plus difficile.** DA-P n'est pas exactement
  calibré quand les effets d'entraînement sont fortement corrélés entre plis.
  C'est le cas de l'expérience 2 sous dépendance modérée et forte, et des
  scénarios où ρ ≥ 0,5. Une composante d'entraînement commune à tous les plis
  n'est pas identifiable à partir d'une seule validation croisée
  (Bengio & Grandvalet, 2004). La CV par blocs répétée ou imbriquée est proposée
  comme piste (section 8.4).
- **Variogramme exponentiel.** Il ne représente pas les dépendances périodiques
  (cycles journaliers et hebdomadaires).
- **Choix méthodologiques post hoc.** La règle à trois issues a été choisie après
  l'échec de la règle fondée sur la portée, constaté sur les réplications
  principales. Le manuscrit le signale et demande une confirmation sur d'autres
  plans d'expérience.
