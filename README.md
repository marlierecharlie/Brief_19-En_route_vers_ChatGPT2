# Brief 19 - En route vers ChatGPT2

## Contexte du projet

Vous êtes invités aux USA et précisément à la Silicon Valley en Californie pour représenter votre pays dans une compétition internationale sur le NLP. En effet OpenAI travaille sur un projet d'envergure et a besoin de talents pour développer leur nouveau chatbot GPT 2.

Pour cela vous allez relever plusieurs challenges. 

Après une phase extrêmement sélective, il ne reste plus que 16 candidats prétendant au titre de Champion avec un récompense de plus de 250 000 $ à la clé. Il intégrera la team OpenAI

## Logiciels et librairies utilisés

Les briefs seront codés avec le langage Python avec l'outil de travail Jupyter Notebook. Plusieurs **librairies de base** vont être utilisées : pandas, numpy et sklearn. Mais pour ce brief, des librairies spécifiques devront être importées : **NLTK, gensim, pyLDAvis et Azure**.

```
pip install pandas
pip install numpy
pip install -U scikit-learn
```

```
pip install --user -U nltk
pip install --upgrade gensim
pip install pyldavis
```

Pour utiliser Azure

```
pip install azure-cognitiveservices-vision-computervision
pip install azure-cognitiveservices-language-luis
pip install azure-cognitiveservices-language-textanalytics
```

## Explications

Pour répondre au brief, je commence par filtrer le texte d'une base de données convertie en dataframe. Voici une fonction qui permet de le faire rapidement:

``` python
def clean_text(df_column):
    clean_text = df_column.str.replace('[^\w\s]','').str.lower().str.replace('\d+', '') # ponctuation & lowercase & chiffres
    
    # supprime les stopwords 
    stop_words = stopwords.words('english')
    clean_text = clean_text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    
    # lemmatize le texte
    lmtzr = WordNetLemmatizer()
    clean_text = clean_text.apply(lambda lst:[lmtzr.lemmatize(word) for word in lst])
    clean_text = clean_text.apply(lambda x : "".join(x))
    
    return clean_text
 ```

### Exercice 2
Pour cet exercice, je vais créer un modèle Multinomial Naive Bayes formé sur une représentation Bag-of-Word (sac de mots) des textes afin d'effectuer une analyse de sentiment.

- La première étape consiste à utiliser la fonction CountVectorizer pour convertir les données textuelles en une matrice de termes de fréquences (BOW, bag of words) qui peut être utilisée pour entraîner un modèle.
- Ensuite, un objet du classificateur Bayes naïf multinomial est créé et entraîné sur les données BOW et les étiquettes cibles.
- Enfin, la fonction cross_val_score est utilisée pour évaluer la performance du modèle en utilisant la validation croisée avec 5 plis. Les scores obtenus sont affichés, ainsi que la moyenne de la précision validée.

Je vais maintenant utiliser des n-grammes de longueur 2 (bigrammes) pour représenter les données textuelles.

    Les bigrammes sont des paires consécutives de mots dans le texte. La représentation BOW sera basée  
    sur le nombre d'occurrences de chaque bigramme dans les données textuelles.
    
### Exercice 3    

Cet exercice permet de trouver les meilleurs hyperparamètres pour un classificateur Bayes utilisant la transformation BOW pour représenter les données textuelles. Pour se faire, je vais créer une pipiline (objet de canal) qui comprend 2 étapes:

- la conversion des données textuelles en une représentation BOW 
- l'entraînement d'un classificateur Bayes sur les données BOW.

Ensuite, j'instancie un dictionnaire qui définit les hyperparamètres qui seront testés dans la recherche sur grille: 
- la plage de n-grammes utilisée par CountVectorizer 
- la valeur d'alpha pour le classificateur Bayes.


        La valeur Alpha est un hyperparamètre utilisé dans le modèle Bayes naïf multinomial pour contrôler l'importance
        donnée aux mots rares dans les données textuelles. Plus la valeur d'alpha est petite, plus les mots rares
        auront un impact important sur les prédictions du modèle. À l'inverse, plus la valeur d'alpha est grande,
        plus les mots rares auront un impact faible et le modèle sera plus tolérant aux mots rares.


### Exercice 4

Dans cet exercice, je vais utiliser des données sur des emails non labélisés. Après les avoir filtrées, je vais les entrainer et utiliser le modèle LDA.  
Pour bien comprendre le modèle LDA (Latent Dirichlet Allocation), je vous invite [à lire ce PDF](https://alberto.bietti.me/files/rapport-lda.pdf)

