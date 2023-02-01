# Brief_19-En_route_vers_ChatGPT2

## Contexte du projet

Vous êtes invités aux USA et précisément à la Silicon Valley en Californie pour représenter votre pays dans une compétition internationale sur le NLP. En effet OpenAI travaille sur un projet d'envergure et a besoin de talents pour développer leur nouveau chatbot GPT 2.

Pour cela vous allez relever plusieurs challenges. 

Après une phase extrêmement sélective, il ne reste plus que 16 candidats prétendant au titre de Champion avec un récompense de plus de 250 000 $ à la clé. Il intégrera la team OpenAI

## Logiciels utilisés

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
