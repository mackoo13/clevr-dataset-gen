# CLEVR Dataset Generation for Polish Language 

This code is an extension of an existing project: [CLEVR](https://github.com/facebookresearch/clevr-dataset-gen). See the project website and original Readme for details. It generates questions about images with some objects arranged in different layouts.

#### Examples of output:

```
Czy żółta kula i szary walec mają ten sam rozmiar?

Czy liczba gumowych walców jest mniejsza, niż liczba czerwonych matowych walców?

Ile małych brązowych błyszczących rzeczy jest za małym walcem?

Jaki jest kolor gumowej rzeczy, która jest na prawo od dużego żółtego błyszczącego przedmiotu?
```

## Step 1: Translating the question templates
First we translate the text of the English templates to Polish, based on a set of regular expressions:

```bash
cd question_generation
python translate.py
```

All tokens (e.g. `<Z>`) are extended by an information about the grammatical properties it should have in the sentence (in Polish language these are noun case, number and gender). The correct forms are automatically selected by the translating script. 

#### Grammatical properties

Polish grammar is relatively complex and the word ending usually depends on its case, number and gender. This information is embedded in the token as in the following example:

````
... <Z:case=S,num=S,gen=S> ... <S:case=gen,num=pl,gen=> ...
````

Token `<Z>` is required to inherit `case`, `num` and `gen` properties from token `<S>` - which in turn is supposed to have genitive (`case=gen`) plural (`num=pl`) form. The gender will be determined after a specific word is inserted there, hence empty reference: `gen=`.


#### Example

```
"How many <Z> <C> <M> <S>s are there?"
```

Becomes the following:

```
"Ile jest <Z:case=S,num=S,gen=S> <C:case=S,num=S,gen=S> <M:case=S,num=S,gen=S> <S:case=gen,num=pl,gen=>?"
```




## Step 2: Generating Questions
From the perspective of the user, the next step is the same as in original CLEVR project.

```bash
cd question_generation
python generate_questions.py
```

At this stage the templates with grammatical properties are transformed into questions with all tokens replaced by words in correct grammatical forms. 

The procedure is considerably more complex than in English, one reason being the need to inflect nouns and adjectives.


#### Word inflection

The word endings are defined in `grammar_pl.json`:

```
...
"niebieski": {
    "case=nom,num=pl,gen=m": "e",       # -> niebieskie
    "case=nom,num=pl,gen=f": "e",       # -> niebieskie
    "case=gen,num=sg,gen=m": "ego",     # -> niebieskiego
    "case=gen,num=sg,gen=f": "ej"       # -> niebieskiej
    },
...
```

#### Word-specific properties

File `grammar_pl.json` also defines what noun case should be used after prepositions - e.g. instrumental is used with 'za' (behind), but genitive with 'na lewo od' (to the left)', and the gender of used nouns - e.g. 'sześcian' (cube) is masculine, but 'kula' (ball) is feminine. These propertoies affect all other words that inherit their forms from the parent (e.g. adjectives have the same form as the noun).

```
"dependent_forms": {
    "kula": "case=,num=,gen=f",
    "sześcian": "case=,num=,gen=m",
    ...
    "za": "case=inst,num=sg,gen=",
    "na lewo od": "case=gen,num=sg,gen="
}
```

#### Token substitution order

If the template contains words whoseform depend on other words, the tokens have to be replaced in a valid order. For example, the following tokens:

```
<R:case=,num=sg,gen=>
<Z:case=S,num=S,gen=S>
<S:case=R,num=sg,gen=>
``` 

Have to be resolved in strict order `R, S, Z`, even if the order in the template is different. 

## Step 3: The output

The file `output/CLEVR_questions.json` will then contain questions for the generated images.

You can [find more details about question generation in CLEVR project here](question_generation/README.md).
