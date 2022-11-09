```python
#import des librairies

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

```


```python
# affichage des données
positif= pd.read_pickle("imdb_raw_pos.pickle")
negatif = pd.read_pickle('imdb_raw_neg.pickle')

```


```python
#Création de tableau avec dataframe
positif={"commentaire":positif}
df1=pd.DataFrame(positif)
df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>commentaire</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I went and saw this movie last night after bei...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Actor turned director Bill Paxton follows up h...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>As a recreational golfer with some knowledge o...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>I saw this film in a sneak preview, and it is ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bill Paxton has taken the true story of the 19...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>12495</th>
      <td>I was extraordinarily impressed by this film. ...</td>
    </tr>
    <tr>
      <th>12496</th>
      <td>Although I'm not a golf fan, I attended a snea...</td>
    </tr>
    <tr>
      <th>12497</th>
      <td>From the start of "The Edge Of Love", the view...</td>
    </tr>
    <tr>
      <th>12498</th>
      <td>This movie, with all its complexity and subtle...</td>
    </tr>
    <tr>
      <th>12499</th>
      <td>I've seen this story before but my kids haven'...</td>
    </tr>
  </tbody>
</table>
<p>12500 rows × 1 columns</p>
</div>




```python
df1["avis"]=1
df1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>commentaire</th>
      <th>avis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I went and saw this movie last night after bei...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Actor turned director Bill Paxton follows up h...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>As a recreational golfer with some knowledge o...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>I saw this film in a sneak preview, and it is ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bill Paxton has taken the true story of the 19...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>12495</th>
      <td>I was extraordinarily impressed by this film. ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12496</th>
      <td>Although I'm not a golf fan, I attended a snea...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12497</th>
      <td>From the start of "The Edge Of Love", the view...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12498</th>
      <td>This movie, with all its complexity and subtle...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12499</th>
      <td>I've seen this story before but my kids haven'...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>12500 rows × 2 columns</p>
</div>




```python
negatif={"commentaire": negatif}
df2=pd.DataFrame(negatif)
df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>commentaire</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Once again Mr. Costner has dragged out a movie...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>This is an example of why the majority of acti...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>First of all I hate those moronic rappers, who...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Not even the Beatles could write songs everyon...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brass pictures (movies is not a fitting word f...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>12495</th>
      <td>I occasionally let my kids watch this garbage ...</td>
    </tr>
    <tr>
      <th>12496</th>
      <td>When all we have anymore is pretty much realit...</td>
    </tr>
    <tr>
      <th>12497</th>
      <td>The basic genre is a thriller intercut with an...</td>
    </tr>
    <tr>
      <th>12498</th>
      <td>Four things intrigued me as to this film - fir...</td>
    </tr>
    <tr>
      <th>12499</th>
      <td>David Bryce's comments nearby are exceptionall...</td>
    </tr>
  </tbody>
</table>
<p>12500 rows × 1 columns</p>
</div>




```python
df2["avis"]=0
df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>commentaire</th>
      <th>avis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Once again Mr. Costner has dragged out a movie...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>This is an example of why the majority of acti...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>First of all I hate those moronic rappers, who...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Not even the Beatles could write songs everyon...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brass pictures (movies is not a fitting word f...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>12495</th>
      <td>I occasionally let my kids watch this garbage ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12496</th>
      <td>When all we have anymore is pretty much realit...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12497</th>
      <td>The basic genre is a thriller intercut with an...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12498</th>
      <td>Four things intrigued me as to this film - fir...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12499</th>
      <td>David Bryce's comments nearby are exceptionall...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>12500 rows × 2 columns</p>
</div>




```python
vecteur=pd.concat([df1,df2])
vecteur
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>commentaire</th>
      <th>avis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I went and saw this movie last night after bei...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Actor turned director Bill Paxton follows up h...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>As a recreational golfer with some knowledge o...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>I saw this film in a sneak preview, and it is ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bill Paxton has taken the true story of the 19...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>12495</th>
      <td>I occasionally let my kids watch this garbage ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12496</th>
      <td>When all we have anymore is pretty much realit...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12497</th>
      <td>The basic genre is a thriller intercut with an...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12498</th>
      <td>Four things intrigued me as to this film - fir...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12499</th>
      <td>David Bryce's comments nearby are exceptionall...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>25000 rows × 2 columns</p>
</div>




```python
x=vecteur["commentaire"]
y=vecteur["avis"]
```


```python
cv= CountVectorizer
```


```python
x
```




    0        I went and saw this movie last night after bei...
    1        Actor turned director Bill Paxton follows up h...
    2        As a recreational golfer with some knowledge o...
    3        I saw this film in a sneak preview, and it is ...
    4        Bill Paxton has taken the true story of the 19...
                                   ...                        
    12495    I occasionally let my kids watch this garbage ...
    12496    When all we have anymore is pretty much realit...
    12497    The basic genre is a thriller intercut with an...
    12498    Four things intrigued me as to this film - fir...
    12499    David Bryce's comments nearby are exceptionall...
    Name: commentaire, Length: 25000, dtype: object




```python
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.3)

```


```python
x_train.shape
```




    (17500,)




```python
#vecteurs
cv= CountVectorizer(stop_words="english",binary=False,ngram_range=(1,1))
```


```python
x_traincv=cv.fit_transform(x_train)
```


```python
x_testcv=cv.fit_transform(x_test)
```


```python
cv.vocabulary_
```




    {'far': 14463,
     'worst': 43909,
     'horror': 19130,
     'movie': 26252,
     'make': 24122,
     'watched': 43045,
     'shame': 35219,
     'block': 4562,
     'buster': 5819,
     'carrying': 6374,
     'type': 41002,
     'crap': 9237,
     'ask': 2619,
     'refund': 32282,
     'think': 39646,
     'br': 5124,
     'bad': 3230,
     'stop': 37763,
     'just': 21635,
     '15': 79,
     'minutes': 25603,
     'watching': 43051,
     'fun': 16023,
     'fuzzy': 16112,
     'youtube': 44266,
     'movies': 26264,
     'piece': 29568,
     'dropping': 12204,
     'marketing': 24462,
     'dude': 12293,
     'silver': 35788,
     'tongue': 40093,
     'thing': 39642,
     'establishment': 13655,
     'blockbuster': 4563,
     've': 42220,
     'seen': 34828,
     'life': 23162,
     'looking': 23566,
     'nice': 27025,
     'theatrical': 39564,
     'effect': 12716,
     'skip': 36042,
     'watch': 43042,
     'camp': 6061,
     'value': 42116,
     'advice': 1120,
     'gather': 16338,
     'sarcastic': 34158,
     'friends': 15873,
     'strictly': 37922,
     'purpose': 31277,
     'making': 24131,
     'silliest': 35783,
     'airport': 1366,
     'probably': 30746,
     '70': 473,
     'disaster': 11291,
     'milk': 25462,
     'franchise': 15709,
     'got': 17093,
     'producers': 30788,
     'sequel': 35004,
     'throws': 39762,
     'action': 892,
     'silly': 35785,
     'subplots': 38144,
     'gratuitous': 17270,
     'star': 37421,
     'appearances': 2227,
     'goofier': 17028,
     'elements': 12874,
     'attempt': 2841,
     'biggest': 4232,
     'concorde': 8338,
     '79': 493,
     '1970': 226,
     'box': 5094,
     'office': 27698,
     'results': 32915,
     'wonder': 43782,
     'genre': 16494,
     'overstayed': 28360,
     'welcome': 43225,
     'film': 14925,
     'opens': 27891,
     'catchy': 6496,
     'score': 34552,
     'lalo': 22513,
     'schifrin': 34422,
     'backed': 3193,
     'impressive': 19883,
     'shots': 35538,
     'titular': 39976,
     'aircraft': 1351,
     'glory': 16842,
     'plane': 29776,
     'used': 41990,
     'involved': 20847,
     'crash': 9253,
     'july': 21592,
     '2000': 271,
     'credits': 9320,
     'appear': 2225,
     'glimpse': 16805,
     'cast': 6440,
     'composed': 8230,
     'faded': 14309,
     'hollywood': 18928,
     'stars': 37455,
     'tv': 40936,
     'actors': 910,
     'popular': 30178,
     'plot': 29898,
     'sillier': 35782,
     'george': 16530,
     'kennedy': 21893,
     'role': 33502,
     'joe': 21414,
     'patroni': 28894,
     'pilot': 29615,
     'piloted': 29616,
     'alain': 1413,
     'delon': 10483,
     'en': 13138,
     'route': 33687,
     'france': 15704,
     'summer': 38332,
     'olympics': 27799,
     'board': 4686,
     'reporter': 32695,
     'maggie': 24031,
     'whelan': 43334,
     'susan': 38556,
     'blakely': 4441,
     'discovered': 11346,
     'boyfriend': 5112,
     'renowned': 32620,
     'weapons': 43127,
     'manufacturer': 24351,
     'kevin': 21936,
     'harrison': 18097,
     'robert': 33411,
     'wagner': 42814,
     'selling': 34898,
     'terrorist': 39465,
     'prevent': 30627,
     'revealing': 33012,
     'news': 26996,
     'world': 43876,
     'sends': 34928,
     'advance': 1090,
     'missiles': 25712,
     'best': 4116,
     'saboteurs': 33896,
     'landing': 22567,
     'dumber': 12339,
     'alcoholic': 1449,
     'myrna': 26522,
     'loy': 23726,
     'singing': 35870,
     'nun': 27441,
     'cicely': 7330,
     'tyson': 41027,
     'transporting': 40475,
     'live': 23390,
     'human': 19314,
     'heart': 18318,
     'cooler': 8835,
     'martha': 24520,
     'raye': 31864,
     'woman': 43772,
     'bladder': 4428,
     'condition': 8359,
     'character': 6835,
     'doesn': 11751,
     'deeper': 10290,
     'walker': 42865,
     'pot': 30308,
     'smoking': 36367,
     'saxophonist': 34283,
     'arguably': 2400,
     'annoying': 2039,
     'eddie': 12644,
     'albert': 1431,
     'married': 24496,
     'old': 27763,
     'wife': 43504,
     'sybil': 38760,
     'danning': 9940,
     'avery': 3036,
     'schreiber': 34491,
     'russian': 33846,
     'coach': 7724,
     'deaf': 10111,
     'daughter': 10034,
     'finally': 14968,
     'love': 23679,
     'story': 37791,
     'jon': 21462,
     'davidson': 10052,
     'gymnast': 17712,
     'andrea': 1899,
     'marcovici': 24393,
     'sour': 36806,
     'eye': 14217,
     'mercedes': 25181,
     'mccambidge': 24803,
     'plus': 29948,
     'gets': 16578,
     'closer': 7650,
     'boat': 4700,
     'episode': 13477,
     'cameos': 6046,
     'charo': 6898,
     'pet': 29340,
     'chihuahua': 7098,
     'bibi': 4199,
     'anderson': 1892,
     'buffs': 5613,
     'doubt': 11958,
     'real': 31916,
     'kick': 21970,
     'balls': 3353,
     'entry': 13431,
     'long': 23545,
     'strain': 37824,
     'sense': 34946,
     'word': 43845,
     'true': 40766,
     'rating': 31807,
     'days': 10082,
     'moving': 26269,
     'showed': 35565,
     'hard': 18022,
     'disrespected': 11536,
     'gay': 16370,
     'community': 8124,
     'really': 31935,
     'etherizes': 13690,
     'passion': 28811,
     'explores': 14111,
     'feelings': 14688,
     'characters': 6851,
     'holds': 18895,
     'depth': 10689,
     'compassion': 8146,
     'speaks': 36924,
     'man': 24211,
     'personally': 29284,
     'high': 18657,
     'time': 39879,
     'homosexual': 18999,
     'religions': 32467,
     'longer': 23547,
     'hold': 18887,
     'head': 18254,
     'beautiful': 3772,
     'self': 34881,
     'opener': 27886,
     'believe': 3922,
     'anti': 2088,
     'schools': 34484,
     'kid': 21980,
     'dislikes': 11439,
     'refer': 32229,
     'hope': 19072,
     'like': 23206,
     'labelled': 22429,
     'directors': 11239,
     'stick': 37648,
     'tell': 39314,
     'eyes': 14233,
     'little': 23383,
     'videos': 42447,
     'dvds': 12441,
     'ray': 31863,
     'winston': 43645,
     'rest': 32879,
     'good': 17001,
     'changing': 6799,
     'robin': 33421,
     'hood': 19041,
     'works': 43871,
     'great': 17301,
     'stories': 37784,
     'twists': 40976,
     'way': 43099,
     'shot': 35534,
     'untrained': 41843,
     'trained': 40360,
     'miss': 25705,
     'interpreted': 20669,
     'ropey': 33598,
     'adds': 975,
     'films': 14946,
     'absorption': 711,
     'audience': 2901,
     'green': 17319,
     'hillsides': 18713,
     'contrast': 8743,
     'lush': 23842,
     'sunny': 38369,
     'lit': 23361,
     'forest': 15539,
     'dark': 9968,
     'corridors': 8985,
     'dungeons': 12377,
     'castles': 6453,
     'definitive': 10362,
     'interpretation': 20667,
     'legend': 22913,
     'stress': 37906,
     'chance': 6782,
     'guys': 17698,
     'computer': 8265,
     'plots': 29907,
     'names': 26613,
     'places': 29755,
     'faces': 14275,
     'changed': 6795,
     'complete': 8192,
     'boredom': 4962,
     'hubby': 19263,
     'handsome': 17942,
     'rich': 33162,
     'busy': 5826,
     'pay': 28955,
     'attention': 2855,
     'gorgeous': 17059,
     'situation': 35937,
     'constantly': 8601,
     'tries': 40643,
     'remedy': 32521,
     'friend': 15869,
     'neighbor': 26877,
     'problem': 30752,
     'interested': 20626,
     'hubbies': 19262,
     'grass': 17254,
     'greener': 17324,
     'throw': 39755,
     'sexy': 35152,
     'scheming': 34413,
     'designs': 10772,
     'everybody': 13781,
     'lot': 23634,
     'stripping': 37947,
     'dipping': 11220,
     'salad': 33995,
     'bowl': 5086,
     'unintelligible': 41565,
     'quality': 31385,
     'bodies': 4728,
     'pretty': 30619,
     'contact': 8649,
     'simulated': 35843,
     'shown': 35578,
     'late': 22678,
     'night': 27081,
     'erotica': 13566,
     'jay': 21219,
     'leno': 22986,
     'sleep': 36157,
     'fan': 14418,
     'original': 28039,
     'commandments': 8049,
     'weep': 43189,
     'inside': 20450,
     'granted': 17234,
     'hour': 19197,
     'currently': 9686,
     'painful': 28499,
     'felt': 14728,
     'duty': 12431,
     'warn': 42975,
     'away': 3079,
     'fans': 14448,
     'subjected': 38108,
     'bastardization': 3607,
     'didn': 11064,
     'possible': 30281,
     'actually': 918,
     'special': 36931,
     'effects': 12722,
     'worse': 43897,
     '1950s': 204,
     '2006': 278,
     'remake': 32501,
     'proves': 31021,
     'wrong': 43995,
     'forgive': 15557,
     'lame': 22528,
     'craptastic': 9252,
     'dialogue': 11003,
     'melodramatic': 25092,
     'lifetime': 23178,
     'style': 38068,
     'schlockiness': 34440,
     'stilted': 37672,
     'wax': 43094,
     'figures': 14909,
     'come': 8000,
     'acting': 890,
     'makes': 24126,
     'll': 23421,
     'rewrite': 33109,
     'drown': 12214,
     'moses': 26150,
     'red': 32147,
     'sea': 34701,
     'terrible': 39445,
     '10': 17,
     'year': 44156,
     'boy': 5105,
     'loved': 23685,
     'target': 39092,
     'demographic': 10543,
     'guess': 17555,
     'hit': 18790,
     'spot': 37156,
     'dull': 12325,
     'adult': 1083,
     'relate': 32420,
     'animated': 1981,
     'lego': 22928,
     'mod': 25820,
     'tech': 39227,
     'ones': 27835,
     'thought': 39697,
     'choice': 7188,
     'virtues': 42582,
     'bionicle': 4311,
     'bit': 4350,
     'odd': 27645,
     'unity': 41591,
     'destiny': 10817,
     'thinks': 39651,
     'sound': 36791,
     'fascistic': 14527,
     'especially': 13627,
     'freedom': 15802,
     'equality': 13501,
     'justice': 21637,
     'oh': 27730,
     'sell': 34892,
     'kind': 22058,
     '74': 482,
     'minute': 25601,
     'advertisement': 1113,
     'sports': 37152,
     'flics': 15247,
     'flic': 15239,
     'storytelling': 37796,
     'fine': 14984,
     'reliably': 32452,
     'fantastic': 14456,
     'hours': 19199,
     'entertainment': 13382,
     'greatest': 17304,
     'game': 16211,
     'qualifies': 31380,
     'mightily': 25410,
     'moves': 26250,
     'paxton': 28954,
     'gone': 16990,
     'director': 11237,
     'school': 34472,
     'ron': 33560,
     'howard': 19224,
     'richie': 33171,
     'cunningham': 9651,
     'happy': 18004,
     'look': 23559,
     'immense': 19736,
     'body': 4731,
     'work': 43856,
     'did': 11059,
     'camera': 6047,
     'actor': 909,
     'superstar': 38430,
     'indication': 20111,
     'things': 39644,
     'follow': 15440,
     'wonderful': 43786,
     'cinematography': 7363,
     'direction': 11231,
     'elias': 12897,
     'koteas': 22307,
     'shia': 35393,
     'lebeouf': 22858,
     'marnie': 24482,
     'mcphail': 24904,
     'josh': 21486,
     'flitter': 15283,
     'stephen': 37594,
     'marcus': 24394,
     'justin': 21646,
     'ashforth': 2604,
     'feel': 14683,
     'cinema': 7350,
     'chuck': 7288,
     'norris': 27287,
     'accomplished': 794,
     'meaning': 24944,
     'reason': 31958,
     'apparently': 2216,
     'trucker': 40760,
     'billy': 4272,
     'dawes': 10062,
     'augenstein': 2917,
     'given': 16736,
     'delivery': 10479,
     'detoured': 10879,
     'forced': 15502,
     'jerkwater': 21299,
     'town': 40274,
     'foolish': 15470,
     'local': 23455,
     'corrupt': 8991,
     'judge': 21547,
     'trimmings': 40656,
     'murdock': 26425,
     'runs': 33823,
     'arrested': 2500,
     'phony': 29476,
     'charges': 6864,
     'brought': 5479,
     'denies': 10583,
     'run': 33810,
     'hick': 18632,
     'cops': 8879,
     'beat': 3752,
     'disappears': 11272,
     'brother': 5474,
     'jd': 21232,
     'john': 21427,
     'goes': 16930,
     'soon': 36718,
     'finds': 14983,
     'buy': 5857,
     'loser': 23626,
     'begins': 3859,
     'meantime': 24955,
     'female': 14730,
     'calls': 6015,
     'truckers': 40761,
     'cb': 6596,
     'radio': 31572,
     'rushing': 33837,
     'demolishing': 10545,
     'lol': 23525,
     'big': 4223,
     'rigs': 33263,
     'eventually': 13769,
     'beats': 3765,
     'thugs': 39772,
     'starts': 37469,
     'ok': 27748,
     'smash': 36325,
     'losers': 23627,
     'virtually': 42580,
     'ending': 13214,
     'house': 19200,
     'rammed': 31685,
     'rig': 33241,
     'know': 22225,
     'happens': 17998,
     'does': 11749,
     'trash': 40497,
     'takes': 38959,
     'thug': 39771,
     'cop': 8862,
     'end': 13193,
     'cares': 6271,
     'manages': 24218,
     'hand': 17904,
     'broken': 5445,
     'fighting': 14899,
     'cash': 6412,
     'craze': 9278,
     'barley': 3503,
     'use': 41989,
     'adding': 964,
     'near': 26801,
     'drag': 12041,
     'bored': 4961,
     'need': 26835,
     'pass': 28793,
     'went': 43251,
     'better': 4151,
     'beginning': 3857,
     '1942': 194,
     'sherlock': 35376,
     'holmes': 18934,
     'portrayed': 30233,
     'basil': 3581,
     'rathbone': 31805,
     'set': 35085,
     'modern': 25832,
     'britain': 5398,
     'purists': 31263,
     'praised': 30393,
     'entries': 13429,
     'series': 35044,
     'produced': 30786,
     'fox': 15665,
     'dismissed': 11458,
     '12': 56,
     'features': 14654,
     'followed': 15442,
     'avid': 3039,
     'reader': 31899,
     'appreciate': 2271,
     'initial': 20358,
     'bias': 4194,
     'fact': 14291,
     'setting': 35094,
     'war': 42937,
     'ii': 19627,
     'period': 29210,
     'british': 5404,
     'propaganda': 30913,
     'evident': 13794,
     'execution': 13943,
     'handled': 17933,
     'care': 6251,
     'previous': 30636,
     'voice': 42676,
     'terror': 39461,
     'fares': 14476,
     'reasons': 31963,
     'roy': 33711,
     'william': 43550,
     'neil': 26886,
     'direct': 11227,
     'second': 34772,
     'pitted': 29719,
     'nazis': 26784,
     'inclusion': 20004,
     'professor': 30811,
     'moriarty': 26085,
     'battle': 3653,
     'intellect': 20567,
     'rivals': 33354,
     'lionel': 23318,
     'atwill': 2887,
     'performance': 29192,
     'criticized': 9422,
     'inferior': 20244,
     'zucco': 44475,
     'henry': 18524,
     'daniel': 9930,
     'script': 34649,
     'seeing': 34818,
     'capable': 6161,
     'diverse': 11658,
     'enthusiastic': 13393,
     'roles': 33504,
     'exceptional': 13886,
     'enjoying': 13299,
     'blame': 4444,
     'writing': 43992,
     'example': 13857,
     'introduction': 20760,
     'immediately': 19735,
     'establish': 13651,
     'superbly': 38385,
     'intelligent': 20577,
     'screen': 34625,
     'shows': 35579,
     'brilliant': 5364,
     'deductions': 10274,
     'shortcomings': 35521,
     'climax': 7594,
     'initially': 20360,
     'upper': 41916,
     'blood': 4584,
     'slowly': 36270,
     'drained': 12056,
     'flaws': 15199,
     'point': 29993,
     'opening': 27888,
     'aren': 2382,
     'germans': 16557,
     'fooled': 15467,
     'easily': 12560,
     'maybe': 24762,
     'book': 4881,
     'seller': 34896,
     'waltzes': 42902,
     'bar': 3443,
     'says': 34290,
     'gonna': 16994,
     'tobel': 40005,
     'answer': 2056,
     'door': 11883,
     'weeks': 43182,
     'hiding': 18649,
     'allow': 1568,
     'walk': 42861,
     'certainly': 6710,
     'danger': 9922,
     'leaving': 22854,
     'secret': 34778,
     'bomb': 4816,
     'parts': 28778,
     'genius': 16488,
     'scientists': 34521,
     'note': 27332,
     'fiancé': 14852,
     'thinking': 39650,
     'endangering': 13195,
     'superb': 38383,
     'scientist': 34520,
     'trying': 40799,
     'save': 34246,
     'england': 13266,
     'isn': 20984,
     'concept': 8298,
     'harsh': 18108,
     'obvious': 27599,
     'god': 16896,
     'supposed': 38467,
     'fit': 15086,
     'chest': 7049,
     'small': 36308,
     'entire': 13401,
     'finding': 14981,
     'ex': 13831,
     'disguise': 11396,
     'explain': 14071,
     'times': 39889,
     'executed': 13940,
     'confusing': 8450,
     'fashion': 14529,
     'fourth': 15658,
     'okay': 27749,
     'dead': 10099,
     'assumes': 2717,
     'couldn': 9058,
     'wasn': 43022,
     'assume': 2715,
     'guy': 17695,
     'face': 14268,
     'lake': 22507,
     'lucky': 23765,
     'fifth': 14892,
     'confusion': 8452,
     'supposedly': 38468,
     'dying': 12473,
     'operating': 27900,
     'table': 38872,
     'recover': 32114,
     'quickly': 31460,
     'sure': 38485,
     'lost': 23632,
     'pints': 29663,
     'saving': 34251,
     'day': 10070,
     'despite': 10799,
     'highly': 18672,
     'entertaining': 13380,
     'disguises': 11398,
     'mystery': 26531,
     'dancing': 9913,
     'men': 25136,
     'scenes': 34385,
     'figuring': 14910,
     'analyzing': 1853,
     'dennis': 10598,
     'hoey': 18866,
     'inspector': 20485,
     'lestrade': 23042,
     'final': 14964,
     'interrogation': 20679,
     'scene': 34381,
     'kinda': 22059,
     'bothers': 5016,
     'sequences': 35008,
     'overall': 28253,
     'place': 29750,
     'solidified': 36648,
     'obsession': 27576,
     'classic': 7485,
     'absolutely': 701,
     'amazing': 1694,
     'humor': 19353,
     'music': 26453,
     'message': 25251,
     'clever': 7567,
     'particularly': 28766,
     'vietnam': 42461,
     'interesting': 20627,
     'army': 2474,
     'officials': 27704,
     'enforcing': 13249,
     'draft': 12035,
     'ridiculous': 33213,
     'actual': 914,
     'conclusion': 8329,
     'leaves': 22851,
     'seething': 34835,
     'anger': 1945,
     'violence': 42552,
     'wow': 43931,
     'cool': 8833,
     'unique': 41581,
     'musical': 26454,
     'opposed': 27928,
     'evita': 13808,
     'wizard': 43737,
     'oz': 28428,
     'lyrics': 23898,
     'don': 11835,
     'mood': 26021,
     'visuals': 42625,
     'songs': 36697,
     'dialog': 11001,
     'donna': 11852,
     'upbeat': 41890,
     'song': 36696,
     'emphasizes': 13101,
     'flesh': 15226,
     'failures': 14333,
     'driving': 12184,
     'intense': 20585,
     'minor': 25587,
     'key': 21938,
     'notice': 27343,
     'lsd': 23733,
     'flattering': 15182,
     'definitely': 10359,
     'going': 16942,
     'drugs': 12226,
     'intended': 20582,
     'considered': 8559,
     'negative': 26854,
     'comment': 8062,
     'say': 34284,
     'hear': 18305,
     'pauly': 28931,
     'shore': 35514,
     'laugh': 22710,
     'butt': 5837,
     'mess': 25250,
     'wasting': 43038,
     'talent': 38968,
     'cute': 9738,
     'coed': 7807,
     'carla': 6291,
     'gugino': 17571,
     'south': 36814,
     'dakota': 9858,
     'invites': 20838,
     'california': 5989,
     'college': 7905,
     'dorm': 11915,
     'counselor': 9070,
     'home': 18952,
     'share': 35257,
     'thanksgiving': 39543,
     'notable': 27326,
     'members': 25114,
     'lane': 22580,
     'smith': 36349,
     'cindy': 7345,
     'pickett': 29539,
     'mason': 24585,
     'adams': 939,
     'drop': 12200,
     'tiffani': 39842,
     'amber': 1700,
     'thiessen': 39634,
     'step': 37585,
     'remember': 32523,
     'insisted': 20467,
     'rent': 32621,
     'weirdest': 43210,
     'wish': 43682,
     'exactly': 13836,
     'dust': 12415,
     'pretends': 30606,
     'doctor': 11722,
     'sucked': 38228,
     'weird': 43208,
     'goings': 16943,
     'black': 4390,
     'help': 18479,
     'ends': 13226,
     'caught': 6550,
     'young': 44250,
     'women': 43779,
     'scratching': 34608,
     'surface': 38491,
     'randomly': 31722,
     'happen': 17992,
     'people': 29142,
     'whipped': 43367,
     'stabbed': 37292,
     'pair': 28513,
     'creepy': 9339,
     'girls': 16720,
     'walked': 42863,
     'shining': 35427,
     'tom': 40060,
     'savini': 34253,
     'imaginary': 19682,
     'religious': 32468,
     'figure': 14906,
     'decides': 10209,
     'want': 42932,
     'anymore': 2153,
     'plan': 29775,
     'sharing': 35264,
     'jesus': 21317,
     'suffering': 38256,
     'level': 23071,
     'sex': 35133,
     'nudity': 27416,
     'disappointment': 11277,
     'mistaking': 25734,
     'intent': 20594,
     'stoner': 37747,
     'paradise': 28648,
     'unexpected': 41453,
     'stuff': 38018,
     'stops': 37773,
     'travels': 40528,
     'girlfriend': 16715,
     'birth': 4331,
     'canal': 6089,
     'reborn': 31991,
     'chalk': 6754,
     'crazy': 9284,
     'nature': 26732,
     'flick': 15241,
     'wordy': 43854,
     'points': 30004,
     'hammered': 17885,
     'bunch': 5705,
     'stoned': 37746,
     'prove': 31014,
     'asset': 2685,
     'acted': 889,
     'vicious': 42415,
     'murderous': 26422,
     'gangster': 16241,
     'bryan': 5541,
     'brown': 5486,
     'teaching': 39199,
     'son': 36686,
     'macrame': 23973,
     'succeeds': 38206,
     'drama': 12062,
     'ledger': 22878,
     'gilrfriend': 16679,
     'succeed': 38203,
     'hilarious': 18692,
     'ironic': 20894,
     'humour': 19359,
     'bank': 3408,
     'robbers': 33403,
     'competition': 8170,
     'winners': 43634,
     'reaction': 31891,
     'busker': 5813,
     'revenge': 33024,
     'worth': 43911,
     'turkey': 40887,
     'overrated': 28338,
     'franco': 15719,
     'gave': 16364,
     'classics': 7488,
     'las': 22648,
     'vampiras': 42122,
     'vampyros': 42132,
     'lesbos': 23030,
     'yes': 44194,
     'adore': 1064,
     'howlers': 19233,
     'doris': 11908,
     'wishman': 43687,
     'dwain': 12447,
     'esper': 13628,
     'ed': 12642,
     'wood': 43804,
     'jr': 21530,
     'proved': 31015,
     'rated': 31802,
     'start': 37460,
     'screenplay': 34631,
     'idiotic': 19590,
     'utmost': 42029,
     'unbelievably': 41184,
     'directing': 11230,
     'nonexistent': 27247,
     'cue': 9608,
     'repeatedly': 32651,
     'taken': 38954,
     ...}




```python
#regression logistique
reg=LogisticRegression(max_iter=35000)
```


```python
x_traincv.shape
```




    (17500, 63966)




```python
y_train.shape
```




    (17500,)




```python
reg_test=reg.fit(x_traincv,y_train)
```


```python
reg_test.score(x_traincv,y_train)
```




    0.9985714285714286




```python
x_testcv.shape
```




    (7500, 44507)




```python
y_test.shape
```




    (7500,)




```python
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.3, random_state=4)

```


```python
cv=CountVectorizer()
```


```python
x_traincv=cv.fit_transform(vecteur)
```


```python
x_traincv.toarray()
```




    array([[0, 1],
           [1, 0]], dtype=int64)




```python
cv.get_feature_names()
```

    C:\Users\akoyv\anaconda3\lib\site-packages\sklearn\utils\deprecation.py:87: FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.
      warnings.warn(msg, category=FutureWarning)
    




    ['avis', 'commentaire']




```python
cv1= CountVectorizer()
```


```python
x_traincv= cv1.fit_transform(x_train)
```


```python
a=x_traincv.toarray()
```


    ---------------------------------------------------------------------------

    MemoryError                               Traceback (most recent call last)

    Input In [34], in <cell line: 1>()
    ----> 1 a=x_traincv.toarray()
    

    File ~\anaconda3\lib\site-packages\scipy\sparse\compressed.py:1039, in _cs_matrix.toarray(self, order, out)
       1037 if out is None and order is None:
       1038     order = self._swap('cf')[0]
    -> 1039 out = self._process_toarray_args(order, out)
       1040 if not (out.flags.c_contiguous or out.flags.f_contiguous):
       1041     raise ValueError('Output array must be C or F contiguous')
    

    File ~\anaconda3\lib\site-packages\scipy\sparse\base.py:1202, in spmatrix._process_toarray_args(self, order, out)
       1200     return out
       1201 else:
    -> 1202     return np.zeros(self.shape, dtype=self.dtype, order=order)
    

    MemoryError: Unable to allocate 8.35 GiB for an array with shape (17500, 64055) and data type int64



```python
a
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Input In [35], in <cell line: 1>()
    ----> 1 a
    

    NameError: name 'a' is not defined



```python
cv= CountVectorizer(min_df=0, max_df=1, binary=False)
cv_trainreviews= cv.fit_transform(vecteur)

cv_train_reviews=cv.fit_transform(vecteur)

print("positif:", vecteur.shape)
print("negatif:", vecteur.shape)
```

    positif: (25000, 2)
    negatif: (25000, 2)
    
