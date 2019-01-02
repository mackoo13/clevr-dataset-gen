from __future__ import print_function
import argparse, json, os
import itertools
import re

param_types = {'D': 'Made',
               'F': 'Big',
               'H': 'What_like',
               'I': 'It',
               'J': 'Its',
               'O': 'Other',
               'W': 'Which'}

class Form:
  def __init__(self, props):
    self.props = props

  def str(self):
    return ','.join([k + '=' + v for (k, v) in self.props.items()])

  def eq(self, other):
    for k, v in self.props:
      if other.props[k] != v:
        return False
    return True

  def merge(self, other):
    for k in self.props:
      if other.props[k] != '':
        self.props[k] = other.props[k]
    return

  def eval(self, forms):
    while True:
      for k, v in self.props.items():
        if v in forms and forms[v].props[k] != '':
          self.props[k] = forms[v].props[k]
          continue
      break

  def is_final(self):
    return all([v.islower() or v == '' for v in self.props.values()])

class WordInflector:
  def __init__(self, t, forms):
    self.t = t
    self.forms = forms

  def inflect_big(self):
    words = self.t.split(' ')

    num_s = len([w for w in words if is_token(w, 'S')])
    first_s_name = 'S' + str(num_s) if num_s > 1 else 'S'
    self.forms['F'] = Form({'case': first_s_name, 'num': first_s_name, 'gen': first_s_name})

  def inflect_its(self):
    words = self.t.split(' ')

    first_s = None
    for w in words:
      if is_token(w, 'S'):
        first_s = get_token_name(w)
        break

    if first_s is not None:
      self.forms['J'] = Form({'case': '', 'num': 'sg', 'gen': first_s})

  def inflect_which(self):
    self.t = self.t.replace('[', '')
    self.t = self.t.replace(']', '')
    words = self.t.split(' ')
    current = 1
    tokens = set()

    for i, w in enumerate(words):
      if w == '<W>':
        if current > 1:
          words[i] = '<W' + str(current) + '>'
        tokens.add(w)
        current += 1

    last_s = None       # name of S token
    last_noun = None    # literal noun
    for i, w in enumerate(words):
      if is_token(w, 'S'):
        last_s = get_token_name(w)
      elif w in ('przedmiot', 'rzecz', 'przedmioty', 'rzeczy'):
        last_noun = w
      elif is_token(w, 'W'):
        w_name = get_token_name(w)
        if last_s is not None:
          self.forms[w_name] = Form({'case': 'nom', 'num': last_s, 'gen': last_s})
        elif last_noun is not None:
          num = 'pl' if last_noun.endswith('y') else 'sg'
          gen = 'f' if last_noun == 'rzecz' else 'm'
          self.forms[w_name] = Form({'case': 'nom', 'num': num, 'gen': gen})

    self.t = ' '.join(words)
    return self.t

  def inflect_all(self):
    self.inflect_big()
    self.inflect_its()
    self.inflect_which()
    return self.t

class FormDetector:
  def __init__(self, t, forms):
    self.t = t
    self.forms = forms

  def add_plural(self):
    words = self.t.split(' ')
    for i, w in enumerate(words):
      if w.endswith('>s'):
        name = w[1:-2]
        self.forms[name].props['num'] = 'pl'
        words[i] = w[:-1]
      elif w.endswith('>s?'):
        name = w[1:-3]
        self.forms[name].props['num'] = 'pl'
        words[i] = w[:-2] + '?'

      if w in ('many', 'more', 'fewer', 'than'):
        for j, w2 in enumerate(words[i+1:]):
          if w2 == 'the':
            continue
          name = get_token_name(w2)
          if name is None:
            break
          if name.startswith('S') or name.startswith('O'):
            self.forms[name].props['num'] = 'pl'

    self.t = ' '.join(words)

  def add_gen(self):
    words = self.t.split(' ')
    feature_names = ('color', 'shape', 'material', 'size')

    for i, w in enumerate(words):
      if w in ('of', 'many', 'more', 'fewer', 'than') \
          or (w == 'as' and words[0] == 'Is' and (words[1] in feature_names or words[2] in feature_names))\
          or (w == 'and' and self.t.startswith('Are there the same number of')) \
          or (w == 'and' and self.t.startswith('Are there an equal number of')):

        for j, w2 in enumerate(words[i+1:]):
          if w2 in ('the', 'other'):
            continue
          name = get_token_name(w2)
          if name is None:
            break
          if name.startswith('S') or name.startswith('O'):
            self.forms[name].props['case'] = 'gen'

  def add_inst(self):
    words = self.t.split(' ')
    for i, w in enumerate(words):
      if w == 'are':
        for j, w2 in enumerate(words[i+1:]):
          name = get_token_name(w2)
          if name is None or name.startswith('R'):
            break
          if name.startswith('S') or name.startswith('O'):
            self.forms[name].props['case'] = 'inst'
            self.forms[name].props['num'] = 'pl'

  # noinspection PyMethodMayBeStatic
  def propagate_forms(self):
    for i in range(1, 5):
      stri = '' if i == 1 else str(i)
      s_name = 'S' + stri
      if s_name in self.forms:
        for name in ('Z', 'C', 'M'):
          self.forms[name + stri] = Form({'case': s_name,
                                          'num': s_name,
                                          'gen': s_name})

  def propagate_forms_r_s(self):
    words = self.t.split(' ')
    last_r = None
    for i, w in enumerate(words):
      if is_token(w, 'R'):
        last_r = get_token_name(w)
      elif last_r is not None and is_token(w, 'S'):
        s_name = get_token_name(w)
        self.forms[s_name] = Form({'case': last_r, 'num': 'sg', 'gen': ''})
      elif w not in ['the'] and not is_token(w):
        last_r = None

  def propagate_forms_s_o(self):
    words = self.t.split(' ')
    for i, w in enumerate(words):
      if is_token(w, 'S'):
        s_name = get_token_name(w)
        self.forms['O'] = Form({'case': s_name, 'num': s_name, 'gen': s_name})
        return

  def add_all(self):
    self.add_plural()
    self.add_gen()
    self.add_inst()
    self.propagate_forms()
    self.propagate_forms_r_s()
    self.propagate_forms_s_o()
    return self.t


parser = argparse.ArgumentParser()

parser.add_argument('--template_dir', default='CLEVR_1.0_templates',
    help="Directory containing JSON templates for questions")

parser.add_argument('--verbose', action='store_true',
    help="Print more verbose output")

def get_token_name(word):
  token = re.search(r'<(.*?)>', word)
  if token is None:
    return None
  name = token.group(1)
  return name

def is_token(w, family=''):
  return w.startswith('<' + family)

def add_forms(t, forms):
  for k, v in forms.items():
    t = re.sub(r'<%s>' % k, '<' + k + ':' + v.str() + '>', t)
  return t

def capitalise(t):
  if t[0].islower():
    return t[0].upper() + t[1:]
  return t

class Translator:
  def __init__(self):
    self.translations = self.build_translations()
    self.forms = {}

    self.all_trfrom = [tn[0] for tn in self.translations]
    self.all_trfrom = set(self.all_trfrom)
    self.used_trfrom = set()

  @staticmethod
  def build_translations():
    def build_feature_translations():
      res = []

      feature_translations = [
        (r'are the same <feature> as', r'ma ten sam <feature> co'),
        (r'does it have the same <feature> as', r'czy ma ten sam <feature>, co'),
        (r'is it the same <feature> as', r'czy ma ten sam <feature>, co'),
        (r'is its <feature> the same as', r'czy ma ten sam <feature>, co'),
        (r'(.*?) is what <feature>\?', r'Jakiego <feature>u jest \1?'),
        (r'(.*?) has what <feature>\?', r'Jaki <feature> ma \1?'),
        (r'is the same <feature>', r'jest tego samego <feature>u'),
        (r'has the same <feature>', r'ma ten sam <feature>'),
        (r'have same <feature>', r'mają ten sam <feature>'),
        (r'have the same <feature>', r'mają ten sam <feature>'),
        (r'of the same <feature> as the', r'tego samego <feature>u, co'),
        (r'that is the same <feature> as the', r'<W> jest tego samego <feature>u, co'),
        (r'the same <feature> as the', r'tego samego <feature>u, co'),
        (r'what is its <feature>', r'jaki jest <J:case=,num=sg,gen=S2> <feature>'),
        (r'what is the <feature> of the other', r'jaki jest <feature> innego'),
        (r'what is the <feature> of', r'jaki jest <feature>'),
        (r'what <feature> is it', r'jaki ma <feature>'),
        (r'what <feature> is', r'jakiego jest <feature>u'),
        (r'what <feature> is the other', r'jaki <feature> ma inny'),
        (r'Is the <feature> of', r'Czy <feature>'),

        (r'Is there any other thing that has the same <feature> as the (.*?)\?',
         r'Czy jest coś, co ma ten sam <feature>, co \1?'),
        (r'Do the (.*?) and the (.*?) have the same <feature>\?',
         r'Czy \1 i \2 mają ten sam <feature>?'),
        (r'Is the <feature> of the (.*?) the same as the <Z2> <C2> <M2> <S2>\?',
         r'Czy <feature> \1 jest taki sam, jak <Z2:case=gen,num=sg,gen=S2> <C2:case=gen,num=sg,gen=S2> <M2:case=gen,num=sg,gen=S2> <S2:case=gen,num=sg,gen=>?'),
      ]

      for f_en, f_pl in (('color', 'kolor'), ('size', 'rozmiar'), ('shape', 'kształt'), ('material', 'materiał')):
        for (trfrom, trto) in feature_translations:
          trfrom = trfrom.replace('<feature>', f_en)
          trto = trto.replace('<feature>', f_pl)
          res.append((trfrom, trto))

      return res

    translations_dict = {

      'compare_integer': [
        (r'\bare there the same number of', r'czy jest taka sama liczba'),
        (r'\bare there an equal number of', r'czy jest tyle samo'),
        (r'\bis the number of', r'czy liczba'),
        (r'\bthe same as the number of', r'jest taka sama jak'),
        (r'\bare there more', r'czy jest więcej'),
        (r'\bare there fewer', r'czy jest mniej'),
        (r'\bgreater', r'jest większa'),
        (r'\bless', r'jest mniejsza'),
        (r'\bmore', r'więcej'),
        (r'\bfewer', r'mniej'),
        (r'\bthan', r'niż'),
        (r'\bis the number of (.*?) the same as the number of (.*?)\?', r'czy liczba \1 jest taka sama jak liczba \2?')
      ],
      'comparison': [
        (r'Is the (.*?) the same size as the (.*?)\?',
         r'Czy \1 jest tak samo <F>, jak \2?'),
        (r'Are the (.*?) and the (.*?) made of the same material\?',
         r'Czy \1 i \2 są [zrobione] z tego samego materiału?'),
        (r'Is the (.*?) made of the same material as the (.*?)\?',
         r'Czy \1 i jest z tego samego materiału, co \2?'),
        (r'made of the same material as the',
         r'z tego samego materiału, co'),
      ],
      'one_hop': [
        (r'There is a (.*?); what number of (.*?) are (.*?) it\?',
         r'Na obrazku jest \1. Ile \2 jest \3 <I:case=R,num=sg,gen=S>?'),
        (r'There is a (.*?); how many (.*?) are (.*?) it\?',
         r'Na obrazku jest \1. Ile \2 jest \3 <I:case=R,num=sg,gen=S>?'),
        (r'What number of (.*?) are (.*?) the (.*?)\?', r'Ile \1 jest \2 \3?'),
        (r'How many (.*?) are (.*?) the (.*?)\?', r'Ile \1 jest \2 \3?'),
        (r'; what is it made of\?', '. Z czego jest <D:case=nom,num=sg,gen=S>?'),
        (r'; what material is it made of\?', '. Z jakiego materiału jest <D:case=nom,num=sg,gen=S>?'),
      ],
      'same_relate': [
        (r'^How many other\b', r'Ile innych'),
        (r'^What number of other\b', r'Ile innych'),
        (r'How many other (.*?) have the same', r'Ile innych \1 ma ten sam'),
        (r'How many other (.*?) are(?: there of)? the same', r'Ile innych \1 ma ten sam'),
        (r'What number of other (.*?) have the same', r'Ile innych \1 ma ten sam'),
        (r'What number of other (.*?) are(?: there of)? the same', r'Ile innych \1 ma ten sam')
      ],
      'single_and': [
        (r'\bboth\b', '')
      ],
      'single_or': [
        (r'\beither\b', 'albo'),
        (r'\bor\b', 'albo'),
        (r'\bhow many (.*?) things are', r'ile \1 rzeczy jest'),
        (r'\bwhat number of (.*?) things are', r'ile \1 rzeczy jest'),
        (r'\bhow many (.*?) objects are', r'ile \1 obiektów jest'),
        (r'\bwhat number of (.*?) objects are', r'ile \1 obiektów jest'),
        (r'\bhow many (.*?) are', r'ile \1 jest'),
        (r'\bwhat number of (.*?) are', r'ile \1 jest')
      ],
      # 'three_hop': [
      # ],
      'two_hop': [
        (r'\bthere is a\b', r'na obrazku jest'),
      ],
      'zero_hop': [
        (r'are there any other', 'czy są jakieś inne'),
        (r'are there any', 'czy są jakieś'),
        (r'how many', 'ile'),
        (r'what number of', 'ile'),
        (r'is there a\b', 'czy [na obrazku] jest '),
        (r'How many (.*?) are there\?', r'Ile jest \1?'),
        (r'What number of (.*?) are there\?', r'Ile jest \1?'),
        (r'Are any (.*?) visible\?', r'Czy widać jakieś \1?'),
      ],

      'other': [
        (r'same as', 'jest taki sam jak'),
        (r'does the', 'czy'),
        (r'does it', 'czy'),
        (r'do the', 'czy'),
        (r'are there', 'jest'),

        (r'What is (.*?) made of\?', r'Z czego jest \1?'),

        (r'\bare [made of] same material as\b',
         'jest z tego samego materiału, co'),     # todo
        (r'\bthat is [made of] same material as\b',
         '<W> jest z tego samego materiału, co'),   # todo
        (r'\bthat is [made of] the same material as\b', '<W> jest [zrobiony] z tego samego materiału, co'),   # !

        (r'\bthat is same color as the', '<W> jest tego samego koloru, co'),   # !
        (r'\bthat is the same color as the', '<W> jest tego samego koloru, co'),   # !

        (r'\bare there any other things', 'czy jest coś'),
        (r'\bis there another', 'czy jest <O>'),
        (r'\bis there anything else that', 'czy jest coś, co'),
        (r'\bis there any other thing that', 'czy jest coś, co'),
        (r'\bare there any other', 'czy są jakieś inne'),
        (r'\bof the other', 'innego'),

        (r'\bthere is another', 'na obrazku jest drugi'),

        (r'^How many objects', 'ile przedmiotów'),
        (r'^What number of objects', 'ile przedmiotów'),
        (r'^How many things', 'ile rzeczy'),
        (r'^What number of things', 'ile rzeczy'),

        (r'^How many other objects', 'ile innych przedmiotów'),
        (r'^What number of other objects', 'ile innych przedmiotów'),
        (r'^How many other things', 'ile innych rzeczy'),
        (r'^What number of other things', 'ile innych rzeczy'),

        (r'\bthat is', '<W> jest'),
        (r'\bthat are', '<W> są'),

        (r'\bhow big is it', '<H:case=F,num=F,gen=F> jest <F>'),
        (r'\bhow big is', 'jak <F> jest'),

        # first words
        (r'^Is', 'Czy'),
        (r'^Are', 'Czy'),

        # last words
        (r'it\?', '?')
      ]
    }

    translations_dict['other'].extend(build_feature_translations())

    translations2 = [
      (r'\bnumber\b', 'liczba'),
      (r'\bthe\b', ' '),
      (r'\bof\b', ' '),
      (r'\band\b', 'i'),
      (r'\banother\b', '<O>'),
      (r'\bother\b', '<O>'),
      (r'\bthings\b', 'rzeczy'),
      (r'\bobjects\b', 'rzeczy'),
      (r'\bsize\b', 'rozmiar'),
      (r'\bcolor\b', 'kolor'),
      (r'\bshape\b', 'ksztalt'),
      (r'\bmaterial\b', 'materiał'),
      (r'\bas\b', 'co'),
      (r'\bis\b', 'jest'),
      (r'\bthat\b', '<W>'),
    ]

    translations = itertools.chain.from_iterable(translations_dict.values())

    translations = sorted(translations, key=lambda t: len(t[0]), reverse=True)
    translations.extend(translations2)

    return translations

  def initialise_forms(self, t):
    self.forms = {}

    tokens = re.findall(r'<(.*?)>', t)
    for token in tokens:
      self.forms[token] = Form({'case': '' if token.startswith('R') else 'nom',
                           'num': 'sg',
                           'gen': ''})

  def translate(self, t):
    self.initialise_forms(t)

    t = FormDetector(t, self.forms).add_all()
    t = WordInflector(t, self.forms).inflect_all()

    for trfrom, trto in self.translations:
      if re.search(trfrom, t, flags=re.IGNORECASE):
        self.used_trfrom.add(trfrom)
      t = re.sub(trfrom, trto, t, flags=re.IGNORECASE)
      t = capitalise(t)

    t = WordInflector(t, self.forms).inflect_which()

    new_params = set()
    for param_prefix in param_types.keys():
      new_params.update(re.findall(r'<(%s\d*).*?>' % param_prefix, t))

    t = add_forms(t, self.forms)

    t = re.sub(r'^ ', '', t)
    t = re.sub(' {2}', ' ', t)
    t = re.sub(';', ' -', t)
    t = re.sub(r'^Jakiego materiału', 'Z jakiego materiału', t)
    t = re.sub(r'(?<!,) kt.r', ', któr', t)
    t = re.sub(r'(?<!,) niż\b', ', niż', t)
    t = re.sub(r'(?<!,) co\b', ', co', t)   # todo double comme

    return t, new_params


# noinspection PyShadowingNames
def main(args):
  templates = {}
  fname = '.json'

  for fn in os.listdir(args.template_dir):
    if not fn.endswith(fname): continue
    with open(os.path.join(args.template_dir, fn), 'r') as f:
      templates[fn] = []
      for i, template in enumerate(json.load(f)):
        templates[fn].append(template)

  translator = Translator()

  for fn, file_templates in templates.items():
    for ti, v in enumerate(file_templates):
      texts = []
      new_params_all = set()
      for i, t in enumerate(v['text']):

        translated, new_params = translator.translate(t)
        texts.append(translated)
        new_params_all.update(new_params)

      for param in new_params_all:
        file_templates[ti]['params'].append({'type': (param_types[param[0]]),
                                             'name': '<' + param + '>'})
      file_templates[ti]['text'] = texts

    with open(os.path.join(args.template_dir + '_pl', fn), 'w') as fout:
      fout.write(json.dumps(file_templates))

  print('Not used:')
  print('\n'.join(translator.all_trfrom.difference(translator.used_trfrom)))


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

