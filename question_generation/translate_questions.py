# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import print_function
import argparse, json, os
import itertools
import re

class Form:
  def __init__(self, s):
    self.props = {}
    for prop in s.split(','):
      kv = prop.split('=')
      if len(kv) == 1:
        kv.append('')
      k, v = kv
      self.props[k] = v

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

class NounInflector:
  def __init__(self, t):
    self.t = t

  def inflect_big(self, forms):
    words = self.t.split(' ')

    num_s = len([w for w in words if w.startswith('<S')])
    first_s_name = 'S' + str(num_s) if num_s > 1 else 'S'
    forms['F'] = Form('case=' + first_s_name + ',num=' + first_s_name + ',gen=' + first_s_name)

  def inflect_its(self, forms):
    words = self.t.split(' ')

    first_s = None
    for w in words:
      if w.startswith('<S'):
        first_s = get_token_name(w)
        break

    if first_s is not None:
      forms['J'] = Form('case=,num=sg,gen=' + first_s)

  def inflect_which(self, forms):
    self.t = self.t.replace('[', '')
    self.t = self.t.replace(']', '')
    words = self.t.split(' ')
    current = 1
    tokens = set()

    for i in range(len(words)):
      if words[i] == '<W>':
        if current > 1:
          words[i] = '<W' + str(current) + '>'
        tokens.add(words[i])
        current += 1

    last_s = None
    for i in range(len(words)):
      if words[i].startswith('<S'):
        last_s = get_token_name(words[i])
      elif last_s is not None and words[i].startswith('<W'):
        w_name = get_token_name(words[i])
        forms[w_name] = Form('case=nom,num=' + last_s + ',gen=' + last_s)

    self.t = ' '.join(words)

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

def add_plural(t, forms):
  words = t.split(' ')
  for i in range(len(words)):
    if words[i].endswith('>s'):
      name = words[i][1:-2]
      forms[name].props['num'] = 'pl'
      words[i] = words[i][:-1]
    elif words[i].endswith('>s?'):       # todo
      name = words[i][1:-3]
      forms[name].props['num'] = 'pl'
      words[i] = words[i][:-2] + '?'

    if words[i] in ('many', 'more', 'fewer', 'than'):
      for j in range(i+1, len(words)):
        if words[j] == 'the':
          continue
        token = re.search(r'<(.*?)>', words[j])
        if token is None:
          break
        name = token.group(1)
        if not name.startswith('S'):
          continue
        forms[name].props['num'] = 'pl'

  return ' '.join(words)

def add_gen(t, forms):
  words = t.split(' ')
  for i in range(len(words)):
    if words[i] in ('of', 'many', 'more', 'fewer', 'than') or re.match(r'<R\d*>', words[i]):
      for j in range(i+1, len(words)):
        if words[j] == 'the':
          continue
        name = get_token_name(words[j])
        if name is None:
          break
        if not name.startswith('S'):
          continue
        forms[name].props['case'] = 'gen'

  return t

def add_inst(t, forms):
  words = t.split(' ')
  for i in range(len(words)):
    if words[i] == 'are':
      for j in range(i+1, len(words)):
        name = get_token_name(words[j])
        if name is not None and name.startswith('R'):
          break

        if name is None or not name.startswith('S'):
          continue

        forms[name].props['case'] = 'inst'
        forms[name].props['num'] = 'pl'

  return t

def handle_noun_cases(t, forms):
  t = add_gen(t, forms)
  t = add_inst(t, forms)
  return t

def propagate_forms(forms):
  for i in range(1, 5):
    stri = '' if i == 1 else str(i)
    if 'S' + stri in forms:
      for name in ('Z', 'C', 'M'):
        forms[name + stri] = Form('case=S' + stri + ',num=S' + stri + ',gen=S' + stri)

def propagate_forms_r_s(t, forms):
  words = t.split(' ')
  last_r = None
  for i in range(len(words)):
    if words[i].startswith('<R'):
      last_r = get_token_name(words[i])
    elif last_r is not None and words[i].startswith('<S'):
      s_name = get_token_name(words[i])
      forms[s_name] = Form('case=' + last_r + ',num=sg,gen=')
    elif words[i] not in ['the'] and not words[i].startswith('<'):
      last_r = None

def add_forms(t, forms):
  for k, v in forms.items():
    t = re.sub(r'<%s>' % k, '<' + k + ':' + v.str() + '>', t)
  return t


def build_translations():

  def build_feature_translations():
    res = []

    feature_translations = [
      ('are the same <feature> as', 'ma ten sam <feature> co'),
      ('does it have the same <feature> as', 'czy ma ten sam <feature>, co'),
      ('is it the same <feature> as', 'czy ma ten sam <feature>, co'),
      ('is its <feature> the same as', 'czy ma ten sam <feature>, co'),
      ('is what <feature>', 'jest jakiego <feature>u'),
      ('has what <feature>', 'ma jaki <feature>'),
      ('have same <feature> as', 'ma ten sam <feature> co'),
      ('have the same <feature> as', 'ma ten sam <feature> co'),
      ('have the same <feature>', 'maja ten sam <feature>'),
      ('of the same <feature> as the', 'tego samego <feature>u, co'),
      ('that is the same <feature> as the', '<W> jest tego samego <feature>u, co'),
      ('the same <feature> as the', 'tego samego <feature>u, co'),
      ('what is its <feature>', 'jaki jest <J:case=,num=sg,gen=S2> <feature>'),
      ('what is the <feature> of the other', 'jaki jest <feature> innego'),
      ('what is the <feature> of', 'jaki jest <feature>'),
      ('what number of other objects are there of the same <feature> as the', 'ile przedmiotow ma ten sam <feature>, co'),
      ('what <feature> is it', 'jaki ma <feature>'),
      ('what <feature> is', 'jakiego <feature>u jest'),
      ('what <feature> is the other', 'jaki <feature> ma inny'),
      (r'Is the <feature> of', 'Czy <feature>'),
    ]

    for f_en, f_pl in (('color', 'kolor'), ('size', 'rozmiar'), ('shape', 'kształt'), ('material', 'materiał')):
      for (trfrom, trto) in feature_translations:
        trfrom = trfrom.replace('<feature>', f_en)
        trto = trto.replace('<feature>', f_pl)
        res.append((trfrom, trto))

    return res

  translations_dict = {

    'compare_integer': [
      ('are there the same number of', 'czy jest taka sama liczba'),
      ('are there an equal number of', 'czy jest tyle samo'),
      ('is the number of', 'czy liczba'),
      ('the same as the number of', 'taka sama jak'),
      ('are there more', 'czy jest więcej'),
      ('are there fewer', 'czy jest mniej'),
      ('greater', 'jest większa'),
      ('less', 'jest mniejsza'),
      ('more', 'więcej'),
      ('fewer', 'mniej'),
      ('than', 'niż'),
      (r'is the number of (.*?) the same as the number of (.*?)?', r'czy liczba \1 jest taka sama jak liczba \2?')
    ],
    'comparison': [
      (r'Is the (.*?) the same size as the (.*?)?', r'Czy \1 jest tak samo duzy/duza, jak \2?'),
      (r'Are the (.*?) and the (.*?) made of the same material?', r'Czy \1 i \2 są [zrobione] z tego samego materiału?'),
      (r'Is the (.*?) made of the same material as the (.*?)?', r'Czy \1 i jest z tego samego materiału, co \2?'),
      ('made of the same material as the', 'z tego samego materiału, co'),
    ],
    'one_hop': [
      (r'There is a (.*?); what number of (.*?) are (.*?) it?',
       r'Na obrazku jest \1. Ile \2 jest \3 <I:case=R,num=sg,gen=S>?'),
      (r'There is a (.*?); how many (.*?) are (.*?) it?',
       r'Na obrazku jest \1. Ile \2 jest \3 niego/nim/nia/niej?'),
      (r'What number of (.*?) are (.*?) the (.*?)\?', r'Ile \1 jest \2 \3?'),
      (r'How many (.*?) are (.*?) the (.*?)\?', r'Ile \1 jest \2 \3?'),
      ('; what is it made of?', '. Z czego jest zrobiony/a?'),
      ('; what material is it made of?', '. Z jakiego materiału jest zrobiony/a?'),
    ],
    'same_relate': [
    ],
    'single_and': [
      (r'\bboth\b', '')
    ],
    'single_or': [
      (r'\beither\b', 'albo'),
      (r'\bor\b', 'albo'),
      (r'how many (.*?) things are', r'ile \1 rzeczy jest'),
      (r'what number of (.*?) things are', r'ile \1 rzeczy jest'),
      (r'how many (.*?) objects are', r'ile \1 obiektów jest'),
      (r'what number of (.*?) objects are', r'ile \1 obiektów jest'),
      (r'how many (.*?) are', r'ile \1 jest'),
      (r'what number of (.*?) are', r'ile \1 jest')
    ],
    'three_hop': [
    ],
    'two_hop': [
      ('there is a ', 'na obrazku jest '),
    ],
    'zero_hop': [
      ('are there any', 'czy są jakieś'),
      ('how many', 'ile'),
      ('what number of', 'ile'),
      ('is there a ', 'czy [na obrazku] jest '),
      (r'How many (.*?) are there\?', r'Ile jest \1?'),
      (r'What number of (.*?) are there\?', r'Ile jest \1?'),
      (r'Are any (.*?) visible\?', r'Czy widać jakieś \1?'),
    ],

    'other': [
    ('same as', 'jest taki sam jak'),                                # !
    ('does the', 'czy'),                                # wymuszaja liczbe
    ('does it', 'czy'),                                # wymuszaja liczbe
    ('do the', 'czy'),

    (r'What is (.*?) made of\?', r'Z czego jest \1?'),

    ('are [made of] same material as', 'jest [zrobione] z tego samego materiału, co'),   # !
    ('that is [made of] same material as', '<W> jest [zrobiony] z tego samego materiału, co'),   # !
    ('that is [made of] the same material as', '<W> jest [zrobiony] z tego samego materiału, co'),   # !

    ('that is same color as the', '<W> jest tego samego koloru, co'),   # !
    ('that is the same color as the', '<W> jest tego samego koloru, co'),   # !

    ('are there any other things', 'czy jest coś'),
    ('is there another', 'czy jest inny'),
    ('is there anything else', 'czy jest coś'),
    ('is there any other thing', 'czy jest coś'),
    ('are there any other', 'czy są jakieo inne'),
    ('of the other', 'innego'),

    ('there is another', 'na obrazku jest drugi'),

    ('how many objects', 'ile przedmiotow'),
    ('how many objects', 'ile innych rzeczy'),
    ('what number of objects', 'ile przedmiotow'),
    ('what number of other objects', 'ile innych rzeczy'),
    ('how many other things', 'ile rzeczy'),
    ('how many other objects', 'ile innych przedmiotow'),
    ('how many other', 'ile innych'),
    ('what number of other', 'ile innych'),

    ('that is', '<W> jest'),                        # !
    ('that are', '<W> sa'),

    ('how big is it', '<H:case=F,num=F,gen=F> jest <F>'),
    ('how big is', 'jak <F> jest'),

    # first words
    ('Is', 'Czy'),
    ('Are', 'Czy'),
  ]
  }

  translations_dict['other'].extend(build_feature_translations())

  translations2 = [
    ('number', 'liczba'),
    ('the ', ' '),
    ('of ', ' '),
    ('and', 'i'),
    ('another', 'inny'),
    ('other', 'inny'),
    ('things', 'rzeczy'),
    ('objects', 'rzeczy'),
    ('size', 'rozmiar'),
    ('color', 'kolor'),
    ('shape', 'ksztalt'),
    ('material', 'materiał'),
  ]

  translations = itertools.chain(translations_dict.values())

  translations = sorted(translations, key=lambda q: len(q[0]), reverse=True)
  translations.extend(translations2)

  return translations


def translate(t):
  translations = build_translations()

  forms = {}
  tokens = re.findall(r'<(.*?)>', t)
  for token in tokens:
    if token.startswith('R'):
      forms[token] = Form('case=,num=sg,gen=')
    else:
      forms[token] = Form('case=nom,num=sg,gen=')


  t = add_plural(t, forms)
  t = handle_noun_cases(t, forms)
  propagate_forms(forms)
  propagate_forms_r_s(t, forms)

  inflector = NounInflector(t)
  inflector.inflect_big(forms)
  inflector.inflect_its(forms)
  inflector.inflect_which(forms)
  t = inflector.t

  for tns in translations:
    for trfrom, trto in tns:
      t = re.sub(trfrom, trto, t, flags=re.IGNORECASE)
      if t[0].islower():
        t = t[0].upper() + t[1:]

  new_params = []
  for param_prefix in ('W', 'F', 'I', 'J', 'H'):
    new_params.extend(re.findall(r'<(%s\d*).*?>' % param_prefix, t))
  print(t)
  print(new_params)
  print()

  t = add_forms(t, forms)

  t = re.sub(r'^ ', '', t)
  t = re.sub(' {2}', ' ', t)
  t = re.sub(r'^Jakiego materiału', 'Z jakiego materiału', t)

  return t, new_params


def main(args):
  num_loaded_templates = 0
  templates = {}
  fn = 'two_hop.json'
  for fn in os.listdir(args.template_dir):
    if not fn.endswith(fn): continue
    with open(os.path.join(args.template_dir, fn), 'r') as f:
      for i, template in enumerate(json.load(f)):
        num_loaded_templates += 1
        key = (fn, i)
        templates[key] = template

  for k, v in templates.items():
    texts = []
    new_params_all = set()
    for i, t in enumerate(v['text']):
      translated, new_params = translate(t)
      texts.append(translated)
      new_params_all.update(new_params)

    param_types = {'W': 'Which', 'F': 'Big', 'I': 'It', 'J': 'Its', 'H': 'What_like'}
    for param in new_params_all:
      templates[k]['params'].append({'type': (param_types[param[0]]), 'name': '<' + param + '>'})
    templates[k]['text'] = texts

  with open(os.path.join(args.template_dir + '_pl', fn), 'w') as fout:
    fout.write(json.dumps(list(templates.values())))


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

