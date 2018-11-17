# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import print_function
import argparse, json, os
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

parser = argparse.ArgumentParser()

parser.add_argument('--template_dir', default='CLEVR_1.0_templates',
    help="Directory containing JSON templates for questions")

parser.add_argument('--verbose', action='store_true',
    help="Print more verbose output")

def add_plural(t, forms):
  words = t.split(' ')
  for i in range(len(words)):
    if words[i].endswith('>s'):
      name = words[i][1:-2]
      forms[name].props['num'] = 'pl'
      words[i] = words[i][:-1]

  return ' '.join(words)

def add_gen(t, forms):
  words = t.split(' ')
  for i in range(len(words)):
    if words[i] in ('of', 'many'):
      for j in range(i+1, len(words)):
        if words[j] == 'the':
          continue
        token = re.search(r'<(.*?)>', words[j])
        if token is None:
          break
        name = token.group(1)
        if not name.startswith('S'):
          continue
        forms[name].props['case'] = 'gen'

  return t

def declinate_big(t, forms):
  words = t.split(' ')
  num_s = len([w for w in words if w.startswith('<S')])
  first_s_name = 'S' + str(num_s) if num_s > 1 else 'S'
  forms['F'] = Form('case=' + first_s_name + ',num=' + first_s_name + ',gen=' + first_s_name)

def uzgodnij(forms):
  for i in range(1, 5):
    stri = '' if i == 1 else str(i)
    if 'S' + stri in forms:
      for name in ('Z', 'C', 'M'):
        forms[name + stri] = Form('case=S' + stri + ',num=S' + stri + ',gen=S' + stri)
    if 'R' + stri in forms:
      forms['S' + stri].props['case'] = 'R' + stri

def add_forms(t, forms):
  for k, v in forms.items():
    t = re.sub(r'<%s:?([^:]*?)>' % k, '<' + k + ':' + v.str() + '>', t)
  return t


def tr(t):
  translations_mutable = [
    ('are the same <feature> as', 'ma ten sam <feature> co'),
    ('does it have the same <feature> as', 'czy ma ten sam <feature>, co'),
    ('is it the same <feature> as', 'czy ma ten sam <feature>, co'),
    ('is its <feature> the same as', 'czy ma ten sam <feature>, co'),
    ('is what <feature>', 'jest jakiego <feature>u'),
    ('has what <feature>', 'ma jaki <feature>'),
    ('have same <feature> as', 'ma ten sam <feature> co'),
    ('have the same <feature> as', 'ma ten sam <feature> co'),
    ('have the same <feature>', 'mają ten sam <feature>'),
    ('of the same <feature> as the', 'tego samego <feature>u, co'),
    ('that is the same <feature> as the', 'który jest tego samego <feature>u, co'),
    ('what is its <feature>', 'jaki jest jego <feature>'),
    ('what is the <feature> of the other', 'jaki jest <feature> innego'),
    ('what is the <feature> of the', 'jaki jest <feature>'),
    ('what number of other objects are there of the same <feature> as the', 'ile przedmiotów ma ten sam <feature>, co'),
    ('what <feature> is it', 'jaki ma <feature>'),
    ('what <feature> is the other', 'jaki <feature> ma inny'),
    ('is the <feature> of', 'czy <feature>'),
  ]
  
  translations = [
    ('same as', 'taka sama jak'),                                # !
    ('does the', 'czy'),                                # wymuszaja liczbe
    ('does it', 'czy'),                                # wymuszaja liczbe
    ('do the', 'czy'),

    ('are [made of] same materiał as', 'jest [zrobione] z tego samego materialu, co'),   # !
    ('that is [made of] same material as', 'który jest [zrobiony] z tego samego materialu, co'),   # !
    ('that is [made of] the same material as', 'który jest [zrobiony] z tego samego materialu, co'),   # !

    ('that is same color as the', 'który jest tego samego koloru, co'),   # !
    ('that is the same color as the', 'który jest tego samego koloru, co'),   # !

    ('are there any other things', 'czy jest coś'),
    ('is there another', 'czy jest inny'),
    ('is there anything else', 'czy jest coś'),
    ('is there any other thing', 'czy jest coś'),
    ('are there any other', 'czy są jakieś inne'),
    ('of the other', 'innego'),

    ('there is another', 'na obrazku jest drugi'),
    ('there is a ', 'na obrazku jest '),
    ('is there a ', 'czy [na obrazku] jest '),
    ('are there any', 'czy [na obrazku] są jakieś'),

    ('how many objects', 'ile przedmiotów'),
    ('how many objects', 'ile innych rzeczy'),
    ('what number of objects', 'ile przedmiotów'),
    ('what number of other objects', 'ile innyh rzeczy'),
    ('how many other things', 'ile rzeczy'),
    ('how many other objects', 'ile innych przedmiotów'),
    ('how many other', 'ile innych'),
    ('what number of other', 'ile innych'),
    ('how many', 'ile'),
    ('are there the same number of', 'czy jest tyle samo'),
    ('are there an equal number of', 'czy jest tyle samo'),
    ('is the number of', 'czy liczba'),
    ('the same as the number of', 'taka sama jak'),
    ('are there more', 'czy jest więcej'),
    ('are there fewer', 'czy jest mniej'),
    ('greater', 'jest większa'),                             # !
    ('less', 'jest mniejsza'),                             # !
    ('more', 'więcej'),
    ('fewer', 'mniej'),
    ('than', 'niż'),

    ('that is', 'który/a jest'),                        # !
    ('that are', 'które są'),

    ('how big is it', 'jaki jest <F>'),
    ('how big is', 'jak <F> jest'),
  ]

  translations2 = [
      ('number', 'liczba'),
      ('the ', ' '),
      ('of ', ' '),
      ('and', 'i'),
      ('either', 'albo'),
      ('another', 'inny'),
      ('other', 'inny'),
      ('things', 'rzeczy'),
      ('size', 'rozmiar'),
      ('color', 'kolor'),
      ('shape', 'kształt'),
      ('material', 'materiał'),
  ]

  forms = {}
  tokens = re.findall(r'<(.*?)>', t)
  for token in tokens:
    forms[token] = Form('case=nom,num=sg,gen=')

  # for k, v in forms.items():
  #   print(k, v.str())

  t = add_plural(t, forms)
  t = add_gen(t, forms)
  uzgodnij(forms)

  # t = add_forms(t, forms)

  for f_en, f_pl in (('color', 'kolor'), ('size', 'rozmiar'), ('shape', 'ksztalt'), ('material', 'material')):
    for (trfrom, trto) in translations_mutable:
      trfrom = trfrom.replace('<feature>', f_en)
      trto = trto.replace('<feature>', f_pl)
      translations.append((trfrom, trto))

  for i, (trfrom, trto) in enumerate(list(translations)):
    if len(trto) == 0:
      continue
    translations.append((trfrom[0].upper() + trfrom[1:], trto[0].upper() + trto[1:]))
  for i, (trfrom, trto) in enumerate(list(translations2)):
    if len(trto) == 0:
      continue
    translations2.append((trfrom[0].upper() + trfrom[1:], trto[0].upper() + trto[1:]))

  translations = sorted(translations, key=lambda q: len(t[0]), reverse=True)

  for trfrom, trto in translations:
    t = re.sub(trfrom, trto, t)
  for trfrom, trto in translations2:
    t = re.sub(trfrom, trto, t)

  declinate_big(t, forms)
  t = add_forms(t, forms)

  return t


def main(args):
  num_loaded_templates = 0
  templates = {}
  for fn in os.listdir(args.template_dir):
    if not fn.endswith('zero_hop.json'): continue
    with open(os.path.join(args.template_dir, fn), 'r') as f:
      base = os.path.splitext(fn)[0]
      for i, template in enumerate(json.load(f)):
        num_loaded_templates += 1
        key = (fn, i)
        templates[key] = template

  texts = []
  for k, v in templates.items():
    for i, t in enumerate(v['text']):
      t = tr(t)

  with open(os.path.join(args.template_dir + '_pl', 't2.json'), 'w') as fout:
    fout.write(json.dumps(templates))


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

