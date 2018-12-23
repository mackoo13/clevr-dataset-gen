from __future__ import print_function
import argparse, json, os
import itertools
import re

param_types = {'W': 'Which', 'F': 'Big', 'I': 'It', 'J': 'Its', 'H': 'What_like', 'D': 'Made'}

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

class WordInflector:
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

  def inflect_all(self, forms):
    self.inflect_big(forms)
    self.inflect_its(forms)
    self.inflect_which(forms)
    return self.t

class FormDetector:
  def __init__(self, t):
    self.t = t

  def add_plural(self, forms):
    words = self.t.split(' ')
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

    self.t = ' '.join(words)

  def add_gen(self, forms):
    words = self.t.split(' ')
    for i in range(len(words)):
      if words[i] in ('of', 'many', 'more', 'fewer', 'than') or re.match(r'<R\d*>', words[i]):
        for j in range(i+1, len(words)):
          if words[j] in ('the', 'other'):
            continue
          name = get_token_name(words[j])
          if name is None:
            break
          if not name.startswith('S'):
            continue
          forms[name].props['case'] = 'gen'

    feature_names = ('color', 'shape', 'material')
    if words[0] == 'Is' and words[1] in feature_names or words[2] in feature_names:
      pass # todo

  def add_inst(self, forms):
    words = self.t.split(' ')
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

  def propagate_forms(self, forms):
    for i in range(1, 5):
      stri = '' if i == 1 else str(i)
      if 'S' + stri in forms:
        for name in ('Z', 'C', 'M'):
          forms[name + stri] = Form('case=S' + stri + ',num=S' + stri + ',gen=S' + stri)

  def propagate_forms_r_s(self, forms):
    words = self.t.split(' ')
    last_r = None
    for i in range(len(words)):
      if words[i].startswith('<R'):
        last_r = get_token_name(words[i])
      elif last_r is not None and words[i].startswith('<S'):
        s_name = get_token_name(words[i])
        forms[s_name] = Form('case=' + last_r + ',num=sg,gen=')
      elif words[i] not in ['the'] and not words[i].startswith('<'):
        last_r = None

  def add_all(self, forms):
    self.add_plural(forms)
    self.add_gen(forms)
    self.add_inst(forms)
    self.propagate_forms(forms)
    self.propagate_forms_r_s(forms)
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

def add_forms(t, forms):
  for k, v in forms.items():
    t = re.sub(r'<%s>' % k, '<' + k + ':' + v.str() + '>', t)
  return t


def build_translations():

  def build_feature_translations():
    res = []

    feature_translations = [
      (r'are the same <feature> as', r'ma ten sam <feature> co'),
      (r'does it have the same <feature> as', r'czy ma ten sam <feature>, co'),
      (r'is it the same <feature> as', r'czy ma ten sam <feature>, co'),
      (r'is its <feature> the same as', r'czy ma ten sam <feature>, co'),
      (r'is what <feature>', r'jest jakiego <feature>u'),
      (r'has what <feature>', r'ma jaki <feature>'),
      (r'has the same <feature>', r'ma ten sam <feature>'),
      (r'have same <feature>\?', r'mają ten sam <feature>?'),
      (r'have the same <feature>\?', r'mają ten sam <feature>?'),
      (r'of the same <feature> as the', r'tego samego <feature>u, co'),
      (r'that is the same <feature> as the', r'<W> jest tego samego <feature>u, co'),
      (r'the same <feature> as the', r'tego samego <feature>u, co'),
      (r'what is its <feature>', r'jaki jest <J:case=,num=sg,gen=S2> <feature>'),
      (r'what is the <feature> of the other', r'jaki jest <feature> innego'),
      (r'what is the <feature> of', r'jaki jest <feature>'),
      (r'what number of other objects are there of the same <feature> as the', 
       r'ile przedmiotow ma ten sam <feature>, co'),
      (r'what <feature> is it', r'jaki ma <feature>'),
      (r'what <feature> is', r'jakiego <feature>u jest'),
      (r'what <feature> is the other', r'jaki <feature> ma inny'),
      (r'Is the <feature> of', r'Czy <feature>'),

      (r'Is there any other thing that has the same <feature> as the (.*?)\?',
       r'Czy jest coś, o ma ten sam <feature>, co \1?'),
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
      (r'\bthe same as the number of', r'taka sama jak'),
      (r'\bare there more', r'czy jest więcej'),
      (r'\bare there fewer', r'czy jest mniej'),
      (r'\bgreater', r'jest większa'),
      (r'\bless', r'jest mniejsza'),
      (r'\bmore', r'więcej'),
      (r'\bfewer', r'mniej'),
      (r'\bthan', r'niż'),
      (r'\bis the number of (.*?) the same as the number of (.*?)?', r'czy liczba \1 jest taka sama jak liczba \2?')
    ],
    'comparison': [
      (r'Is the (.*?) the same size as the (.*?)?',
       r'Czy \1 jest tak samo <F>, jak \2?'),
      (r'Are the (.*?) and the (.*?) made of the same material?',
       r'Czy \1 i \2 są [zrobione] z tego samego materiału?'),
      (r'Is the (.*?) made of the same material as the (.*?)?',
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
      (r'How many (<.*?) are (.*?) the (.*?)\?', r'Ile \1 jest \2 \3?'),
      (r'; what is it made of\?', '. Z czego jest <D:case=nom,num=sg,gen=S>?'),
      (r'; what material is it made of\?', '. Z jakiego materiału jest <D:case=nom,num=sg,gen=S>?'),
    ],
    'same_relate': [
      (r'^How many other\b', r'Ile innych'),
      (r'^What number of other\b', r'Ile innych')
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
      (r'\bhow many (<.*?) are', r'ile \1 jest'),
      (r'\bwhat number of (.*?) are', r'ile \1 jest')
    ],
    # 'three_hop': [
    # ],
    'two_hop': [
      (r'\bthere is a\b', r'na obrazku jest'),
    ],
    'zero_hop': [
      (r'\bare there any\b', 'czy są jakieś'),
      (r'\bhow many\b', 'ile'),
      (r'\bwhat number of\b', 'ile'),
      (r'\bis there a\b', 'czy [na obrazku] jest '),
      (r'How many (.*?) are there\?', r'Ile jest \1?'),
      (r'What number of (.*?) are there\?', r'Ile jest \1?'),
      (r'Are any (.*?) visible\?', r'Czy widać jakieś \1?'),
    ],

    'other': [
      (r'\bsame as\b', 'jest taki sam jak'),
      (r'\bdoes the\b', 'czy'),
      (r'\bdoes it\b', 'czy'),
      (r'\bdo the\b', 'czy'),

      (r'What is (.*?) made of\?', r'Z czego jest \1?'),

      (r'\bare [made of] same material as\b',
       'jest z tego samego materiału, co'),     # todo
      (r'\bthat is [made of] same material as\b',
       '<W> jest z tego samego materiału, co'),   # todo
      (r'\bthat is [made of] the same material as\b', '<W> jest [zrobiony] z tego samego materiału, co'),   # !

      (r'\bthat is same color as the', '<W> jest tego samego koloru, co'),   # !
      (r'\bthat is the same color as the', '<W> jest tego samego koloru, co'),   # !

      (r'\bare there any other things', 'czy jest coś'),
      (r'\bis there another', 'czy jest inny'),
      (r'\bis there anything else', 'czy jest coś'),
      (r'\bis there any other thing', 'czy jest coś'),
      (r'\bare there any other', 'czy są jakieo inne'),
      (r'\bof the other', 'innego'),

      (r'\bthere is another', 'na obrazku jest drugi'),

      (r'^How many objects', 'ile przedmiotów'),
      (r'^What number of objects', 'ile przedmiotów'),
      (r'^What number of other objects', 'ile innych rzeczy'),
      (r'^How many other things', 'ile rzeczy'),
      (r'^How many other objects', 'ile innych przedmiotów'),
      (r'^How many other', 'ile innych'),
      (r'^What number of other', 'ile innych'),

      (r'\bthat is', '<W> jest'),
      (r'\bthat are', '<W> są'),

      (r'\bhow big is it', '<H:case=F,num=F,gen=F> jest <F>'),
      (r'\bhow big is', 'jak <F> jest'),

      # first words
      (r'^Is', 'Czy'),
      (r'^Are', 'Czy'),
    ]
  }

  translations_dict['other'].extend(build_feature_translations())

  translations2 = [[
    (r'\bnumber\b', 'liczba'),
    (r'\bthe\b', ' '),
    (r'\bof\b', ' '),
    (r'\band\b', 'i'),
    (r'\banother\b', 'inny'),
    (r'\bother\b', 'inny'),
    (r'\bthings\b', 'rzeczy'),
    (r'\bobjects\b', 'rzeczy'),
    (r'\bsize\b', 'rozmiar'),
    (r'\bcolor\b', 'kolor'),
    (r'\bshape\b', 'ksztalt'),
    (r'\bmaterial\b', 'materiał'),
    (r'\bas\b', 'co'),
  ]]

  translations = itertools.chain(translations_dict.values())

  translations = sorted(translations, key=lambda q: len(q[0]), reverse=True)
  translations.extend(translations2)

  return translations


def translate(t, translations):

  forms = {}
  tokens = re.findall(r'<(.*?)>', t)
  for token in tokens:
    if token.startswith('R'):
      forms[token] = Form('case=,num=sg,gen=')
    else:
      forms[token] = Form('case=nom,num=sg,gen=')

  t = FormDetector(t).add_all(forms)
  t = WordInflector(t).inflect_all(forms)

  for tns in translations:
    for trfrom, trto in tns:
      t = re.sub(trfrom, trto, t, flags=re.IGNORECASE)
      if t[0].islower():
        t = t[0].upper() + t[1:]

  new_params = set()
  for param_prefix in param_types.keys():
    new_params.update(re.findall(r'<(%s\d*).*?>' % param_prefix, t))

  t = add_forms(t, forms)

  t = re.sub(r'^ ', '', t)
  t = re.sub(' {2}', ' ', t)
  t = re.sub(r'^Jakiego materiału', 'Z jakiego materiału', t)

  return t, new_params


def main(args):
  templates = {}
  fname = '.json'

  translations = build_translations()

  for fn in os.listdir(args.template_dir):
    if not fn.endswith(fname): continue
    with open(os.path.join(args.template_dir, fn), 'r') as f:
      templates[fn] = []
      for i, template in enumerate(json.load(f)):
        templates[fn].append(template)

  for fn, file_templates in templates.items():
    for ti, v in enumerate(file_templates):
      texts = []
      new_params_all = set()
      for i, t in enumerate(v['text']):
        translated, new_params = translate(t, translations)
        texts.append(translated)
        new_params_all.update(new_params)

      for param in new_params_all:
        file_templates[ti]['params'].append({'type': (param_types[param[0]]), 'name': '<' + param + '>'})
      file_templates[ti]['text'] = texts

    with open(os.path.join(args.template_dir + '_pl', fn), 'w') as fout:
      fout.write(json.dumps(file_templates))


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

