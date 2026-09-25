"""Operand dtypes of every dot/convolution in an HLO text (static instances)."""
import re, sys, collections
types = {}
lines = open(sys.argv[1]).read().splitlines()
for l in lines:
    m = re.match(r'\s*(?:ROOT )?([\w.\-]+) = (\w+)\[', l)
    if m: types[m[1]] = m[2]
c = collections.Counter(); ex = {}
for l in lines:
    m = re.match(r'\s*(?:ROOT )?[\w.\-]+ = (\w+)\[([\d,]*)\]\S* (dot|convolution)\(([\w.\-]+), ([\w.\-]+)\)', l)
    if not m: continue
    key = (m[3], f'{types.get(m[4],"?")}x{types.get(m[5],"?")}->{m[1]}')
    c[key] += 1; ex.setdefault(key, l.strip()[:230])
for k, v in c.most_common():
    print(f'{k[0]:<12} {k[1]:<20} static instances {v}')
    print('    e.g.', ex[k])
