"""FLOPs of every dot and convolution in an HLO text, split by operand dtype."""
import re, sys, collections
from math import prod
def shape(t):  # 'bf16[2,3,4]{...}' -> ('bf16', [2,3,4])
    m = re.match(r'(\w+)\[([\d,]*)\]', t); return m[1], [int(x) for x in m[2].split(',') if x]
tot = collections.Counter(); n = collections.Counter()
for line in open(sys.argv[1]):
    m = re.search(r'= (\w+\[[\d,]*\])\S* (dot|convolution)\((\w+\[[\d,]*\])\S* %\S+, (\w+\[[\d,]*\])\S* %\S+\)(.*)', line)
    if not m: continue
    (odt, out), op, (ldt, lhs), (rdt, rhs), rest = shape(m[1]), m[2], shape(m[3]), shape(m[4]), m[5]
    if op == 'dot':
        cd = re.search(r'lhs_contracting_dims=\{([\d,]*)\}', rest)
        k = prod(lhs[int(i)] for i in cd[1].split(',') if i) if cd else 1
        fl = 2 * prod(out) * k
    else:
        lab = re.search(r'dim_labels=\w+_(\w+)->', rest)[1]
        fl = 2 * prod(out) * prod(rhs) // rhs[lab.index('o')]
    key = (op, f'{ldt}x{rdt}->{odt}'); tot[key] += fl; n[key] += 1
T = sum(tot.values())
for k in sorted(tot, key=lambda k: -tot[k]):
    print(f'{k[0]:<12} {k[1]:<22} n {n[k]:>5}  GFLOP {tot[k]/1e9:>10.3f}  {100*tot[k]/T:6.2f}%')
print('TOTAL GFLOP', T/1e9)
