from typing import Dict
from rnghax import *

# stats about how much of the decoded rid is usually valid
#
# a deltas:
# delta 0: 85.72800000000001%
# delta 6: 0.361%
# delta 1: 13.178999999999998%
# delta 4: 0.13999999999999999%
# delta 2: 0.215%
# delta 5: 0.27999999999999997%
# delta 3: 0.063%
# delta 11: 0.016%
# delta 10: 0.009000000000000001%
# delta 16: 0.001%
# delta 7: 0.001%
# delta 9: 0.003%
# delta 8: 0.004%
# ---
# b deltas:
# delta 1: 10.821%
# delta 51: 5.645%
# delta 0: 71.58099999999999%
# delta 52: 3.0620000000000003%
# delta 50: 3.785%
# delta 2: 0.18%
# delta 49: 2.122%
# delta 47: 0.525%
# delta 6: 0.281%
# delta 44: 0.066%
# delta 46: 0.28800000000000003%
# delta 48: 1.039%
# delta 5: 0.20500000000000002%
# delta 4: 0.126%
# delta 45: 0.136%
# delta 41: 0.005%
# delta 3: 0.059000000000000004%
# delta 43: 0.034%
# delta 42: 0.017%
# delta 11: 0.006999999999999999%
# delta 40: 0.008%
# delta 9: 0.002%
# delta 10: 0.001%
# delta 7: 0.001%
# delta 8: 0.002%
# delta 39: 0.002%
# ---
# c deltas:
# delta 33: 0.893%
# delta 51: 8.974%
# delta 41: 16.029%
# delta 36: 6.932%
# delta 35: 3.53%
# delta 40: 7.825%
# delta 50: 6.027%
# delta 49: 3.2419999999999995%
# delta 42: 18.176000000000002%
# delta 52: 5.856%
# delta 37: 6.237%
# delta 39: 3.933%
# delta 47: 2.223%
# delta 43: 0.149%
# delta 31: 0.75%
# delta 34: 1.7260000000000002%
# delta 45: 0.62%
# delta 46: 1.347%
# delta 44: 0.302%
# delta 48: 1.804%
# delta 29: 0.157%
# delta 38: 2.048%
# delta 32: 0.6669999999999999%
# delta 27: 0.048%
# delta 30: 0.365%
# delta 28: 0.096%
# delta 26: 0.023%
# delta 25: 0.006%
# delta 24: 0.009000000000000001%
# delta 21: 0.001%
# delta 23: 0.003%
# delta 22: 0.001%
# delta 19: 0.001%

def msb(x: int):
    msb = 0;
    while (x > 0):
        x = int(x / 2)
        msb += 1
    return msb


diffa: Dict[int, int] = {}
diffb: Dict[int, int] = {}
diffc: Dict[int, int] = {}
rng = Xs128pV8(u64(123), u64(456))
s0 = u64(123)
s1 = u64(456)
#for _ in range(1000): s0, s1 = xs128p_v8(s0, s1)
# initialize rng
for _ in range(1000): rng.shift()


n = 100000
for _ in range(n):
    a = rng.next_f64()
    b = rng.next_f64()
    c = rng.next_f64()
    
    sess = b36enc(a) + b36enc(b) + b36enc(c)
    sess = sess[:24]

    ax, bx, cx = dec_rid(sess)
    da = msb(abs(int(dq_v8(a)) - int(dq_v8(ax.val))))
    db = msb(abs(int(dq_v8(b)) - int(dq_v8(bx.val))))
    dc = msb(abs(int(dq_v8(c)) - int(dq_v8(cx.val))))

    count = diffa.get(da, 0)
    diffa[da] = count +1

    count = diffb.get(db, 0)
    diffb[db] = count +1

    count = diffc.get(dc, 0)
    diffc[dc] = count +1

print('---')
print('a deltas:')
for k, v in diffa.items(): print(f'delta {k}: {(v / n) * 100}%')

print('---')
print('b deltas:')
for k, v in diffb.items(): print(f'delta {k}: {(v / n) * 100}%')

print('---')
print('c deltas:')
for k, v in diffc.items(): print(f'delta {k}: {(v / n) * 100}%')
