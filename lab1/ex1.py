import numpy as np

u = 1.0
m = 0

while 1.0 + u != 1.0:
    m += 1
    u = 10 ** -m

u = 10 ** -(m-1)

print(f"Cel mai mic u care satisface condiția este 10^(-{m-1}) = {u}")

print(f"Precizia mașină din numpy: {np.finfo(float).eps}")
