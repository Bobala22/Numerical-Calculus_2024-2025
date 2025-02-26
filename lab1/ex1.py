u = 1.0
m = 0

while 1.0 + u != 1.0:
    m += 1
    u = 10 ** -m

print(f"Cel mai mic u care satisface conditia este 10^(-{m-1})")
