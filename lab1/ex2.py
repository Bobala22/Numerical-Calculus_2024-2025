import random
import sys

import numpy as np


def find_machine_precision():
    u = 1.0
    while 1.0 + u != 1.0:
        u /= 2.0
    return u * 2.0

def check_addition_non_associativity(u):
    x = 1.0
    y = u / 10
    z = u / 10
    
    left_sum = (x + y) + z
    right_sum = x + (y + z)
    
    print("Verificare neasociativitate adunare:")
    print(f"u (precizie mașină) = {u}")
    print(f"x = {x}")
    print(f"y = u/10 = {y}")
    print(f"z = u/10 = {z}")
    print(f"(x + y) + z = {left_sum}")
    print(f"x + (y + z) = {right_sum}")
    print(f"Neasociativ: {left_sum != right_sum}")
    
    return left_sum, right_sum

def check_multiplication_non_associativity():
    print("\nVerificare neasociativitate înmulțire:")
    for _ in range(10):
        a = random.uniform(1e-10, 1e10)
        b = random.uniform(1e-10, 1e10)
        c = random.uniform(1e-10, 1e10)
        
        left_mult = (a * b) * c
        right_mult = a * (b * c)
        
        if not np.isclose(left_mult, right_mult, rtol=1e-10):
            print(f"Exemplu de neasociativitate:")
            print(f"a = {a}")
            print(f"b = {b}")
            print(f"c = {c}")
            print(f"(a * b) * c = {left_mult}")
            print(f"a * (b * c) = {right_mult}")
            print(f"Diferență absolută: {abs(left_mult - right_mult)}")
            return True
    
    print("Nu s-a găsit un exemplu de neasociativitate în 10 încercări.")
    return False

def main():
    # Găsește precizia mașină
    u = find_machine_precision()
    
    # Verifică neasociativitatea adunării
    check_addition_non_associativity(u)
    
    # Verifică neasociativitatea înmulțirii
    check_multiplication_non_associativity()

if __name__ == "__main__":
    main()
