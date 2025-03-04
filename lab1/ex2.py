import numpy as np


def find_machine_precision():
    u = 1.0
    m = 0
    while 1.0 + u != 1.0:
        m += 1
        u = 10 ** -m
    u = 10 ** -(m-1)
    return u

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
    x = 1.0e308 # f aprope de repr. maxima a unui nr in virgula mobila => cand inmultim cu 2, rezultatul va fi infinit (overflow)
    y = 3.0      
    z = 1.0e-300
    
    left_product = (x * y) * z
    right_product = x * (y * z)
    
    print("Verificare neasociativitate înmulțire:")
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"z = {z}")
    print(f"(x * y) * z = {left_product}")
    print(f"x * (y * z) = {right_product}")
    print(f"Neasociativ: {left_product != right_product}") 

    print()
    print("Another example: ")
    x = 1.234567e10
    y = 1.234567e-5
    z = 1.234567e-5

    left_product = (x * y) * z
    right_product = x * (y * z)
    
    print("Verificare neasociativitate înmulțire:")
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"z = {z}")
    print(f"(x * y) * z = {left_product}")
    print(f"x * (y * z) = {right_product}")
    print(f"Neasociativ: {left_product != right_product}")

def main():
    u = find_machine_precision()
    print()
    
    check_addition_non_associativity(u)
    print()
    
    check_multiplication_non_associativity()

if __name__ == "__main__":
    main()
