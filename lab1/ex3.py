import math
import numpy as np

PI = np.pi
LEFT_SIN_INT = -PI/2
RIGHT_SIN_INT = PI/2
C1 = 0.16666666666666666666666666666667
C2 = 0.00833333333333333333333333333333
C3 = 1.984126984126984126984126984127e-4
C4 = 2.7557319223985890652557319223986e-6
C5 = 2.5052108385441718775052108385442e-8
C6 = 1.6059043836821614599392377170155e-10

def P1(x):
    y = x**2
    return x * (1 - y * (C1 + C2 * y))

def P2(x):
    y = x**2
    return x * (1 + y * (-C1 + y * (C2 - C3 * y)))

def P3(x):
    y = x**2
    return x * (1 + y * (-C1 + y * (C2 + y * (-C3 + C4 * y))))

def P4(x):
    y = x**2
    return x * (1 + y * (-0.166 + y * (0.00833 + y * (-C3 + C4 * y))))

def P5(x):
    y = x**2
    return x * (1 + y * (-0.1666 + y * (0.008333 + y * (-C3 + C4 * y))))

def P6(x):
    y = x**2
    return x * (1 + y * (-0.16666 + y * (0.0083333 + y * (-C3 + C4 * y))))

def P7(x):
    y = x**2
    return x * (1 + y * (-C1 + y * (C2 + y * (-C3 + y * (C4 - C5 * y)))))

def P8(x):
    y = x**2
    return x * (1 + y * (-C1 + y * (C2 + y * (-C3 + y * (C4 + y * (-C5 + C6 * y))))))

high_inclusive = np.nextafter(RIGHT_SIN_INT, RIGHT_SIN_INT + 1) # pentru a include capatul din dreapta
random_numbers = np.random.uniform(LEFT_SIN_INT, high_inclusive, 10000)

polynomials = [P1, P2, P3, P4, P5, P6, P7, P8]
top3 = [0, 0, 0, 0, 0, 0, 0, 0]

for point in random_numbers:
    exact_value = math.sin(point)
    errors = [abs(p(point) - exact_value) for p in polynomials]
    for i in sorted(range(len(polynomials)), key=lambda x: errors[x])[:3]:
        top3[i] += 1

rank = sorted(range(8), key=lambda i: top3[i], reverse=True)
print()
print("Top 3 polinoame cu cele mai mici erori:")
for i in rank[:3]:
    print(f"Polinomul {i+1} a fost in top 3 de {top3[i]} ori")

print()
print("Pentru a calcula topul celor 8 polinoame(toate), vom volosi mean square error")
mse_errors = []
for p in polynomials:
    mse = np.mean([(p(x) - math.sin(x))**2 for x in random_numbers])
    mse_errors.append((p, mse))

print("Topul celor 8 polinoame:")
for i in sorted(mse_errors, key=lambda x: x[1])[:8]:
    print(i[0].__name__, i[1])
    

