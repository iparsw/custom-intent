from functools import wraps
import sys
from decimal import Decimal, localcontext
import random
import math
import gmpy2
import turtle
import numpy as np
import matplotlib.pyplot as plt
import cmath
from matplotlib import rcParams
from numpy import pi
from time import perf_counter

sys.setrecursionlimit(100_000)


def memorize(func):
    cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return wrapper


def Ptimeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        total_time = end_time - start_time
        # first item in the args, ie `args[0]` is `self`
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


# for better memory performance just write @memorize before your function

# fast math functions

# squre root
def Psqrt(n, one):
    """
    Return the square root of n as a fixed point number with the one
    passed in.  It uses a second order Newton-Raphson convergence.  This
    doubles the number of significant figures on each iteration.
    """
    # Use floating point arithmetic to make an initial guess
    floating_point_precision = 10 ** 16
    n_float = float((n * floating_point_precision) // one) / floating_point_precision
    javab = (int(floating_point_precision * math.sqrt(n_float)) * one) // floating_point_precision
    n_one = n * one
    while 1:
        x_old = javab
        javab = (javab + n_one // javab) // 2
        if javab == x_old:
            break
    return javab


def Psqrt_approx(num, root=2, n_dec=10):
    nat_num = 1
    result = None
    while nat_num ** root <= num:
        result = nat_num
        nat_num += 1
    for d in range(1, n_dec + 1):
        increment = 10 ** -d
        count = 1
        before = result
        while (before + increment * count) ** root <= num:
            result = before + increment * count
            count += 1
    return round(result, n_dec)


# squre root with taylor series
def fact(n):
    if n == 1 or n == 0:
        return 1
    else:
        return n * fact(n - 1)


def Psqrt_taylor_bin_coef(m, k):  # m choose k
    numerator = 1
    for factor in range(0, k):
        numerator = numerator * (m - factor)
    denominator = fact(k)
    coef = numerator / denominator
    return coef


def Psqrt_taylor(num):
    exp = len(str(num))
    x = (num / (10 ** exp)) - 1
    sumation = 0
    for term in range(10):
        sumation += Psqrt_taylor_bin_coef(1 / 2, term) * (x ** term)
    if exp % 2 == 0:
        result = sumation * (10 ** (exp / 2))
    else:
        result = sumation * 3.16227766017 * (10 ** ((exp - 1) / 2))
    return result


# kmm kamtarin mazrab moshtarak lcm
def Pkmm(firts, second) -> int:
    k1 = int(firts)
    k2 = int(second)
    while firts % second != 0:
        r = firts % second
        firts = second
        second = r
    gcd = int(second)
    lcm = int((k1 * k2) / gcd)
    return lcm


# bmm bozorg tarin maghsoom alayh moshtarak ( gcd )
def Pbmm(first, second) -> int:
    while first % second != 0:
        r = first % second
        first = second
        second = r
    gcd = second
    return gcd


# fibonacci series
@memorize
def fibonacci(n) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def fibFast1(n):
    s = 5 ** 0.5
    P = (1 + s) / 2
    p = (1 - s) / 2
    return round((P ** n - p ** n) / s)


def fibFast2(num):
    a, b = 0, 1
    for _ in range(num - 1):
        a, b = b, a + b
    return b


def fib_smart(n):
    if n >= 70:
        return fibFast1(n)
    else:
        return fibFast2(n)


def fibonacci_taghsim(n) -> int:
    if n == 1:
        a = 1
    else:
        a = fibonacci(n) / fibonacci(n - 1)
    return a


# e number

def calculating_e(n) -> int:
    e_value = 1
    for i in range(n):
        e_value += 1 / math.factorial(i + 1)
    return e_value


def calculating_e_pro(n, precision):
    with localcontext() as ctx:
        ctx.prec = precision  # "precision" digits precision
        e_value = Decimal(1)
        for i in range(n):
            e_value += Decimal(1) / Decimal(math.factorial(i + 1))
    return e_value


def calculating_e_2(n):
    e_value = ((1 + (1 / n)) ** n)
    return e_value


def calculating_e_2_pro(n, precision):
    with localcontext() as ctx:
        ctx.prec = precision  # "precision" digits precision
        e_value = Decimal((Decimal(1) + Decimal((Decimal(1) / Decimal(n)))) ** Decimal(n))
        return e_value


def calculating_e_2_smart(digits):
    seed = Decimal(Decimal(10) ** (Decimal(digits - 1)))
    prc = digits
    result = calculating_e_2_pro(seed, prc)
    return result


def calculating_e_printing_every_step(n) -> int:
    e_value = 1
    for i in range(n):
        e_value += 1 / math.factorial(i + 1)
        print(e_value)
    return e_value


def e_number():
    e_value = calculating_e(17)
    return e_value


# pi number

def pi_1(n, precision):
    with localcontext() as ctx:
        ctx.prec = precision  # "precision" digits precision
        pi_value = Decimal(0)
        for k in range(n):
            pi_value += (Decimal(4) / (Decimal(8) * k + 1) - Decimal(2) / (Decimal(8) * k + 4) - Decimal(1) / (
                    Decimal(8) * k + 5) - Decimal(1) / (Decimal(8) * k + 6)) / Decimal(16) ** k
    return pi_value


def pi_Nilakantha_1(reps, decimals):
    with localcontext() as ctx:
        ctx.prec = decimals
        answer = Decimal(3.0)
        op = 1
        for n in range(2, 2 * reps + 1, 2):
            answer += 4 / Decimal(n * (n + 1) * (n + 2) * op)
            op *= -1
    return answer


def pi_Chudnovsky_bs_1(digits):
    """
    Compute int(pi * 10**digits)

    This is done using Chudnovsky's series with binary splitting
    """
    C = 640320
    C3_OVER_24 = C ** 3 // 24

    def bs(a, b):
        """
        Computes the terms for binary splitting the Chudnovsky infinite series

        a(a) = +/- (13591409 + 545140134*a)
        p(a) = (6*a-5)*(2*a-1)*(6*a-1)
        b(a) = 1
        q(a) = a*a*a*C3_OVER_24

        returns P(a,b), Q(a,b) and T(a,b)
        """
        if b - a == 1:
            # Directly compute P(a,a+1), Q(a,a+1) and T(a,a+1)
            if a == 0:
                Pab = Qab = gmpy2.mpz(1)
            else:
                Pab = gmpy2.mpz((6 * a - 5) * (2 * a - 1) * (6 * a - 1))
                Qab = gmpy2.mpz(a * a * a * C3_OVER_24)
            Tab = Pab * (13591409 + 545140134 * a)  # a(a) * p(a)
            if a & 1:
                Tab = -Tab
        else:
            # Recursively compute P(a,b), Q(a,b) and T(a,b)
            # m is the midpoint of a and b
            m = (a + b) // 2
            # Recursively calculate P(a,m), Q(a,m) and T(a,m)
            Pam, Qam, Tam = bs(a, m)
            # Recursively calculate P(m,b), Q(m,b) and T(m,b)
            Pmb, Qmb, Tmb = bs(m, b)
            # Now combine
            Pab = Pam * Pmb
            Qab = Qam * Qmb
            Tab = Qmb * Tam + Pam * Tmb
        return Pab, Qab, Tab

    # how many terms to compute
    DIGITS_PER_TERM = math.log10(C3_OVER_24 / 6 / 2 / 6)
    N = int(digits / DIGITS_PER_TERM + 1)
    # Calculate P(0,N) and Q(0,N)
    P, Q, T = bs(0, N)
    one = 10 ** digits
    sqrtC = Psqrt(10005 * one, one)
    return (Q * 426880 * sqrtC) // T


def pi_MonteCarlo_visiual(number_of_points):
    number_of_points = int(number_of_points)

    turtle.speed("fastest")
    length = 300  # radius of circle and length of the square in pixels

    # draw y axis
    turtle.pensize(2)
    turtle.forward(length + 40)
    turtle.left(135)
    turtle.forward(20)
    turtle.back(20)
    turtle.left(90)
    turtle.forward(20)

    turtle.penup()
    turtle.home()
    turtle.pendown()

    # draw x axis
    turtle.left(90)
    turtle.forward(length + 40)
    turtle.left(135)
    turtle.forward(20)
    turtle.back(20)
    turtle.left(90)
    turtle.forward(20)

    turtle.penup()
    turtle.goto(0, length)
    turtle.left(45)
    turtle.left(180)
    turtle.pendown()

    # draw quarter of circle
    turtle.pencolor("red")
    turtle.circle(length, -90)

    inside = 0
    for i in range(0, number_of_points):
        # get dot position
        x = random.uniform(0, length)
        y = random.uniform(0, length)
        # determine distance from center
        d = math.sqrt(x ** 2 + y ** 2)
        if d <= length:
            inside += 1
            turtle.pencolor("red")
        else:
            turtle.pencolor("blue")
        # draw dot
        turtle.penup()
        turtle.goto(x, y)
        turtle.pendown()
        turtle.dot()
    return (inside / number_of_points) * 4.0


def pi_MonteCarlo_Visiual_2(number_of_points):
    # Draw a square and a circle to frame out simulation
    squareX = [1, -1, -1, 1, 1]
    squareY = [1, 1, -1, -1, 1]
    circleX, circleY = [], []

    for i in range(361):
        circleX.append(np.cos(np.pi * i / 180))
        circleY.append(np.sin(np.pi * i / 180))

    # Start keeping track of values we're interested in
    insideX, insideY, outsideX, outsideY, Iteration, CurrentPi = [], [], [], [], [], []
    insideCounter = 0

    # Generate a bunch of values of x and y between -1 and 1, then assess their combined radius on a xy plane
    for i in range(number_of_points):
        x = 2 * (np.random.random() - 0.5)
        y = 2 * (np.random.random() - 0.5)
        r = np.sqrt(x ** 2 + y ** 2)
        Iteration.append(i)
        if r <= 1:
            insideCounter += 1
            insideX.append(x)
            insideY.append(y)
        else:
            outsideX.append(x)
            outsideY.append(y)
        CurrentPi.append(4 * insideCounter / (i + 1))

    piValue = 4 * insideCounter / number_of_points
    piError = round(100 * ((piValue - pi) / pi), 4)

    # Draw a 2D plot of where our iterations landed compared to the square and circle
    rcParams['figure.figsize'] = 5, 5
    plt.plot(squareX, squareY, color='#000000')
    plt.plot(circleX, circleY, color='#0000CC')
    plt.scatter(insideX, insideY, color='#00CC00', marker=".")
    plt.scatter(outsideX, outsideY, color='#CC0000', marker=".")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # Draw a psuedo-time series plot of current estimate of pi vs. iteration number
    plt.plot(Iteration, CurrentPi, color='#009900')
    plt.axhline(y=pi, color='#0F0F0F', ls='--')
    plt.axis([0, number_of_points, 0, 4.1])
    plt.xlabel('Iteration Number')
    plt.ylabel('Estimate for pi')
    plt.show()

    # print out our final estimate and how it compares to the true value
    print('\n' + f'Pi is approximately {piValue}\n')
    print(f'This is {piError}% off the true value.\n')
    return piValue


def pi_GregoryLeibniz_1(reps):
    n = 1
    pi_value = 0

    for i in range(reps):
        if i % 2 == 0:
            pi_value += (1 / n)
        else:
            pi_value -= (1 / n)
        n += 2

    pi_value *= 4
    return pi_value


def pi_GregoryLeibniz_pro(reps, decimals):
    with localcontext() as ctx:
        ctx.prec = decimals
        n = Decimal(1)
        pi_value = Decimal(0)

        for i in range(reps):
            if i % 2 == 0:
                pi_value += Decimal((Decimal(1) / n))
            else:
                pi_value -= Decimal((Decimal(1) / n))
            n += Decimal(2)

        pi_value *= 4
    return pi_value


def pi_RamanujanSato_1(reps):
    pi_sum = 0

    for i in range(reps):
        pi_sum += (math.factorial(4 * i) * (26390 * i + 1103)) / ((math.factorial(i) ** 4) * 396 ** (4 * i))

    pi_sum *= (2 * math.sqrt(2) / (99 ** 2))
    pi_value = 1 / pi_sum
    return pi_value


def pi_RamanujanSato_pro(reps, decimals):
    with localcontext() as ctx:
        ctx.prec = decimals
        pi_sum = Decimal(0)

        for i in range(reps):
            pi_sum = pi_sum + Decimal(
                (math.factorial(4 * 1) * (26390 * i + 1103)) / ((math.factorial(i) ** 4) * 396 ** (4 * i)))

        pi_sum = pi_sum * (Decimal(2 * math.sqrt(2)) / Decimal(99 ** 2))
        pi_value = Decimal(1) / pi_sum
    return pi_value


# simple golden ratio function
def golden_ratio_1():
    golden_ratio = (1 + math.sqrt(5)) / 2
    return golden_ratio


def golden_ratio_1_pro(decimals):
    with localcontext() as ctx:
        ctx.prec = decimals
        sqrtOf5 = Decimal(5).sqrt()
        golden_ratio = Decimal((1 + sqrtOf5) / 2)
        return golden_ratio


# golden ratio as limit of fibonacci taghsim
def golden_ratio_2(seed):
    fn = fibonacci(seed)
    fn2 = fibonacci(seed - 1)
    golden_ratio = fn / fn2
    return golden_ratio


def golden_ratio_2_pro(seed, decimals):
    with localcontext() as ctx:
        ctx.prec = decimals
        fn = Decimal(fibonacci(seed))
        fn2 = Decimal(fibonacci(seed - 1))
        golden_ratio = fn / fn2
        return golden_ratio


def golden_ratio_2_smart(digits):
    if digits < 350:
        seed = int(((digits - 1) * 10) / 4)
        golden_ratio = golden_ratio_2_pro(seed, digits)
        return golden_ratio
    else:
        return False


def golden_ratio_3(value, times):
    for i in range(times):
        value = 1 / value
        value = value + 1
    return value


def golden_ratio_3_pro(value, times, decimals):
    with localcontext() as ctx:
        ctx.prec = decimals
        value = Decimal(value)
        for i in range(times):
            value = 1 / value
            value = value + 1
    return value


def golden_ratio_3_smart(decimals, times=0):
    if times == 0:
        times = int(decimals * 2.5)
    with localcontext() as ctx:
        value = 1.618033988749894
        ctx.prec = decimals
        value = Decimal(value)
        for i in range(times):
            value = 1 / value
            value = value + 1
    return value


# algebra functions

def quadratic_solution_count(a, b, c):
    delta = (b ** 2) - (4 * a * c)
    if delta > 0:
        return 2
    elif delta < 0:
        return 0
    else:
        return 1


def quadratic_solution_sum(a, b, c):
    return (0 - b) / a


def quadratoc_solution_product(a, b, c):
    return c / a


def quadratic_solution_legacy(a, b, c):
    delta = (b ** 2) - (4 * a * c)
    if delta >= 0:
        x1 = ((0 - b) + math.sqrt(delta)) / (2 * a)
        x2 = ((0 - b) - math.sqrt(delta)) / (2 * a)
        return [x1, x2]
    else:
        return False


def quadratic_solution(a, b, c):
    discriminant = b ** 2 - 4 * a * c
    if discriminant > 0:
        root1 = float(-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        root2 = float(-b - math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        return [root1, root2]
    elif discriminant == 0:
        root1 = float(-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        return root1
    elif discriminant < 0:
        root1 = (-b + cmath.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        root2 = (-b - cmath.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        return [root1, root2]


def is_float(element: any) -> bool:
    # If you expect None to be passed:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


# matrix oporatin

def determinant2by2(array):
    determinant = float((array[0, 0] * array[1, 1]) - (array[0, 1] * array[1, 0]))
    return determinant


def matrix_multiplication2by2(matrix1, matrix2):
    result = np.array([
        [((matrix1[0, 0] * matrix2[0, 0]) + (matrix1[0, 1] * matrix2[1, 0])),
         ((matrix1[0, 0] * matrix2[0, 1]) + (matrix1[0, 1] * matrix2[1, 1]))],
        [((matrix1[1, 0] * matrix2[0, 0]) + (matrix1[1, 1] * matrix2[1, 0])),
         ((matrix1[1, 0] * matrix2[0, 1]) + (matrix1[1, 1] * matrix2[1, 1]))],
    ])
    return result


# ecualidean_distance
def ecualidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
