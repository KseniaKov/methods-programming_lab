"""
Вычисление символа Лежандра при помощи символа Якоби
(по материалам «Исследование квадратных сравнений. Квадратичные вычеты и невычеты»).

Обозначения:
    (a/p)  - символ Лежандра, p - нечётное простое;
    (a/n)  - символ Якоби,  n - нечётное натуральное;
    при простом n символ Якоби совпадает с символом Лежандра.

Алгоритм (трудоёмкость O(log^2 n), факторизация не нужна):
    Шаг 1. Заменить a на b: a = b (mod n), |b| < n/2.
    Шаг 2. Представить b = (-1)^t * 2^k * c, c нечётно.
           (a/n) = (-1/n)^t * (2/n)^k * (c/n).
    Шаг 3. Вычислить (-1/n) = (-1)^((n-1)/2) и (2/n) = (-1)^((n^2-1)/8),
           если t или k нечётны.
    Шаг 4. Для (c/n) применить закон взаимности Гаусса
           (c/n) = (n/c) * (-1)^((c-1)/2 * (n-1)/2)
           и вернуться к шагу 1 с парой (n mod c, c).

Соглашение: если gcd(a, n) > 1, символ равен 0 (в методичке рассматривается
только gcd(a, n) = 1).
"""

# ----------------------------------------------------------------------
# Знаки из шага 3 и шага 4. Формулы (-1)^((n-1)/2) и (-1)^((n^2-1)/8)
# зависят только от n mod 4 и n mod 8, поэтому степени считать не нужно.
# ----------------------------------------------------------------------


def sign_minus_one(n: int) -> int:
    """(-1/n) = (-1)^((n-1)/2)."""
    return -1 if n % 4 == 3 else 1


def sign_two(n: int) -> int:
    """(2/n) = (-1)^((n^2-1)/8)."""
    return -1 if n % 8 in (3, 5) else 1


def sign_reciprocity(a: int, n: int) -> int:
    """(-1)^((a-1)/2 * (n-1)/2) для нечётных a, n."""
    return -1 if a % 4 == 3 and n % 4 == 3 else 1


def _check_odd_modulus(n: int) -> None:
    if n <= 0 or n % 2 == 0:
        raise ValueError("модуль должен быть нечётным натуральным числом")


# ----------------------------------------------------------------------
# Символ Якоби
# ----------------------------------------------------------------------


def jacobi(a: int, n: int, trace: list = None) -> int:
    """
    Символ Якоби (a/n), n - нечётное натуральное.
    Возвращает 1, -1 или 0 (если gcd(a, n) > 1).

    Если передан список trace, в него добавляются строки с описанием шагов.
    (Строки строятся только при включённой трассировке: перевод больших чисел
    в текст дорог и испортил бы оценку O(log^2 n).)
    """
    _check_odd_modulus(n)
    tracing = trace is not None

    result = 1
    if tracing:
        trace.append(f"вычисляем ({a}/{n})")
    while True:
        if n == 1:
            return result
        # Шаг 1: наименьший по модулю вычет, |b| < n/2
        b = a % n
        if b > n // 2:
            b -= n
        if tracing and b != a:
            trace.append(f"шаг 1: ({a}/{n}) = ({b}/{n})")
        if b == 0:
            if tracing:
                trace.append("a делится на n, значит gcd > 1: символ равен 0")
            return 0

        # Шаги 2-3: b = (-1)^t * 2^k * c
        if b < 0:
            s = sign_minus_one(n)
            if s == -1:
                result = -result
            if tracing:
                trace.append(f"шаг 3: (-1/{n}) = {s}")
            b = -b
        k = (b & -b).bit_length() - 1          # число двоек в b
        c = b >> k
        if k % 2 == 1:
            s = sign_two(n)
            if s == -1:
                result = -result
            if tracing:
                trace.append(f"шаг 3: (2/{n}) = {s}  (двойка в нечётной степени k={k})")
        if tracing and k:
            trace.append(f"шаг 2: b = 2^{k} * {c}")

        # Шаг 4: закон взаимности для (c/n)
        if c == 1:
            if tracing:
                trace.append(f"c = 1, (1/{n}) = 1; результат {result}")
            return result
        s = sign_reciprocity(c, n)
        if s == -1:
            result = -result
        if tracing:
            trace.append(f"шаг 4: ({c}/{n}) = ({n % c}/{c}) * ({s})")
        a, n = n % c, c


# ----------------------------------------------------------------------
# Символ Лежандра
# ----------------------------------------------------------------------

_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)


def is_probable_prime(n: int) -> bool:
    """Тест Миллера-Рабина; детерминирован для n < 3.3 * 10^24."""
    if n < 2:
        return False
    for q in _MR_BASES:
        if n % q == 0:
            return n == q
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def legendre(a: int, p: int, check_prime: bool = True) -> int:
    """
    Символ Лежандра (a/p) для нечётного простого p: 1 (вычет), -1 (невычет),
    0 (a делится на p). Вычисляется через символ Якоби.
    """
    if p < 3 or p % 2 == 0:
        raise ValueError("p должно быть нечётным простым числом")
    if check_prime and not is_probable_prime(p):
        raise ValueError("p не является простым числом")
    return jacobi(a, p)


def num_solutions(a: int, p: int) -> int:
    """Число решений сравнения x^2 = a (mod p): 1 + (a/p)."""
    return 1 + legendre(a, p)


# ----------------------------------------------------------------------
# Вспомогательные алгоритмы для сравнения и проверки
# ----------------------------------------------------------------------


def legendre_euler(a: int, p: int) -> int:
    """Критерий Эйлера: a^((p-1)/2) = (a/p) (mod p)."""
    r = pow(a % p, (p - 1) // 2, p)
    return -1 if r == p - 1 else r


def legendre_gauss(a: int, p: int) -> int:
    """Критерий Гаусса: (a/p) = (-1)^m, m - число отрицательных вычетов
    среди наименьших по модулю вычетов чисел a, 2a, ..., (p-1)/2 * a. O(p)."""
    if a % p == 0:
        return 0
    m = sum(1 for i in range(1, (p - 1) // 2 + 1) if (i * a) % p > p // 2)
    return -1 if m % 2 else 1


def _factorize(n: int) -> dict:
    """Разложение на простые пробными делениями, O(sqrt(n))."""
    f = {}
    while n % 2 == 0:
        f[2] = f.get(2, 0) + 1
        n //= 2
    q = 3
    while q * q <= n:
        while n % q == 0:
            f[q] = f.get(q, 0) + 1
            n //= q
        q += 2
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f


def legendre_factorization(a: int, p: int) -> int:
    """
    Первый алгоритм из методички: символ Лежандра с факторизацией |b|.
    Основной вклад в сложность вносит факторизация (экспоненциальная по длине p).
    p должно быть нечётным простым.
    """
    a %= p
    if a == 0:
        return 0
    b = a if 2 * a < p else a - p              # шаг 1
    result = 1
    if b < 0:                                  # знак: (-1/p)
        result *= sign_minus_one(p)
        b = -b
    for q, e in _factorize(b).items():         # шаг 2
        if e % 2 == 0:
            continue
        if q == 2:                             # шаг 3: (2/p)
            result *= sign_two(p)
        else:                                  # шаг 4: взаимность и рекурсия
            result *= sign_reciprocity(q, p) * legendre_factorization(p % q, q)
    return result


if __name__ == "__main__":
    steps = []
    value = jacobi(1001, 9907, steps)
    print("\n".join(steps))
    print(f"(1001/9907) = {value}")
    assert value == legendre_euler(1001, 9907)
    print("Проверка критерием Эйлера: OK")
