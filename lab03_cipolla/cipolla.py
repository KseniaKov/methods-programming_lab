"""
Вероятностный алгоритм Чипполы: извлечение квадратного корня в конечном поле Z_p
(по материалам «Решение некоторых типов уравнений в кольцах вычетов»).

ДАНО:  нечётное простое p, a из Z, (a, p) = 1, символ Лежандра (a/p) = 1.
ВЫХОД: x0 - решение уравнения x^2 = a в Z_p (второе решение равно -x0).

Шаг 1. Случайно выбрать b, 0 <= b <= p-1, такое что ((b^2 - 4a)/p) = -1.
Шаг 2. Положить f(y) = y^2 - b*y + a из Z_p[y].
Шаг 3. x0 - остаток от деления y^((p+1)/2) на f(y). Это константа.

Почему это работает. Многочлен f неприводим над Z_p (его дискриминант -
квадратичный невычет), поэтому Z_p[y]/(f) - поле из p^2 элементов, а y - корень f.
Второй корень f равен b - y и совпадает с y^p (автоморфизм Фробениуса), поэтому
    y^(p+1) = y * y^p = y * (b - y) = a,
то есть x = y^((p+1)/2) удовлетворяет x^2 = a. У уравнения t^2 = a в поле Z_p^2
лишь два корня, +-r, и они лежат в Z_p (a - вычет), значит, x - константа.

Элемент Z_p[y]/(f) хранится парой (u, v) = u + v*y. Умножение многочленов с
последующим взятием остатка по f(y) (y^2 = b*y - a) даёт
    (u1 + v1 y)(u2 + v2 y) = (u1 u2 - a v1 v2) + (u1 v2 + u2 v1 + b v1 v2) y.

Сложность: для a - вычета ровно (p-1)/2 из p значений b подходят, поэтому в среднем
нужно 2p/(p-1) < 3 попыток (для p > 3), каждая - один символ Лежандра за O(log^3 p) по
критерию Эйлера; возведение в степень - O(log p) умножений в поле, итого в
среднем O(log^3 p).
"""
import random

# ----------------------------------------------------------------------
# Простота и символ Лежандра (критерий Эйлера)
# ----------------------------------------------------------------------

_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)


def is_probable_prime(n: int) -> bool:
    """Тест Миллера - Рабина; детерминирован для n < 3.3 * 10^24."""
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


def legendre_euler(a: int, p: int) -> int:
    """Символ Лежандра по критерию Эйлера: a^((p-1)/2) mod p -> 1, -1 или 0."""
    r = pow(a % p, (p - 1) // 2, p)
    return -1 if r == p - 1 else r


def _check_prime(p: int) -> None:
    if p < 3 or p % 2 == 0 or not is_probable_prime(p):
        raise ValueError("p должно быть нечётным простым числом")


# ----------------------------------------------------------------------
# Арифметика в Z_p[y] / (y^2 - b*y + a)
# ----------------------------------------------------------------------


def fp2_mul(x: tuple, y: tuple, a: int, b: int, p: int) -> tuple:
    """Произведение (u1 + v1 y)(u2 + v2 y) по модулю f(y) = y^2 - b*y + a."""
    u1, v1 = x
    u2, v2 = y
    t = v1 * v2 % p
    return ((u1 * u2 - a * t) % p, (u1 * v2 + u2 * v1 + b * t) % p)


def fp2_pow(x: tuple, e: int, a: int, b: int, p: int) -> tuple:
    """Бинарное возведение в степень e >= 0 в Z_p[y] / (f)."""
    result = (1, 0)
    while e:
        if e & 1:
            result = fp2_mul(result, x, a, b, p)
        x = fp2_mul(x, x, a, b, p)
        e >>= 1
    return result


# ----------------------------------------------------------------------
# Алгоритм Чипполы
# ----------------------------------------------------------------------


def cipolla(a: int, p: int, rng=None, info: dict = None) -> int:
    """
    Решение x0 уравнения x^2 = a в Z_p алгоритмом Чипполы.

    rng  - генератор с методом randrange (по умолчанию модуль random);
           нужен, чтобы получать воспроизводимые результаты.
    info - если передан словарь, в него записываются b, число попыток,
           дискриминант b^2 - 4a и коэффициенты f(y).

    Если a делится на p, возвращает 0. Если a - невычет, выбрасывает ValueError.
    """
    _check_prime(p)
    a %= p
    if a == 0:
        if info is not None:
            info.update(b=None, attempts=0, discriminant=None, f=None)
        return 0
    if legendre_euler(a, p) != 1:
        raise ValueError("a не является квадратичным вычетом по модулю p")

    rng = rng if rng is not None else random
    # Шаг 1: случайный b с ((b^2 - 4a)/p) = -1
    attempts = 0
    while True:
        b = rng.randrange(p)
        attempts += 1
        if legendre_euler(b * b - 4 * a, p) == -1:
            break

    # Шаги 2-3: остаток y^((p+1)/2) по модулю f(y) = y^2 - b*y + a
    u, v = fp2_pow((0, 1), (p + 1) // 2, a, b, p)
    if v != 0:                                   # по теории невозможно
        raise RuntimeError("результат не лежит в Z_p")
    if info is not None:
        info.update(b=b, attempts=attempts, discriminant=(b * b - 4 * a) % p,
                    f=(1, (-b) % p, a))
    return u


def sqrt_mod(a: int, p: int, rng=None) -> tuple:
    """Все решения x^2 = a (mod p) по возрастанию: два, одно (a = 0) или ValueError."""
    x0 = cipolla(a, p, rng)
    return tuple(sorted({x0, (-x0) % p}))


def sqrt_3mod4(a: int, p: int) -> int:
    """Явная формула для p = 3 (mod 4): x0 = a^((p+1)/4) mod p (для сравнения)."""
    _check_prime(p)
    if p % 4 != 3:
        raise ValueError("формула применима только при p = 3 (mod 4)")
    a %= p
    x = pow(a, (p + 1) // 4, p)
    if x * x % p != a:
        raise ValueError("a не является квадратичным вычетом по модулю p")
    return x


if __name__ == "__main__":
    p, a = 13, 10
    info = {}
    x0 = cipolla(a, p, random.Random(1), info)
    print(f"x^2 = {a} (mod {p})")
    print(f"b = {info['b']} (попыток: {info['attempts']}), "
          f"b^2 - 4a = {info['discriminant']} mod {p}, "
          f"f(y) = y^2 - {info['b']}*y + {a}")
    print(f"x0 = {x0}, второе решение {p - x0}")
    assert x0 * x0 % p == a
    print("Проверка x0^2 = a (mod p): OK")
