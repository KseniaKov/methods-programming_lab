"""
Вероятностные тесты простоты: Ферма, Соловея - Штрассена, Миллера - Рабина
(по материалам «Проверка простоты целых чисел»).

Каждый тест работает с одним основанием a из {1, ..., N-1} (функции *_round) и
повторяется для нескольких случайных a (функции *_test).

Ответ теста для одного основания:
    False - «N - составное» (доказано: найден свидетель непростоты a);
    True  - «неизвестно»    (N простое либо a - «лжец» для составного N).
Ответ *_test: False - N точно составное; True - все s проверок дали «неизвестно»,
то есть N, вероятно, простое. Простое N всегда даёт True.

Тест Ферма:
    Шаг 1. d = (a, N); если d > 1 - «составное».
    Шаг 2. Если a^(N-1) != 1 (mod N) - «составное», иначе «неизвестно».
Тест Соловея - Штрассена:
    Шаг 1. d = (a, N); если d > 1 - «составное».
    Шаг 2. Если a^((N-1)/2) != (a/N) (mod N) - «составное», иначе «неизвестно».
    (a/N) - символ Якоби.
Тест Миллера - Рабина (N - 1 = 2^t * u, u нечётно):
    Шаг 1. d = (a, N); если d > 1 - «составное».
    Шаг 2. r_k = a^(2^k * u) mod N, k = 0..t-1. Если r_0 = 1 или r_k = -1 при
    некотором k - «неизвестно», иначе «составное».

Для составного N вероятность ответа «составное» за один раунд:
    Ферма - не гарантирована (числа Кармайкла: P0 = 1 - phi(N)/(N-1));
    Соловей - Штрассен - не менее 1/2;
    Миллер - Рабин - не менее 3/4.
Трудоёмкость одного раунда каждого теста - O(log^3 N).
"""
import random
from math import gcd

# ----------------------------------------------------------------------
# Символ Якоби (нужен тесту Соловея - Штрассена)
# ----------------------------------------------------------------------


def jacobi(a: int, n: int) -> int:
    """Символ Якоби (a/n), n - нечётное натуральное; 0, если gcd(a, n) > 1."""
    if n <= 0 or n % 2 == 0:
        raise ValueError("n должно быть нечётным натуральным числом")
    a %= n
    result = 1
    while a:
        while a % 2 == 0:               # выносим двойки: (2/n) = (-1)^((n^2-1)/8)
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a                     # закон взаимности
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a %= n
    return result if n == 1 else 0


# ----------------------------------------------------------------------
# Один раунд теста с заданным основанием a
# ----------------------------------------------------------------------


def _check_args(N: int, a: int) -> None:
    if N < 3 or N % 2 == 0:
        raise ValueError("N должно быть нечётным числом, N >= 3")
    if not 1 <= a <= N - 1:
        raise ValueError("основание a должно лежать в {1, ..., N-1}")


def fermat_round(N: int, a: int) -> bool:
    """Тест Ферма для основания a. False - «составное», True - «неизвестно»."""
    _check_args(N, a)
    if gcd(a, N) > 1:                                  # шаг 1
        return False
    return pow(a, N - 1, N) == 1                       # шаг 2, сравнение (1)


def solovay_strassen_round(N: int, a: int) -> bool:
    """Тест Соловея - Штрассена для основания a. False - «составное»."""
    _check_args(N, a)
    if gcd(a, N) > 1:                                  # шаг 1
        return False
    return pow(a, (N - 1) // 2, N) == jacobi(a, N) % N  # шаг 2, сравнение (2)


def mr_decompose(N: int) -> tuple:
    """N - 1 = 2^t * u, u нечётно; возвращает (t, u)."""
    t, u = 0, N - 1
    while u % 2 == 0:
        u //= 2
        t += 1
    return t, u


def mr_sequence(N: int, a: int) -> list:
    """Последовательность r_k = a^(2^k * u) mod N, k = 0..t-1 (для наглядности)."""
    _check_args(N, a)
    t, u = mr_decompose(N)
    r = pow(a, u, N)
    seq = [r]
    for _ in range(t - 1):
        r = r * r % N
        seq.append(r)
    return seq


def miller_rabin_round(N: int, a: int) -> bool:
    """Тест Миллера - Рабина для основания a. False - «составное»."""
    _check_args(N, a)
    if gcd(a, N) > 1:                                  # шаг 1
        return False
    t, u = mr_decompose(N)
    r = pow(a, u, N)                                   # r_0
    if r == 1 or r == N - 1:
        return True
    for _ in range(t - 1):                             # r_1, ..., r_{t-1}
        r = r * r % N
        if r == N - 1:
            return True
    return False


# ----------------------------------------------------------------------
# Тесты с несколькими случайными основаниями
# ----------------------------------------------------------------------


def _run(single, N: int, rounds: int, rng, info: dict) -> bool:
    if rounds < 1:
        raise ValueError("число раундов должно быть >= 1")
    if N < 2:
        raise ValueError("N должно быть больше 1")
    if info is not None:
        info.update(rounds=0, witness=None, gcd=None)
    if N == 2:
        return True
    if N % 2 == 0:
        if info is not None:
            info.update(witness=None, gcd=2)
        return False
    rng = rng if rng is not None else random
    for i in range(1, rounds + 1):
        a = rng.randrange(1, N)                        # a из {1, ..., N-1}
        ok = single(N, a)
        if info is not None:
            info["rounds"] = i
        if not ok:
            if info is not None:
                d = gcd(a, N)
                info.update(witness=a, gcd=d if d > 1 else None)
            return False
    return True


def fermat_test(N: int, rounds: int = 20, rng=None, info: dict = None) -> bool:
    """Тест Ферма для `rounds` случайных оснований. False - N составное."""
    return _run(fermat_round, N, rounds, rng, info)


def solovay_strassen_test(N: int, rounds: int = 20, rng=None, info: dict = None) -> bool:
    """Тест Соловея - Штрассена для `rounds` случайных оснований."""
    return _run(solovay_strassen_round, N, rounds, rng, info)


def miller_rabin_test(N: int, rounds: int = 20, rng=None, info: dict = None) -> bool:
    """Тест Миллера - Рабина для `rounds` случайных оснований."""
    return _run(miller_rabin_round, N, rounds, rng, info)


# ----------------------------------------------------------------------
# Вспомогательное: «лжецы» (для анализа вероятностей)
# ----------------------------------------------------------------------


def liars(N: int, single) -> list:
    """Все основания a из {1, ..., N-1}, для которых single(N, a) даёт «неизвестно»."""
    return [a for a in range(1, N) if single(N, a)]


if __name__ == "__main__":
    for N in (97, 91, 561, 2047):
        row = []
        for name, test in (("Ферма", fermat_test), ("Соловей-Штрассен", solovay_strassen_test),
                           ("Миллер-Рабин", miller_rabin_test)):
            info = {}
            verdict = "неизвестно" if test(N, 20, random.Random(1), info) else "составное"
            row.append(f"{name}: {verdict}")
        print(f"N = {N:5d} | " + " | ".join(row))
