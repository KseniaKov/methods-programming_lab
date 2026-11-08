"""Тесты вероятностных тестов простоты. Запуск: `pytest` или `python -m unittest`."""
import random
import unittest
from math import gcd

from primality import (fermat_round, fermat_test, jacobi, liars, mr_decompose,
                       mr_sequence, miller_rabin_round, miller_rabin_test,
                       solovay_strassen_round, solovay_strassen_test)

ROUNDS = (fermat_round, solovay_strassen_round, miller_rabin_round)
TESTS = (fermat_test, solovay_strassen_test, miller_rabin_test)


# ---------------- независимые вспомогательные функции ----------------

def factorize(n):
    f, d = {}, 2
    while d * d <= n:
        while n % d == 0:
            f[d] = f.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f


def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def phi(n):
    r = n
    for p in factorize(n):
        r = r // p * (p - 1)
    return r


def is_carmichael(n):
    """Критерий Корселта: n составное, свободно от квадратов, (p-1) | (n-1)."""
    f = factorize(n)
    return (len(f) > 1 and all(e == 1 for e in f.values())
            and all((n - 1) % (p - 1) == 0 for p in f))


def brute_jacobi(a, n):
    """Определение 2.4: произведение символов Лежандра (перебор квадратов)."""
    res = 1
    for p, e in factorize(n).items():
        if a % p == 0:
            return 0
        res *= (1 if a % p in {x * x % p for x in range(1, p)} else -1) ** e
    return res


def chernick_carmichael(bits, rng):
    """Число Кармайкла (6k+1)(12k+1)(18k+1), все три множителя простые."""
    k = rng.getrandbits(bits) | (1 << (bits - 1))
    for _ in range(100000):                        # ограничение: поиск должен быть быстрым
        k += 1
        f = (6 * k + 1, 12 * k + 1, 18 * k + 1)
        if all(miller_rabin_test(x, 25, rng) for x in f):
            return f[0] * f[1] * f[2], f
    raise RuntimeError("не удалось найти число Кармайкла (тест простоты отвергает простые?)")


ODD_COMPOSITES = [n for n in range(9, 700, 2) if not is_prime(n)]
PRIMES = [p for p in range(3, 400) if is_prime(p)]


class TestJacobi(unittest.TestCase):
    def test_against_definition(self):
        for n in range(1, 150, 2):
            for a in range(-n, 2 * n):
                self.assertEqual(jacobi(a, n), brute_jacobi(a, n), (a, n))

    def test_invalid_modulus(self):
        for n in (0, -3, 2, 10):
            with self.assertRaises(ValueError):
                jacobi(3, n)


class TestPrimesAlwaysPass(unittest.TestCase):
    def test_every_base_for_every_prime(self):
        """Простое N: на любом основании ответ «неизвестно»."""
        for p in PRIMES:
            for single in ROUNDS:
                for a in range(1, p):
                    self.assertTrue(single(p, a), (single.__name__, p, a))

    def test_multi_round_accepts_primes(self):
        rnd = random.Random(1)
        for p in PRIMES + [2**61 - 1, 2**127 - 1, 2**521 - 1]:
            for test in TESTS:
                self.assertTrue(test(p, 20, rnd), (test.__name__, p))


class TestSingleRound(unittest.TestCase):
    def test_gcd_step(self):
        """Шаг 1: если (a, N) > 1, ответ «составное»."""
        for single in ROUNDS:
            self.assertFalse(single(91, 7))
            self.assertFalse(single(91, 26))
            self.assertFalse(single(15, 9))

    def test_base_one_is_useless(self):
        for N in (9, 15, 91, 561):
            for single in ROUNDS:
                self.assertTrue(single(N, 1))

    def test_known_pseudoprimes(self):
        self.assertTrue(fermat_round(341, 2))              # 341 = 11 * 31, псевдопростое по основанию 2
        self.assertFalse(fermat_round(341, 3))
        self.assertTrue(solovay_strassen_round(561, 2))    # наименьшее эйлерово псевдопростое по основанию 2
        self.assertTrue(miller_rabin_round(2047, 2))       # наименьшее сильное псевдопростое по основанию 2
        self.assertFalse(miller_rabin_round(2047, 3))

    def test_smallest_pseudoprimes_base_2(self):
        for single, expected in ((fermat_round, 341), (solovay_strassen_round, 561),
                                 (miller_rabin_round, 2047)):
            first = next(n for n in range(3, 5000, 2) if not is_prime(n) and single(n, 2))
            self.assertEqual(first, expected, single.__name__)

    def test_invalid_arguments(self):
        for single in ROUNDS:
            for N, a in ((2, 1), (10, 3), (1, 1), (-5, 2), (11, 0), (11, 11), (11, -1)):
                with self.assertRaises(ValueError):
                    single(N, a)

    def test_mr_decompose_and_sequence(self):
        for N in range(3, 2000, 2):
            t, u = mr_decompose(N)
            self.assertEqual(2**t * u, N - 1)
            self.assertEqual(u % 2, 1)
        # N = 561: N - 1 = 2^4 * 35, r_k = 2^(2^k * 35) mod 561
        self.assertEqual(mr_decompose(561), (4, 35))
        seq = mr_sequence(561, 2)
        self.assertEqual(len(seq), 4)
        self.assertEqual(seq, [pow(2, 2**k * 35, 561) for k in range(4)])


class TestMultiRound(unittest.TestCase):
    def test_small_and_even_inputs(self):
        for test in TESTS:
            self.assertTrue(test(2))
            self.assertTrue(test(3))
            for n in (4, 6, 100):
                self.assertFalse(test(n))
            for n in (-4, 0, 1):
                with self.assertRaises(ValueError):
                    test(n)
            with self.assertRaises(ValueError):
                test(11, rounds=0)

    def test_classification_matches_sieve(self):
        """Для всех N < 3000: ответы теста совпадают с точной проверкой простоты."""
        for test, rounds in ((fermat_test, 80), (solovay_strassen_test, 30),
                             (miller_rabin_test, 20)):
            rnd = random.Random(2)
            for N in range(2, 3000):
                self.assertEqual(test(N, rounds, rnd), is_prime(N), (test.__name__, N))

    def test_info(self):
        rnd = random.Random(3)
        info = {}
        self.assertFalse(miller_rabin_test(91 * 97, 20, rnd, info))
        a = info["witness"]
        self.assertGreaterEqual(info["rounds"], 1)
        self.assertFalse(miller_rabin_round(91 * 97, a))
        info = {}
        self.assertTrue(miller_rabin_test(97, 7, rnd, info))
        self.assertEqual((info["rounds"], info["witness"], info["gcd"]), (7, None, None))
        info = {}
        fermat_test(100, 5, rnd, info)
        self.assertEqual(info["gcd"], 2)

    def test_gcd_reported(self):
        """Свидетель с общим делителем: в info записан делитель."""
        class FixedRng:
            def randrange(self, lo, hi):
                return 7
        info = {}
        self.assertFalse(solovay_strassen_test(91, 5, FixedRng(), info))
        self.assertEqual((info["witness"], info["gcd"], info["rounds"]), (7, 7, 1))

    def test_reproducible_with_seed(self):
        for test in TESTS:
            r1, r2 = {}, {}
            v1 = test(1000003 * 999983, 10, random.Random(5), r1)
            v2 = test(1000003 * 999983, 10, random.Random(5), r2)
            self.assertEqual((v1, r1), (v2, r2))

    def test_bases_are_from_1_to_n_minus_1(self):
        seen = set()

        class Spy:
            def randrange(self, lo, hi):
                seen.add((lo, hi))
                return 2
        for test in TESTS:
            test(35, 1, Spy())
        self.assertEqual(seen, {(1, 35)})


class TestLiarStructure(unittest.TestCase):
    """Утверждения из методички о множествах оснований, для которых N «выглядит простым»."""

    def test_fermat_liars_form_subgroup_of_half_size(self):
        """Не число Кармайкла: лжецов не больше phi(N)/2; число Кармайкла: ровно phi(N)."""
        for N in ODD_COMPOSITES:
            L = liars(N, fermat_round)
            if is_carmichael(N):
                self.assertEqual(len(L), phi(N), N)
            else:
                self.assertLessEqual(len(L), phi(N) // 2, N)
            self.assertTrue(all(gcd(a, N) == 1 for a in L))

    def test_carmichael_numbers(self):
        self.assertEqual([n for n in range(3, 3000, 2) if is_carmichael(n)],
                         [561, 1105, 1729, 2465, 2821])
        for N in (561, 1105, 1729, 2465, 2821):
            self.assertEqual(len(liars(N, fermat_round)), phi(N))

    def test_detection_probability_for_carmichael(self):
        """P0 = 1 - phi(N)/(N-1) для чисел Кармайкла (слайд про ситуацию 3)."""
        N = 561
        p0 = 1 - len(liars(N, fermat_round)) / (N - 1)
        self.assertAlmostEqual(p0, 1 - phi(N) / (N - 1))
        self.assertAlmostEqual(p0, 1 - 320 / 560)

    def test_euler_liars_at_most_half(self):
        """Соловей - Штрассен: лжецов не больше phi(N)/2 (подгруппа, не вся группа)."""
        for N in ODD_COMPOSITES:
            self.assertLessEqual(len(liars(N, solovay_strassen_round)), phi(N) // 2, N)

    def test_strong_liars_at_most_quarter(self):
        """Миллер - Рабин: лжецов не больше phi(N)/4 (единственное исключение N = 9)."""
        for N in ODD_COMPOSITES:
            L = liars(N, miller_rabin_round)
            if N == 9:
                self.assertEqual(L, [1, 8])
            else:
                self.assertLessEqual(len(L), phi(N) // 4, N)

    def test_inclusion_of_liar_sets(self):
        """Сильные лжецы входят в эйлеровы, а те - в лжецов Ферма."""
        for N in ODD_COMPOSITES:
            F = set(liars(N, fermat_round))
            S = set(liars(N, solovay_strassen_round))
            M = set(liars(N, miller_rabin_round))
            self.assertTrue(M <= S <= F, N)

    def test_primes_have_no_witnesses(self):
        for p in PRIMES[:30]:
            for single in ROUNDS:
                self.assertEqual(len(liars(p, single)), p - 1)


class TestCarmichaelLargeNumbers(unittest.TestCase):
    def test_fermat_fooled_but_others_are_not(self):
        """Большое число Кармайкла: Ферма почти никогда не видит составность, остальные видят."""
        rnd = random.Random(7)
        N, factors = chernick_carmichael(40, rnd)
        self.assertTrue(all((N - 1) % (p - 1) == 0 for p in factors))   # критерий Корселта
        for a in (2, 3, 5, 12345):
            self.assertEqual(pow(a, N - 1, N), 1)
        self.assertTrue(fermat_test(N, 40, rnd))                  # ошибочное «неизвестно»
        self.assertFalse(solovay_strassen_test(N, 40, rnd))
        self.assertFalse(miller_rabin_test(N, 40, rnd))

    def test_detection_rates_respect_bounds(self):
        """Для одного раунда: Соловей - Штрассен >= 1/2, Миллер - Рабин >= 3/4 (с запасом на шум)."""
        rnd = random.Random(8)
        N, _ = chernick_carmichael(40, rnd)
        n = 2000
        rate = lambda single: sum(not single(N, rnd.randrange(1, N)) for _ in range(n)) / n
        self.assertLess(rate(fermat_round), 0.01)
        self.assertGreater(rate(solovay_strassen_round), 0.45)
        self.assertGreater(rate(miller_rabin_round), 0.70)

    def test_large_composites_and_primes(self):
        rnd = random.Random(9)
        p, q = 2**89 - 1, 2**107 - 1
        for test in (solovay_strassen_test, miller_rabin_test):
            self.assertTrue(test(p, 20, rnd))
            self.assertFalse(test(p * q, 20, rnd))
        self.assertFalse(fermat_test(p * q, 20, rnd))
        self.assertFalse(miller_rabin_test(2**127 + 1, 20, rnd))   # делится на 3


if __name__ == "__main__":
    unittest.main()
