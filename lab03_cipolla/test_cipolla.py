"""Тесты алгоритма Чипполы. Запуск: `pytest` или `python -m unittest`."""
import random
import unittest

from cipolla import (cipolla, fp2_mul, fp2_pow, is_probable_prime,
                     legendre_euler, sqrt_3mod4, sqrt_mod)

SMALL_PRIMES = [p for p in range(3, 300) if all(p % d for d in range(2, int(p ** 0.5) + 1))]
# простые с большой степенью двойки в p - 1 (трудный случай для p = 1 mod 8)
HARD_PRIMES = [65537,                 # 2^16 + 1
               998244353,             # 119 * 2^23 + 1
               2013265921,            # 15 * 2^27 + 1
               2**64 - 2**32 + 1]     # p - 1 делится на 2^32
MERSENNE = [2**127 - 1, 2**521 - 1, 2**607 - 1]


def poly_mul(f, g, p):
    """Умножение многочленов (списки коэффициентов от младшего) в Z_p[y]."""
    res = [0] * (len(f) + len(g) - 1)
    for i, c in enumerate(f):
        for j, d in enumerate(g):
            res[i + j] = (res[i + j] + c * d) % p
    return res


def poly_mod(f, m, p):
    """Остаток от деления f на унитарный многочлен m в Z_p[y]."""
    f = f[:]
    while len(f) >= len(m):
        c, shift = f[-1], len(f) - len(m)
        for i, d in enumerate(m):
            f[shift + i] = (f[shift + i] - c * d) % p
        f.pop()
    return f + [0] * (len(m) - 1 - len(f))


def brute_roots(a, p):
    return sorted(x for x in range(p) if (x * x - a) % p == 0)


class TestFieldArithmetic(unittest.TestCase):
    def test_mul_matches_polynomial_division(self):
        """Умножение пар = умножение многочленов и остаток по f(y) = y^2 - b y + a."""
        rnd = random.Random(1)
        for _ in range(500):
            p = rnd.choice(SMALL_PRIMES)
            a, b = rnd.randrange(1, p), rnd.randrange(p)
            x = (rnd.randrange(p), rnd.randrange(p))
            y = (rnd.randrange(p), rnd.randrange(p))
            f = [a, (-b) % p, 1]                         # y^2 - b y + a
            expected = poly_mod(poly_mul(list(x), list(y), p), f, p)
            self.assertEqual(list(fp2_mul(x, y, a, b, p)), expected)

    def test_mul_commutative_associative(self):
        rnd = random.Random(2)
        p, a, b = 10007, 5, 3
        for _ in range(200):
            x, y, z = [(rnd.randrange(p), rnd.randrange(p)) for _ in range(3)]
            self.assertEqual(fp2_mul(x, y, a, b, p), fp2_mul(y, x, a, b, p))
            self.assertEqual(fp2_mul(fp2_mul(x, y, a, b, p), z, a, b, p),
                             fp2_mul(x, fp2_mul(y, z, a, b, p), a, b, p))

    def test_pow_matches_repeated_multiplication(self):
        p, a, b = 101, 7, 4
        x = (3, 5)
        acc = (1, 0)
        for e in range(40):
            self.assertEqual(fp2_pow(x, e, a, b, p), acc)
            acc = fp2_mul(acc, x, a, b, p)


class TestTheory(unittest.TestCase):
    def good_b(self, a, p):
        return [b for b in range(p) if legendre_euler(b * b - 4 * a, p) == -1]

    def test_number_of_suitable_b(self):
        """Для вычета a подходит ровно (p-1)/2 значений b."""
        for p in SMALL_PRIMES:
            squares = {x * x % p for x in range(1, p)}
            for a in squares:
                self.assertEqual(len(self.good_b(a, p)), (p - 1) // 2, (a, p))

    def test_frobenius_and_norm(self):
        """y^p = b - y и y^(p+1) = a в Z_p[y]/(f), если (b^2-4a / p) = -1."""
        rnd = random.Random(3)
        for p in SMALL_PRIMES:
            a = rnd.choice(sorted({x * x % p for x in range(1, p)}))
            for b in self.good_b(a, p)[:5]:
                self.assertEqual(fp2_pow((0, 1), p, a, b, p), (b % p, p - 1))
                self.assertEqual(fp2_pow((0, 1), p + 1, a, b, p), (a % p, 0))

    def test_f_is_irreducible(self):
        """При подходящем b у f(y) нет корней в Z_p."""
        for p in SMALL_PRIMES[:20]:
            a = next(x * x % p for x in range(1, p))
            for b in self.good_b(a, p):
                self.assertFalse(any((y * y - b * y + a) % p == 0 for y in range(p)))


class TestCipolla(unittest.TestCase):
    def test_all_residues_small_primes(self):
        """Все a для всех малых p: корень верен, множество корней совпадает с перебором."""
        for p in SMALL_PRIMES:
            for a in range(0, p):
                if a != 0 and legendre_euler(a, p) != 1:
                    continue
                x0 = cipolla(a, p, random.Random(p * 1000 + a))
                self.assertEqual((x0 * x0 - a) % p, 0, (a, p))
                self.assertTrue(0 <= x0 < p)
                self.assertEqual(sqrt_mod(a, p, random.Random(1)),
                                 tuple(brute_roots(a, p)), (a, p))

    def test_nonresidue_raises(self):
        for p in SMALL_PRIMES:
            for a in range(1, p):
                if legendre_euler(a, p) == -1:
                    with self.assertRaises(ValueError):
                        cipolla(a, p)

    def test_zero(self):
        self.assertEqual(cipolla(0, 13), 0)
        self.assertEqual(cipolla(26, 13), 0)
        self.assertEqual(sqrt_mod(0, 13), (0,))

    def test_any_integer_a(self):
        """Отрицательные a и a >= p приводятся по модулю p."""
        p = 10007
        for a in (-3, -10006, 10007 + 4, 5 * p + 9, -p * 7 + 16):
            if legendre_euler(a, p) == 1:
                x0 = cipolla(a, p, random.Random(5))
                self.assertEqual((x0 * x0 - a) % p, 0)

    def test_two_roots_are_opposite(self):
        for p in (7, 13, 101, 10007):
            a = 4
            r = sqrt_mod(a, p)
            self.assertEqual(r, tuple(sorted({2, p - 2})))
            self.assertEqual(sum(r) % p, 0)

    def test_primes_with_large_two_adic_valuation(self):
        rnd = random.Random(4)
        for p in HARD_PRIMES:
            self.assertTrue(is_probable_prime(p))
            for _ in range(50):
                a = rnd.randrange(1, p) ** 2 % p
                x0 = cipolla(a, p, rnd)
                self.assertEqual(x0 * x0 % p, a)

    def test_residue_classes_of_p(self):
        """Работает при p = 3, 5, 1 (mod 8)."""
        rnd = random.Random(6)
        seen = set()
        for p in (p for p in range(10**6, 10**6 + 500) if is_probable_prime(p)):
            seen.add(p % 8)
            a = rnd.randrange(1, p) ** 2 % p
            x0 = cipolla(a, p, rnd)
            self.assertEqual(x0 * x0 % p, a)
        self.assertEqual(seen, {1, 3, 5, 7})

    def test_large_primes(self):
        rnd = random.Random(7)
        for p in MERSENNE + [2**255 - 19]:
            for _ in range(5):
                a = rnd.randrange(1, p) ** 2 % p
                x0 = cipolla(a, p, rnd)
                self.assertEqual(x0 * x0 % p, a)

    def test_agrees_with_formula_for_3_mod_4(self):
        rnd = random.Random(8)
        for p in (p for p in SMALL_PRIMES if p % 4 == 3):
            for a in range(1, p):
                if legendre_euler(a, p) == 1:
                    x = sqrt_3mod4(a, p)
                    self.assertIn(cipolla(a, p, rnd), (x, p - x))
        with self.assertRaises(ValueError):
            sqrt_3mod4(4, 13)               # p = 1 (mod 4)

    def test_invalid_modulus(self):
        for p in (-7, 0, 1, 2, 4, 9, 15, 561):
            with self.assertRaises(ValueError):
                cipolla(4, p)

    def test_reproducible_with_seed(self):
        i1, i2 = {}, {}
        x1 = cipolla(10, 13, random.Random(1), i1)
        x2 = cipolla(10, 13, random.Random(1), i2)
        self.assertEqual((x1, i1), (x2, i2))
        self.assertEqual(legendre_euler(i1["discriminant"], 13), -1)
        self.assertEqual(i1["f"], (1, (-i1["b"]) % 13, 10))

    def test_average_number_of_attempts(self):
        """Среднее число попыток около 2p/(p-1) ~ 2."""
        p, rnd = 10007, random.Random(9)
        total, runs = 0, 3000
        for _ in range(runs):
            info = {}
            a = rnd.randrange(1, p) ** 2 % p
            cipolla(a, p, rnd, info)
            total += info["attempts"]
        mean = total / runs
        self.assertAlmostEqual(mean, 2 * p / (p - 1), delta=0.12)


class TestAuxiliary(unittest.TestCase):
    def test_legendre_euler(self):
        for p in SMALL_PRIMES:
            squares = {x * x % p for x in range(1, p)}
            for a in range(-p, 2 * p):
                expected = 0 if a % p == 0 else (1 if a % p in squares else -1)
                self.assertEqual(legendre_euler(a, p), expected)

    def test_primality(self):
        small = {p for p in range(2, 5000) if all(p % d for d in range(2, int(p ** 0.5) + 1))}
        for n in range(-5, 5000):
            self.assertEqual(is_probable_prime(n), n in small)
        self.assertFalse(is_probable_prime(561))        # число Кармайкла
        for p in MERSENNE:
            self.assertTrue(is_probable_prime(p))


if __name__ == "__main__":
    unittest.main()
