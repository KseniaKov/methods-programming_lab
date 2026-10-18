"""Тесты символов Лежандра и Якоби. Запуск: `pytest` или `python -m unittest`."""
import random
import unittest
from math import gcd

from legendre_jacobi import (_factorize, is_probable_prime, jacobi, legendre,
                             legendre_euler, legendre_factorization,
                             legendre_gauss, num_solutions, sign_minus_one,
                             sign_reciprocity, sign_two)

SMALL_PRIMES = [p for p in range(3, 300) if all(p % d for d in range(2, int(p ** 0.5) + 1))]
MERSENNE = [2**127 - 1, 2**521 - 1, 2**607 - 1]        # простые числа Мерсенна


def brute_legendre(a, p):
    """Определение: перебор квадратов по модулю p."""
    a %= p
    if a == 0:
        return 0
    return 1 if a in {x * x % p for x in range(1, p)} else -1


def jacobi_by_definition(a, n):
    """Определение 2.4: произведение символов Лежандра по разложению n."""
    result = 1
    for q, e in _factorize(n).items():
        result *= brute_legendre(a, q) ** e
    return result


class TestSigns(unittest.TestCase):
    def test_signs_match_formulas(self):
        for n in range(1, 400, 2):
            self.assertEqual(sign_minus_one(n), (-1) ** ((n - 1) // 2))
            self.assertEqual(sign_two(n), (-1) ** ((n * n - 1) // 8))
            for a in range(1, 60, 2):
                self.assertEqual(sign_reciprocity(a, n),
                                 (-1) ** ((a - 1) // 2 * ((n - 1) // 2)))


class TestLegendre(unittest.TestCase):
    def test_against_definition_all_residues(self):
        """Все a для малых простых p: сравнение с перебором квадратов."""
        for p in SMALL_PRIMES:
            squares = {x * x % p for x in range(1, p)}
            for a in range(-p, 2 * p):
                expected = 0 if a % p == 0 else (1 if a % p in squares else -1)
                self.assertEqual(legendre(a, p), expected, (a, p))

    def test_against_euler_criterion(self):
        rnd = random.Random(1)
        primes = [p for p in range(10**5, 10**5 + 2000) if is_probable_prime(p)]
        for _ in range(1000):
            p, a = rnd.choice(primes), rnd.randrange(-10**6, 10**6)
            self.assertEqual(legendre(a, p), legendre_euler(a, p))

    def test_against_gauss_criterion(self):
        for p in SMALL_PRIMES[:25]:
            for a in range(-p, 2 * p):
                self.assertEqual(legendre(a, p), legendre_gauss(a, p))

    def test_against_factorization_algorithm(self):
        rnd = random.Random(2)
        primes = [p for p in range(10**6, 10**6 + 3000) if is_probable_prime(p)]
        for _ in range(300):
            p, a = rnd.choice(primes), rnd.randrange(10**7)
            self.assertEqual(legendre(a, p), legendre_factorization(a, p))

    def test_known_values(self):
        self.assertEqual(legendre(1001, 9907), -1)     # классический пример
        self.assertEqual(legendre(2, 7), 1)            # 3^2 = 2 (mod 7)
        self.assertEqual(legendre(3, 7), -1)
        self.assertEqual(legendre(5, 11), 1)           # 4^2 = 5 (mod 11)
        self.assertEqual(legendre(0, 13), 0)
        self.assertEqual(legendre(26, 13), 0)

    def test_minus_one_and_two(self):
        for p in SMALL_PRIMES:
            self.assertEqual(legendre(-1, p), (-1) ** ((p - 1) // 2))
            self.assertEqual(legendre(2, p), (-1) ** ((p * p - 1) // 8))

    def test_counts_of_residues(self):
        """Следствие 1: (p-1)/2 вычетов и (p-1)/2 невычетов."""
        for p in SMALL_PRIMES:
            values = [legendre(a, p) for a in range(1, p)]
            self.assertEqual(values.count(1), (p - 1) // 2)
            self.assertEqual(values.count(-1), (p - 1) // 2)

    def test_num_solutions(self):
        """a=0 - одно решение, вычет - два, невычет - ни одного."""
        for p in (7, 11, 13, 101):
            for a in range(p):
                real = sum(1 for x in range(p) if (x * x - a) % p == 0)
                self.assertEqual(num_solutions(a, p), real)

    def test_large_mersenne_primes(self):
        rnd = random.Random(3)
        for p in MERSENNE:
            for _ in range(20):
                a = rnd.getrandbits(p.bit_length() + 10)
                self.assertEqual(legendre(a, p), legendre_euler(a, p))

    def test_huge_numbers(self):
        """Числа длиннее 4300 десятичных цифр: без трассировки строки не строятся."""
        rnd = random.Random(8)
        n = rnd.getrandbits(40000) | (1 << 39999) | 1
        a = rnd.getrandbits(40000)
        self.assertIn(jacobi(a, n), (-1, 0, 1))
        self.assertEqual(jacobi(a, n), jacobi(a % n, n))

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            legendre(3, 2)
        with self.assertRaises(ValueError):
            legendre(3, 1)
        with self.assertRaises(ValueError):
            legendre(3, 15)                     # составное
        self.assertEqual(legendre(2, 15, check_prime=False), jacobi(2, 15))


class TestJacobi(unittest.TestCase):
    def test_against_definition(self):
        """(a/n) = произведение (a/p_i)^k_i по каноническому разложению n."""
        for n in range(1, 120, 2):
            for a in range(-n, 2 * n):
                if gcd(a, n) == 1:
                    self.assertEqual(jacobi(a, n), jacobi_by_definition(a, n), (a, n))

    def test_zero_when_not_coprime(self):
        for n in range(3, 200, 2):
            for a in range(0, 2 * n):
                if gcd(a, n) > 1:
                    self.assertEqual(jacobi(a, n), 0)

    def test_n_equals_one(self):
        self.assertEqual(jacobi(5, 1), 1)

    def test_property_periodicity(self):
        rnd = random.Random(4)
        for _ in range(500):
            n = rnd.randrange(1, 10**9, 2)
            a = rnd.randrange(-10**9, 10**9)
            self.assertEqual(jacobi(a, n), jacobi(a + n * rnd.randrange(-9, 9), n))

    def test_property_multiplicative_in_a(self):
        rnd = random.Random(5)
        for _ in range(500):
            n = rnd.randrange(3, 10**9, 2)
            a, c = rnd.randrange(10**9), rnd.randrange(10**9)
            self.assertEqual(jacobi(a * c, n), jacobi(a, n) * jacobi(c, n))

    def test_property_multiplicative_in_n(self):
        rnd = random.Random(6)
        for _ in range(500):
            b, c = rnd.randrange(3, 10**6, 2), rnd.randrange(3, 10**6, 2)
            a = rnd.randrange(10**9)
            self.assertEqual(jacobi(a, b * c), jacobi(a, b) * jacobi(a, c))

    def test_property_reciprocity(self):
        rnd = random.Random(7)
        for _ in range(500):
            a, b = rnd.randrange(3, 10**9, 2), rnd.randrange(3, 10**9, 2)
            if gcd(a, b) != 1:
                continue
            self.assertEqual(jacobi(a, b),
                             jacobi(b, a) * (-1) ** ((a - 1) // 2 * ((b - 1) // 2)))

    def test_minus_one_and_two(self):
        for n in range(1, 500, 2):
            self.assertEqual(jacobi(-1, n), (-1) ** ((n - 1) // 2))
            self.assertEqual(jacobi(2, n), (-1) ** ((n * n - 1) // 8))

    def test_one_does_not_mean_residue(self):
        """Замечание из методички: (a/n) = 1 не означает, что x^2 = a (mod n) разрешимо."""
        self.assertEqual(jacobi(2, 15), 1)
        self.assertFalse(any((x * x - 2) % 15 == 0 for x in range(15)))

    def test_prime_modulus_equals_legendre(self):
        for p in SMALL_PRIMES:
            for a in range(p):
                self.assertEqual(jacobi(a, p), legendre(a, p))

    def test_invalid_modulus(self):
        for n in (0, -3, 2, 10):
            with self.assertRaises(ValueError):
                jacobi(3, n)

    def test_trace(self):
        steps = []
        self.assertEqual(jacobi(1001, 9907, steps), -1)
        self.assertGreater(len(steps), 3)
        self.assertTrue(steps[0].startswith("вычисляем"))


    def test_step1_symmetric_residue(self):
        """Шаг 1: a заменяется на вычет b с |b| < n/2 (здесь 10 -> -1 по модулю 11)."""
        steps = []
        jacobi(10, 11, steps)
        self.assertIn("шаг 1: (10/11) = (-1/11)", steps)
        self.assertIn("шаг 3: (-1/11) = -1", steps)


class TestAuxiliary(unittest.TestCase):
    def test_factorize(self):
        for n in range(1, 2000):
            prod = 1
            for q, e in _factorize(n).items():
                self.assertTrue(is_probable_prime(q))
                prod *= q ** e
            self.assertEqual(prod, n)

    def test_primality(self):
        small = {p for p in range(2, 5000) if all(p % d for d in range(2, int(p ** 0.5) + 1))}
        for n in range(-5, 5000):
            self.assertEqual(is_probable_prime(n), n in small)
        for p in MERSENNE:
            self.assertTrue(is_probable_prime(p))
        self.assertFalse(is_probable_prime(2**127 + 1))
        self.assertFalse(is_probable_prime(561))        # число Кармайкла


if __name__ == "__main__":
    unittest.main()
