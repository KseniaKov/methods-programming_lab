"""Тесты алгоритмов Монтгомери. Запуск: `pytest` или `python -m unittest`."""
import random
import unittest
from math import gcd

from montgomery import (Montgomery, ext_gcd, inv_mod,
                        montgomery_mul, montgomery_pow)

BASES = [2, 3, 10, 16, 256, 2**16]


def random_cases(count=300, bits=200, seed=1):
    """Случайные пары (N, b) с gcd(N, b) = 1."""
    rnd = random.Random(seed)
    while count:
        b = rnd.choice(BASES)
        N = rnd.randrange(3, 2**bits)
        if gcd(N, b) == 1:
            count -= 1
            yield rnd, N, b


class TestEuclid(unittest.TestCase):
    def test_ext_gcd(self):
        rnd = random.Random(0)
        for _ in range(500):
            a, b = rnd.randrange(1, 10**20), rnd.randrange(1, 10**20)
            g, u, v = ext_gcd(a, b)
            self.assertEqual(g, gcd(a, b))
            self.assertEqual(a * u + b * v, g)

    def test_inv_mod(self):
        self.assertEqual(inv_mod(3, 7), 5)
        with self.assertRaises(ValueError):
            inv_mod(6, 9)


class TestParameters(unittest.TestCase):
    def test_linear_decomposition(self):
        """R*R' - N*N' = 1, 0 < R' < N, 0 < N' < R."""
        for _, N, b in random_cases():
            M = Montgomery(N, b)
            self.assertEqual(M.R * M.Rinv - N * M.Np, 1)
            self.assertTrue(0 < M.Rinv < N and 0 < M.Np < M.R)
            self.assertTrue(M.R > N)

    def test_invalid_modulus(self):
        with self.assertRaises(ValueError):
            Montgomery(10)            # чётный N при b = 2
        with self.assertRaises(ValueError):
            Montgomery(1)
        with self.assertRaises(ValueError):
            Montgomery(9, b=3)


class TestReduction(unittest.TestCase):
    def test_redc_theorem(self):
        """Алгоритм 1: результат = x R^{-1} mod N для 0 <= x < RN."""
        for rnd, N, b in random_cases():
            M = Montgomery(N, b)
            x = rnd.randrange(M.R * N)
            self.assertEqual(M.redc(x), x * M.Rinv % N)
            self.assertTrue(0 <= M.redc(x) < N)


class TestMultiplication(unittest.TestCase):
    def test_mont_product(self):
        """Алгоритм 2: phi_R(xy) = x y R^{-1} mod N."""
        for rnd, N, b in random_cases():
            M = Montgomery(N, b)
            x, y = rnd.randrange(N), rnd.randrange(N)
            self.assertEqual(M.mont_product(x, y), x * y * M.Rinv % N)

    def test_mul_mod(self):
        for rnd, N, b in random_cases():
            M = Montgomery(N, b)
            x, y = rnd.randrange(N), rnd.randrange(N)
            self.assertEqual(M.mul_mod(x, y), x * y % N)

    def test_mont_mul(self):
        """x * y = x y R mod N."""
        for rnd, N, b in random_cases():
            M = Montgomery(N, b)
            x, y = rnd.randrange(N), rnd.randrange(N)
            self.assertEqual(M.mont_mul(x, y), x * y * M.R % N)

    def test_roundtrip(self):
        for rnd, N, b in random_cases():
            M = Montgomery(N, b)
            x = rnd.randrange(N)
            self.assertEqual(M.to_mont(x), x * M.R % N)
            self.assertEqual(M.from_mont(M.to_mont(x)), x)

    def test_edge_values(self):
        M = Montgomery(97)
        for x in (0, 1, 96):
            for y in (0, 1, 96):
                self.assertEqual(M.mul_mod(x, y), x * y % 97)

    def test_helper(self):
        self.assertEqual(montgomery_mul(12345, 67890, 1000003),
                         12345 * 67890 % 1000003)


class TestPower(unittest.TestCase):
    def test_pow_mod(self):
        for rnd, N, b in random_cases():
            M = Montgomery(N, b)
            x, k = rnd.randrange(N), rnd.randrange(0, 500)
            self.assertEqual(M.pow_mod(x, k), pow(x, k, N))

    def test_pow_naive(self):
        for rnd, N, b in random_cases(count=100):
            M = Montgomery(N, b)
            x, k = rnd.randrange(N), rnd.randrange(1, 50)
            self.assertEqual(M.pow_naive(x, k), pow(x, k, N))

    def test_zero_exponent(self):
        M = Montgomery(101)
        self.assertEqual(M.pow_mod(7, 0), 1)
        self.assertEqual(M.pow_naive(7, 0), 1)

    def test_negative_exponent(self):
        with self.assertRaises(ValueError):
            Montgomery(101).pow_mod(7, -1)

    def test_fermat(self):
        p = 2**127 - 1                      # простое число Мерсенна
        self.assertEqual(montgomery_pow(123456789, p - 1, p), 1)

    def test_large_modulus(self):
        rnd = random.Random(7)
        N = rnd.getrandbits(512) | 1 | (1 << 511)
        x, k = rnd.getrandbits(500), rnd.getrandbits(512)
        self.assertEqual(montgomery_pow(x, k, N), pow(x, k, N))
        self.assertEqual(montgomery_pow(x, k, N, b=2**16), pow(x, k, N))


if __name__ == "__main__":
    unittest.main()
