"""
Алгоритмы Монтгомери в кольце вычетов Z_N (по методичке «Сложность
арифметических операций в кольцах вычетов»).

Обозначения:
    b      - основание системы счисления (по умолчанию 2), gcd(N, b) = 1
    n      - число цифр N в системе по основанию b, R = b^n
    R'     - R^{-1} mod N,  N' = -N^{-1} mod R,  R*R' - N*N' = 1
    phi_R(x) = x * R^{-1} mod N
    x * y    = x*y*R^{-1} mod N        (умножение по Монтгомери)
"""
from math import gcd


def ext_gcd(a: int, b: int):
    """Расширенный алгоритм Евклида: (g, u, v), g = a*u + b*v."""
    u0, v0, u1, v1 = 1, 0, 0, 1
    while b:
        q = a // b
        a, b = b, a - q * b
        u0, u1 = u1, u0 - q * u1
        v0, v1 = v1, v0 - q * v1
    return a, u0, v0


def inv_mod(a: int, m: int) -> int:
    g, u, _ = ext_gcd(a % m, m)
    if g != 1:
        raise ValueError("обратного элемента не существует")
    return u % m


class Montgomery:
    def __init__(self, N: int, b: int = 2):
        if N <= 1:
            raise ValueError("N должно быть > 1")
        if gcd(N, b) != 1:
            raise ValueError("N должно быть взаимно просто с основанием b "
                             "(при b = 2 модуль должен быть нечётным)")
        self.N, self.b = N, b
        # n = L_b(N): число цифр N по основанию b
        n, t = 0, N
        while t:
            t //= b
            n += 1
        self.n = n
        self.R = b ** n
        # R R' - N N' = 1  =>  R' = R^{-1} mod N,  N' = -N^{-1} mod R
        self.Rinv = inv_mod(self.R, N)
        self.Np = (-inv_mod(N, self.R)) % self.R
        self.Np_b = self.Np % b          # N' mod b для алгоритма 2
        self.R2 = self.R * self.R % N    # R^2 mod N, считается один раз

    # ---------- Алгоритм 1: phi_R(x) для 0 <= x < R*N ----------
    def redc(self, x: int) -> int:
        N, R = self.N, self.R
        m = (x * self.Np) % R              # шаг 1
        t = (x + m * N) // R               # шаг 2 (делится нацело)
        return t - N if t >= N else t      # шаг 3

    # ---------- Алгоритм 2: phi_R(x*y) без полного умножения ----------
    def mont_product(self, x: int, y: int) -> int:
        """Возвращает x*y*R^{-1} mod N для 0 <= x, y < N."""
        N, b, n = self.N, self.b, self.n
        z = 0
        for _ in range(n):
            xi = x % b                      # очередная цифра x
            x //= b
            u = (z + xi * y) % b            # шаг 1
            v = (u * self.Np_b) % b         # шаг 2
            z = (z + xi * y + v * N) // b   # z_{i+1}
        return z - N if z >= N else z

    # ---------- Переход в/из представления Монтгомери ----------
    def to_mont(self, x: int) -> int:
        """phi_R^{-1}(x) = xR mod N  =  phi_R(x * R^2 mod N)  (замечание 2)."""
        return self.mont_product(x % self.N, self.R2)

    def from_mont(self, x: int) -> int:
        """phi_R(x) = x R^{-1} mod N."""
        return self.mont_product(x, 1)

    # ---------- Умножение по модулю N ----------
    def mul_mod(self, x: int, y: int) -> int:
        """xy mod N: z = phi_R(xy), затем phi_R^{-1}(z)."""
        z = self.mont_product(x % self.N, y % self.N)   # xyR^{-1}
        return self.to_mont(z)                          # xy

    # ---------- Умножение по Монтгомери x*y = xyR mod N ----------
    def mont_mul(self, x: int, y: int) -> int:
        a = self.to_mont(x)                             # xR
        b = self.to_mont(y)                             # yR
        return self.mont_product(a, b)                  # xyR

    # ---------- Возведение в степень ----------
    def pow_naive(self, x: int, k: int) -> int:
        """Линейный вариант из методички: O(k n^2)."""
        y1 = self.to_mont(x)
        y = y1
        for _ in range(k - 1):
            y = self.mont_product(y, y1)     # y_i = x^i R
        return self.from_mont(y) if k >= 1 else 1 % self.N

    def pow_mod(self, x: int, k: int) -> int:
        """Бинарное возведение в степень (замечание 3): O(log k * n^2)."""
        if k < 0:
            raise ValueError("k должно быть неотрицательным")
        result = self.to_mont(1)             # 1 в представлении Монтгомери = R mod N
        base = self.to_mont(x)               # xR mod N
        while k:
            if k & 1:
                result = self.mont_product(result, base)
            base = self.mont_product(base, base)
            k >>= 1
        return self.from_mont(result)        # убираем множитель R


def montgomery_pow(x: int, k: int, N: int, b: int = 2) -> int:
    return Montgomery(N, b).pow_mod(x, k)


def montgomery_mul(x: int, y: int, N: int, b: int = 2) -> int:
    return Montgomery(N, b).mul_mod(x, y)


if __name__ == "__main__":
    N = 1_000_000_007
    M = Montgomery(N)
    print(f"N = {N}, b = {M.b}, n = {M.n}, R = {M.R}")
    print("123456 * 654321 mod N =", M.mul_mod(123456, 654321))
    print("5^(10^18) mod N       =", M.pow_mod(5, 10**18))
    assert M.mul_mod(123456, 654321) == 123456 * 654321 % N
    assert M.pow_mod(5, 10**18) == pow(5, 10**18, N)
    print("Проверка со встроенным pow: OK")
