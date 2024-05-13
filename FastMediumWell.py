import numpy as np
import logging
logger = logging.getLogger(__name__)

def spheroid(eta, m, alpha):
    """
    Calculates spheriodal wave functions. See Schwab 1984 for details.
    This implementation follows MIRIAD's grid.for subroutine.
    """
    twoalp = 2 * alpha
    if np.abs(eta) > 1:
        logger.debug('bad eta value!')
    if (twoalp < 1 or twoalp > 4):
        logger.debug('bad alpha value!')
    if (m < 4 or m > 8):
        logger.debug('bad width value!')
    etalim = np.float32([1., 1., 0.75, 0.775, 0.775])
    nnum = np.int8([5, 7, 5, 5, 6])
    ndenom = np.int8([3, 2, 3, 3, 3])
    p = np.float32(
            [
                [[5.613913E-2, -3.019847E-1, 6.256387E-1,
                  -6.324887E-1, 3.303194E-1, 0.0, 0.0],
                 [6.843713E-2, -3.342119E-1, 6.302307E-1,
                  -5.829747E-1, 2.765700E-1, 0.0, 0.0],
                 [8.203343E-2, -3.644705E-1, 6.278660E-1,
                  -5.335581E-1, 2.312756E-1, 0.0, 0.0],
                 [9.675562E-2, -3.922489E-1, 6.197133E-1,
                  -4.857470E-1, 1.934013E-1, 0.0, 0.0],
                 [1.124069E-1, -4.172349E-1, 6.069622E-1,
                  -4.405326E-1, 1.618978E-1, 0.0, 0.0]
                 ],
                [[8.531865E-4, -1.616105E-2, 6.888533E-2,
                  -1.109391E-1, 7.747182E-2, 0.0, 0.0],
                 [2.060760E-3, -2.558954E-2, 8.595213E-2,
                  -1.170228E-1, 7.094106E-2, 0.0, 0.0],
                 [4.028559E-3, -3.697768E-2, 1.021332E-1,
                  -1.201436E-1, 6.412774E-2, 0.0, 0.0],
                 [6.887946E-3, -4.994202E-2, 1.168451E-1,
                  -1.207733E-1, 5.744210E-2, 0.0, 0.0],
                 [1.071895E-2, -6.404749E-2, 1.297386E-1,
                  -1.194208E-1, 5.112822E-2, 0.0, 0.0]
                 ]
            ])
    q = np.float32(
            [
                [[1., 9.077644E-1, 2.535284E-1],
                 [1., 8.626056E-1, 2.291400E-1],
                 [1., 8.212018E-1, 2.078043E-1],
                 [1., 7.831755E-1, 1.890848E-1],
                 [1., 7.481828E-1, 1.726085E-1]
                 ],
                [[1., 1.101270, 3.858544E-1],
                 [1., 1.025431, 3.337648E-1],
                 [1., 9.599102E-1, 2.918724E-1],
                 [1., 9.025276E-1, 2.575337E-1],
                 [1., 8.517470E-1, 2.289667E-1]
                 ]
            ])
    i = m - 4
    if (np.abs(eta) > etalim[i]):
        ip = 1
        x = eta * eta - 1
    else:
        ip = 0
        x = eta * eta - etalim[i] * etalim[i]
        # numerator via Horner's rule
    mnp = np.int32(nnum[i] - 1)
    twoalp = np.int32(twoalp)
    num = p[ip, twoalp, mnp]
    for j in np.arange(mnp):
        num = num * x + p[ip, twoalp, mnp - 1 - j]
        # denominator via Horner's rule
    nq = ndenom[i] - 1
    denom = q[ip, twoalp, nq]
    for j in np.arange(nq):
        denom = denom * x + q[ip, twoalp, nq - 1 - j]
    return np.float32(num / denom)
def gcf(n, width):
    """
    Create table with spheroidal gridding function, C
    This implementation follows MIRIAD's grid.for subroutine.
    """
    alpha = 1.
    j = 2 * alpha
    p = 0.5 * j
    phi = np.zeros(n, dtype=np.float32)
    for i in np.arange(n):
        x = np.float32(2 * i - (n - 1)) / (n - 1)
        phi[i] = (np.sqrt(1 - x * x) ** j) * spheroid(x, width, p)
    return phi

def corrfun( n, width):
    """
    Create gridding correction function, c
    This implementation follows MIRIAD's grid.for subroutine.
    """
    alpha = 1.
    dx = 2. / n
    i0 = n / 2 + 1
    phi = np.zeros(n, dtype=np.float32)
    for i in np.arange(n):
        x = (i - i0 + 1) * dx
        phi[i] = spheroid(x, width, alpha)
    return phi

import cupy

