import numpy as np, soundfile as sf, os, sys
from scipy import signal

SR = 44100
rng = np.random.default_rng(7)
out = sys.argv[1]
os.makedirs(out, exist_ok=True)


def t(d):
    return np.arange(int(SR * d)) / SR


def noise(d):
    return rng.standard_normal(int(SR * d))


def bp(x, lo, hi, order=4):
    return signal.sosfilt(signal.butter(order, [lo, hi], 'band', fs=SR, output='sos'), x)


def lp(x, f, order=4):
    return signal.sosfilt(signal.butter(order, f, 'low', fs=SR, output='sos'), x)


def hp(x, f, order=4):
    return signal.sosfilt(signal.butter(order, f, 'high', fs=SR, output='sos'), x)


def env_exp(d, tau):
    return np.exp(-t(d) / tau)


def norm(x, peak=0.89):
    x = x - np.mean(x)
    return x / (np.max(np.abs(x)) + 1e-9) * peak


def fade(x, a=0.004, b=0.02):
    n = len(x)
    na, nb = int(SR * a), int(SR * b)
    x = x.copy()
    x[:na] *= np.linspace(0, 1, na)
    x[-nb:] *= np.linspace(1, 0, nb)
    return x


def loopify(x, xf=0.25):
    n = int(SR * xf)
    body = x[:-n].copy()
    head = x[-n:]
    w = np.linspace(0, 1, n)
    body[:n] = body[:n] * w + head * (1 - w)
    return body


def save(name, x, loop=False):
    if loop:
        x = loopify(x)
    else:
        x = fade(x)
    x = norm(x)
    sf.write(os.path.join(out, name + '.wav'), x.astype(np.float32), SR, subtype='PCM_16')


# 1. arco de soldadura (bucle): zumbido electrico + crepitar
def arco(d=5.0):
    tt = t(d)
    hum = sum(np.sin(2 * np.pi * 100 * k * tt + rng.uniform(0, 6)) / k for k in range(1, 9)) * 0.25
    fry = bp(noise(d), 1500, 7000)
    gate = (rng.random(int(d / 0.004) + 2) > 0.55).astype(float)
    gate = np.repeat(gate, int(SR * 0.004))
    gate = np.pad(gate, (0, max(0, len(tt) - len(gate))))[:len(tt)]
    gate = lp(gate, 400, 2)
    crack = hp(noise(d), 3000) * gate * 1.6
    x = hum * 1.1 + fry * 0.25 + lp(crack, 9000) * 0.35
    x *= 1 + 0.25 * np.sin(2 * np.pi * 7 * tt)
    return x

save('arco_soldadura_loop', arco(), loop=True)


# 2. encendido del arco (zap)
def zap():
    d = 0.7
    tt = t(d)
    f = 1800 * np.exp(-tt * 9) + 120
    ph = 2 * np.pi * np.cumsum(f) / SR
    tone = np.sign(np.sin(ph)) * 0.4 + np.sin(ph) * 0.6
    burst = hp(noise(d), 2500) * np.exp(-tt / 0.07)
    return (tone * np.exp(-tt / 0.18) * 0.6 + burst * 1.0)

save('arco_encendido_zap', zap())


# 3. chispas (crepitar de particulas)
def chispas(d=2.2):
    x = np.zeros(int(SR * d))
    for _ in range(140):
        p = int(rng.uniform(0, d - 0.05) * SR)
        dur = int(SR * rng.uniform(0.002, 0.012))
        amp = rng.uniform(0.15, 1.0) * (1.0 - 0.6 * p / (SR * d))
        c = rng.standard_normal(dur) * np.exp(-np.arange(dur) / (dur / 3))
        x[p:p + dur] += c * amp
    x = lp(hp(x, 2200), 9000)
    return x

save('chispas', chispas())


# 4. radial cortando metal
def radial(d=3.2):
    tt = t(d)
    f0 = 3100
    rise = np.minimum(tt / 0.5, 1.0)
    f = f0 * (0.4 + 0.6 * rise) + 40 * np.sin(2 * np.pi * 3 * tt)
    ph = 2 * np.pi * np.cumsum(f) / SR
    tone = sum(np.sin(k * ph) / k for k in range(1, 5))
    scrape = bp(noise(d), 1200, 9000) * (0.7 + 0.3 * np.sin(2 * np.pi * 45 * tt))
    sp = chispas(d) * 0.7
    x = tone * 0.35 + scrape * 0.9 + sp
    return x * np.minimum(tt / 0.25, 1) * np.minimum((d - tt) / 0.35, 1)

save('radial_corte', radial())


# 5. golpe de martillo en yunque (ping metalico)
def yunque(d=1.4):
    tt = t(d)
    parts = [(1180, 0.35), (1960, 0.22), (3010, 0.14), (4470, 0.09), (6120, 0.05)]
    x = sum(np.sin(2 * np.pi * f * tt + rng.uniform(0, 3)) * np.exp(-tt / tau) * (1 / (i + 1)) for i, (f, tau) in enumerate(parts))
    click = hp(noise(d), 800) * np.exp(-tt / 0.006) * 1.2
    body = np.sin(2 * np.pi * 240 * tt) * np.exp(-tt / 0.05) * 0.6
    return x * 0.8 + click + body

save('martillo_yunque', yunque())


# 6. golpe seco de metal (piqueta / chapa)
def golpe_seco(d=0.5):
    tt = t(d)
    click = hp(noise(d), 1500) * np.exp(-tt / 0.01)
    tone = np.sin(2 * np.pi * 820 * tt) * np.exp(-tt / 0.05) + np.sin(2 * np.pi * 1730 * tt) * np.exp(-tt / 0.03) * 0.5
    return click * 1.2 + tone * 0.6

save('golpe_metal_seco', golpe_seco())


# 7. picar escoria con piqueta (varios toquecitos)
def piqueta(d=1.3):
    x = np.zeros(int(SR * d))
    pos = [0.02, 0.19, 0.31, 0.52, 0.64, 0.81, 0.93, 1.07]
    for p in pos:
        g = golpe_seco(0.25) * rng.uniform(0.5, 1.0)
        i = int(p * SR)
        seg = x[i:i + len(g)]
        seg += g[:len(seg)] * (1 if rng.random() > 0.3 else 0.6)
    return x

save('picar_escoria', piqueta())


# 8. gas / soplete (siseo, bucle)
def gas(d=4.0):
    tt = t(d)
    x = bp(noise(d), 2500, 9000, 2) * 0.8 + bp(noise(d), 300, 1500, 2) * 0.35
    x *= 1 + 0.12 * np.sin(2 * np.pi * 0.6 * tt)
    return x

save('gas_siseo_loop', gas(), loop=True)


# 9. taladro (zumbido barrena)
def taladro(d=2.6):
    tt = t(d)
    f = 420 + 700 * np.minimum(tt / 0.8, 1) - 120 * np.clip((tt - (d - 0.7)) / 0.7, 0, 1)
    ph = 2 * np.pi * np.cumsum(f) / SR
    tone = sum(np.sin(k * ph) / (k ** 1.2) for k in range(1, 7))
    rough = bp(noise(d), 700, 4000) * 0.4
    x = tone * 0.7 + rough
    return x * np.minimum(tt / 0.15, 1) * np.minimum((d - tt) / 0.3, 1)

save('taladro', taladro())


# 10. cepillo de alambre (rascado)
def cepillo(d=1.8):
    tt = t(d)
    x = bp(noise(d), 1800, 11000, 2)
    ch = 0.55 + 0.45 * np.abs(np.sin(2 * np.pi * 3.3 * tt))
    return x * ch * np.minimum(tt / 0.05, 1) * np.minimum((d - tt) / 0.15, 1)

save('cepillo_alambre', cepillo())


# 11. golpe de chapa que cae (clang con reverb corto)
def clang(d=1.8):
    tt = t(d)
    modes = [(310, 0.6), (790, 0.45), (1430, 0.33), (2260, 0.22), (3340, 0.15), (4900, 0.08)]
    x = sum(np.sin(2 * np.pi * f * tt + rng.uniform(0, 3)) * np.exp(-tt / tau) / (1 + i * 0.5) for i, (f, tau) in enumerate(modes))
    imp = hp(noise(d), 500) * np.exp(-tt / 0.015) * 1.3
    return x + imp

save('chapa_cae_clang', clang())


# 12. whoosh metalico (transicion)
def whoosh(d=0.9):
    tt = t(d)
    sweep = np.exp(-((tt - 0.4 * d) ** 2) / (2 * (0.18 * d) ** 2))
    n = noise(d)
    f = 300 + 5200 * sweep
    out_ = np.zeros_like(n)
    blocks = 40
    L = len(n) // blocks
    for b in range(blocks):
        lo = max(200, f[b * L] * 0.6)
        hi = min(12000, f[b * L] * 1.6 + 400)
        out_[b * L:(b + 1) * L] = bp(n[b * L:(b + 1) * L], lo, hi, 2)
    return out_ * sweep

save('whoosh_metal', whoosh())


# 13. clic de interfaz / contador
def click(d=0.12):
    tt = t(d)
    return (np.sin(2 * np.pi * 1900 * tt) * np.exp(-tt / 0.012) + hp(noise(d), 3000) * np.exp(-tt / 0.003) * 0.6)

save('clic_contador', click())


# 14. acierto / recompensa (campanilla corta)
def ding(d=1.2):
    tt = t(d)
    return (np.sin(2 * np.pi * 1320 * tt) * np.exp(-tt / 0.4) + np.sin(2 * np.pi * 1980 * tt) * np.exp(-tt / 0.28) * 0.5 + np.sin(2 * np.pi * 2640 * tt) * np.exp(-tt / 0.18) * 0.25)

save('ding_acierto', ding())


# 15. impacto grave (boom)
def boom(d=1.6):
    tt = t(d)
    f = 110 * np.exp(-tt * 2.5) + 38
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) * np.exp(-tt / 0.55) + lp(noise(d), 400) * np.exp(-tt / 0.12) * 0.9
    return x

save('impacto_grave', boom())

print(sorted(os.listdir(out)))
