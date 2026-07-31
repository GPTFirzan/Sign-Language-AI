"""
Modul ekstraksi fitur tangan.
Dipakai bersama oleh create_dataset.py dan train_classifier.py
agar fitur selalu IDENTIK antara training dan inferensi.
"""
import numpy as np
import math


def _angle(a, b, c):
    """Sudut di titik b, dibentuk oleh segmen b->a dan b->c (dalam derajat)."""
    ax, ay = a[0] - b[0], a[1] - b[1]
    cx, cy = c[0] - b[0], c[1] - b[1]
    dot = ax * cx + ay * cy
    mag = (math.hypot(ax, ay) * math.hypot(cx, cy)) + 1e-6
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _derotate_points(pts):
    """Putar SELURUH titik tangan supaya vektor pergelangan (landmark 0) ->
    pangkal jari tengah (landmark 9) selalu menghadap arah yang sama
    ("ke atas"), sebelum bounding box dihitung.

    Kenapa perlu: sebelumnya koordinat cuma dinormalisasi pakai bounding
    box axis-aligned (min/max x,y) tanpa koreksi rotasi. Itu artinya
    gesture yang SAMA tapi tangannya dimiringkan sedikit saja (pergelangan
    diputar) menghasilkan 42 nilai koordinat yang berbeda jauh — padahal
    bentuk relatif jari-jarinya identik. Ini penyebab utama gesture yang
    "dimiringkan" (mis. huruf E) atau sedikit berubah posisi jadi sulit
    dikenali: variasi rotasi di dunia nyata (sudut pergelangan, sudut
    kamera) jauh lebih lebar daripada rotasi yang tercakup di data
    training, jadi vektor fitur gampang "kabur" dari pola yang dikenali
    model. Sudut sendi jari & jarak fingertip-ke-pergelangan (bagian 2 &
    3 di bawah) sebenarnya sudah invariant terhadap rotasi ini secara
    matematis; fungsi ini membuat bagian koordinat (bagian 1 & 4) ikut
    invariant juga.
    """
    wrist = pts[0]
    mid_mcp = pts[9]
    dx, dy = mid_mcp[0] - wrist[0], mid_mcp[1] - wrist[1]
    current_angle = math.atan2(dy, dx)
    target_angle = -math.pi / 2  # "menghadap atas" (y kecil = atas di gambar)
    delta = target_angle - current_angle
    cos_d, sin_d = math.cos(delta), math.sin(delta)

    rotated = []
    for x, y in pts:
        rx, ry = x - wrist[0], y - wrist[1]
        nx = rx * cos_d - ry * sin_d
        ny = rx * sin_d + ry * cos_d
        rotated.append((nx + wrist[0], ny + wrist[1]))
    return rotated


def extract_hand_features(hand_landmarks, mirror=False):
    """
    Ekstrak fitur dari satu tangan (21 landmark MediaPipe).
    Mengembalikan list float yang merepresentasikan bentuk tangan.

    Fitur yang diekstrak:
    1. Koordinat ternormalisasi (42 nilai) — posisi relatif
    2. Jarak jari ke pergelangan (5 nilai) — panjang relatif tiap jari
    3. Sudut sendi tiap jari (15 nilai) — bentuk tekukan jari
    4. Rasio lebar/tinggi bounding box tangan (1 nilai) — proporsi tangan

    Total per tangan: 63 nilai
    Total 2 tangan: 126 nilai (tangan ke-2 diisi nol jika tidak ada)

    mirror: kalau True, koordinat x di-cerminkan (1 - x) SEBELUM fitur
    dihitung. Dipakai supaya tangan kiri diproyeksikan ke "bentuk" yang
    sama seperti tangan kanan melakukan gesture yang sama — tanpa ini,
    gesture 1 tangan yang direkam dominan pakai tangan kanan akan punya
    pola koordinat yang BERBEDA (cerminan) saat dilakukan pakai tangan
    kiri, walau bentuk isyaratnya identik secara visual.

    Setelah mirror (kalau ada), titik juga di-derotasi lewat
    _derotate_points() supaya kemiringan/rotasi tangan di gambar tidak
    ikut memengaruhi fitur koordinat. Fitur sudut & jarak sebenarnya
    sudah invariant terhadap refleksi maupun rotasi; 42 nilai koordinat
    mentah yang tidak — makanya perlu dikoreksi di sini.
    """
    lm = hand_landmarks.landmark
    if mirror:
        pts = [(1.0 - l.x, l.y) for l in lm]
    else:
        pts = [(l.x, l.y) for l in lm]

    pts = _derotate_points(pts)

    # 1. Koordinat ternormalisasi (42 nilai)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, min_y = min(xs), min(ys)
    range_x = max(xs) - min_x + 1e-6
    range_y = max(ys) - min_y + 1e-6

    coords = []
    for p in pts:
        coords.append((p[0] - min_x) / range_x)
        coords.append((p[1] - min_y) / range_y)

    # 2. Jarak ujung jari ke pergelangan, dinormalisasi (5 nilai)
    wrist = pts[0]
    fingertips = [pts[4], pts[8], pts[12], pts[16], pts[20]]
    palm_size = _dist(wrist, pts[9]) + 1e-6
    tip_dists = [_dist(wrist, tip) / palm_size for tip in fingertips]

    # 3. Sudut sendi tiap jari (15 nilai: 3 sendi x 5 jari)
    finger_indices = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20],
    ]
    angles = []
    for fi in finger_indices:
        for k in range(len(fi) - 2):
            a = pts[fi[k]]
            b = pts[fi[k + 1]]
            c = pts[fi[k + 2]]
            angles.append(_angle(a, b, c) / 180.0)

    # 4. Rasio aspek tangan (1 nilai)
    aspect = range_x / (range_y + 1e-6)
    aspect = min(aspect, 3.0) / 3.0

    features = coords + tip_dists + angles + [aspect]
    return features  # total: 42 + 5 + 15 + 1 = 63


FEATURES_PER_HAND = 63
TOTAL_FEATURES = FEATURES_PER_HAND * 2  # 126


def augment_feature_vector(features, rng, noise_std=0.02):
    """Tambahkan noise Gaussian kecil ke satu vektor fitur yang SUDAH
    diekstrak, untuk data augmentation.

    Kenapa perlu: satu foto training cuma merekam SATU titik persis di
    ruang fitur. Kalau classifier dibiarkan menghafal titik-titik itu
    persis (lihat catatan max_depth di train_classifier.py), gesture
    yang sama tapi meleset sedikit dari titik itu (jari sedikit lebih
    tekuk, tangan sedikit gemetar, dsb — wajar terjadi di pemakaian
    nyata) gampang jatuh ke wilayah yang salah.

    Fungsi ini membuat beberapa "tetangga" sintetis di sekitar tiap
    sampel asli, supaya classifier belajar wilayah yang lebih longgar
    (toleran terhadap variasi kecil) dan bukan cuma titik persis itu.
    Nilai fitur di-clip ke [0, 1.5] karena mayoritas fitur memang
    berada di rentang itu (koordinat & sudut dinormalisasi ke [0,1];
    beberapa rasio jarak/aspek bisa sedikit >1).
    """
    return [max(0.0, min(1.5, v + rng.gauss(0, noise_std))) for v in features]


def _handedness_label(handedness_obj):
    """Ambil label 'Left'/'Right' dari satu elemen results.multi_handedness.
    Balikin None kalau tidak tersedia (mis. dipanggil tanpa parameter itu)."""
    if handedness_obj is None:
        return None
    try:
        return handedness_obj.classification[0].label
    except (AttributeError, IndexError, KeyError):
        return None


def build_feature_vector(multi_hand_landmarks, multi_handedness=None):
    """
    Bangun vektor fitur 126 nilai dari hasil deteksi MediaPipe.
    Selalu mengembalikan list 126 nilai (padding nol jika <2 tangan).

    multi_handedness: hasil `results.multi_handedness` dari MediaPipe
    (list sejajar/parallel dengan multi_hand_landmarks). Dipakai untuk:

    1. MIRROR — tangan berlabel "Left" di-cerminkan (lihat mirror di
       extract_hand_features) supaya polanya konsisten dengan tangan
       "Right", tidak peduli gesture dilakukan pakai tangan kiri atau
       kanan.
    2. URUTAN SLOT — untuk gesture 2 tangan, tangan diurutkan SELALU
       "Left" dulu baru "Right" sebelum diisi ke slot [0:63] / [63:126].
       Tanpa ini, urutan multi_hand_landmarks dari MediaPipe cuma
       berdasarkan urutan deteksi (bisa tertukar antar frame), jadi
       gesture 2 tangan yang sama bisa menghasilkan vektor fitur yang
       beda-beda tergantung tangan mana yang "kedeteksi duluan".

    Kalau multi_handedness tidak diberikan (None), perilaku turun ke versi
    lama: tidak ada mirror/reorder (untuk kompatibilitas kalau ada
    pemanggil lama yang belum di-update).
    """
    if not multi_hand_landmarks:
        return [0.0] * TOTAL_FEATURES

    hands_to_use = multi_hand_landmarks[:2]

    if multi_handedness:
        handedness_to_use = list(multi_handedness[:2])
        # Samakan panjang kalau MediaPipe entah kenapa kasih jumlah beda
        while len(handedness_to_use) < len(hands_to_use):
            handedness_to_use.append(None)
        paired = list(zip(hands_to_use, handedness_to_use))
        paired.sort(key=lambda item: 0 if _handedness_label(item[1]) == "Left" else 1)
    else:
        paired = [(hl, None) for hl in hands_to_use]

    result = []
    for hand_lm, handed in paired:
        mirror = (_handedness_label(handed) == "Left")
        result.extend(extract_hand_features(hand_lm, mirror=mirror))

    while len(result) < TOTAL_FEATURES:
        result.append(0.0)

    return result[:TOTAL_FEATURES]
