"""
Sign Language Detector — Aplikasi Streamlit
Fokus: PENGGUNA AKHIR (deteksi isyarat real-time + kamus isyarat).
Tidak ada fitur training/pengumpulan data di sini — itu tetap dilakukan
secara terpisah lewat skrip di folder training/, lalu model.p yang sudah
jadi dipakai di sini.

Jalankan:
    streamlit run streamlit_app.py
"""

import os
import json
import time
import pickle
import threading
from collections import deque

import av
import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

from feature_extractor import build_feature_vector, TOTAL_FEATURES

# ──────────────────────────── Konfigurasi ────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.p")
DICT_PATH = os.path.join(BASE_DIR, "gesture_dictionary.json")
GESTURE_IMG_DIR = os.path.join(BASE_DIR, "assets", "gestures")

CONFIDENCE_THRESHOLD = 0.80
SMOOTH_WINDOW = 10
# Toleransi frame berturut-turut tanpa tangan sebelum buffer stabilizer direset.
NO_HAND_GRACE_FRAMES = 3


@st.cache_data(ttl=3000)
def get_ice_servers():
    """Ambil daftar ICE server (STUN + TURN) untuk koneksi kamera WebRTC.

    Kalau di-deploy ke server cloud (mis. Streamlit Community Cloud), STUN
    saja SERING TIDAK CUKUP — koneksi kamera akan macet/loading terus tanpa
    error yang jelas. Butuh TURN server sebagai jalur cadangan.

    Kalau secret METERED_API_KEY & METERED_DOMAIN diisi (lihat README, opsi
    hosting gratis lewat metered.ca), TURN server gratis dipakai otomatis.
    Kalau tidak diisi, fallback ke STUN publik Google saja — cukup untuk
    development di localhost, tapi kemungkinan besar TIDAK cukup untuk
    kamera yang di-deploy online.
    """
    api_key = st.secrets.get("METERED_API_KEY", "")
    domain = st.secrets.get("METERED_DOMAIN", "")
    if api_key and domain:
        try:
            resp = requests.get(
                f"https://{domain}.metered.live/api/v1/turn/credentials",
                params={"apiKey": api_key},
                timeout=5,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            pass  # fallback ke STUN publik di bawah kalau gagal ambil TURN

    return [{"urls": ["stun:stun.l.google.com:19302"]}]


RTC_CONFIGURATION = RTCConfiguration({"iceServers": get_ice_servers()})

st.set_page_config(
    page_title="Sign Language Detector",
    page_icon="🤟",
    layout="wide",
)


# ──────────────────────────── Load Model & Kamus (cache) ─────────────
@st.cache_resource(show_spinner="Memuat model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            "File **model.p** tidak ditemukan di folder app.\n\n"
            "Model harus sudah dilatih terlebih dahulu (lewat skrip di folder "
            "`training/`) sebelum aplikasi ini bisa dipakai."
        )
        st.stop()

    with open(MODEL_PATH, "rb") as f:
        model_dict = pickle.load(f)

    model = model_dict["model"]
    model_name = model_dict.get("model_name", "Unknown")

    # MODEL_CLASSES[i] = kelas_index asli untuk kolom ke-i pada model.predict_proba().
    # WAJIB dipakai untuk menerjemahkan hasil np.argmax(proba) ke kelas_index.
    model_classes = model_dict.get("classes")
    if model_classes is None:
        model_classes = list(range(len(getattr(model, "classes_", []))))
    return model, model_name, model_classes


@st.cache_resource(show_spinner="Memuat kamus isyarat...")
def load_dictionary():
    with open(DICT_PATH, "r", encoding="utf-8") as f:
        dictionary = json.load(f)

    labels = {}
    for name, info in dictionary.items():
        idx = info.get("kelas_index")
        if idx is not None:
            labels[int(idx)] = name

    required_hands = {
        name: info.get("jumlah_tangan", 1) for name, info in dictionary.items()
    }
    return dictionary, labels, required_hands


model, MODEL_NAME, MODEL_CLASSES = load_model()
DICTIONARY, LABELS, REQUIRED_HANDS = load_dictionary()


@st.cache_resource(show_spinner=False)
def load_gesture_image(filename, size=(320, 320)):
    """Muat gambar gesture. Kalau file tidak ada, kembalikan placeholder."""
    path = os.path.join(GESTURE_IMG_DIR, filename) if filename else None
    if path and os.path.exists(path):
        img = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
        return img

    img = Image.new("RGB", size, (26, 29, 39))
    d = ImageDraw.Draw(img)
    d.text((size[0] // 2 - 10, size[1] // 2 - 14), "?", fill=(142, 149, 176))
    return img


# ──────────────────────────── State bersama (kamera <-> UI) ──────────
class SharedState:
    """Objek ini dipakai untuk berbagi data antara thread pemroses video
    (streamlit-webrtc) dengan thread utama Streamlit yang menggambar UI."""

    def __init__(self):
        self.lock = threading.Lock()
        self.num_hands = 0
        self.confidence = 0.0
        self.stability = 0.0
        self.last_stable = ""
        self.history = []


if "shared" not in st.session_state:
    st.session_state.shared = SharedState()
shared: SharedState = st.session_state.shared


# ──────────────────────────── Video Processor ─────────────────────────
class SignProcessor(VideoProcessorBase):
    # Proses deteksi (MediaPipe + model) hanya tiap N frame. Frame yang
    # dilewati tetap ditampilkan (pakai overlay hasil terakhir) supaya
    # video di layar tetap terasa mulus, sementara beban CPU turun jauh.
    PROCESS_EVERY_N_FRAMES = 2
    # Lebar maksimum frame yang diproses MediaPipe/model. Frame kamera asli
    # tetap dikirim balik dalam resolusi aslinya (cuma perhitungan yang
    # dilakukan di ukuran kecil) — ini adalah penyebab utama "berat &
    # freeze": tanpa downscale, tiap frame di-mediapipe-proses di resolusi
    # penuh kamera (mis. 1280x720), yang jauh lebih lambat daripada 480px.
    PROCESS_MAX_WIDTH = 480

    def __init__(self):
        import mediapipe as mp

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.hands_proc = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            # model_complexity=0 pakai varian model MediaPipe yang paling ringan
            # ("Lite"). Ini yang paling menentukan performa real-time — di
            # complexity=1 (default), tiap frame dengan tangan terdeteksi bisa
            # 2-3x lebih lambat, dan itulah yang bikin video "ngelag/freeze"
            # tepat saat tangan muncul di kamera.
            model_complexity=0,
            # Diturunkan sedikit dari 0.5 -> 0.4: gesture 2 tangan yang posisinya
            # berdekatan/tumpang tindih bikin MediaPipe kadang cuma nangkep 1 dari
            # 2 tangan di ambang 0.5. Validasi jumlah tangan + confidence threshold
            # classifier (0.80) tetap jadi penyaring utama.
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self.pred_hist = deque(maxlen=SMOOTH_WINDOW)
        self.no_hand_streak = 0
        self._frame_count = 0
        self._last_overlay = None  # (boxes_and_text info) dari frame terakhir yang diproses

    def _get_stable(self, cls_idx, conf):
        self.pred_hist.append(cls_idx if conf >= CONFIDENCE_THRESHOLD else -1)
        if len(self.pred_hist) < SMOOTH_WINDOW // 2:
            return -1, 0.0
        counts = {}
        for p in self.pred_hist:
            counts[p] = counts.get(p, 0) + 1
        most = max(counts, key=counts.get)
        stab = counts[most] / len(self.pred_hist)
        return (most, stab) if stab >= 0.6 and most != -1 else (-1, 0.0)

    def _run_detection(self, rgb, W, H):
        """Jalankan MediaPipe + model pada frame `rgb` (bisa versi kecil untuk
        inferensi). Mengembalikan (results, num_hands, confidence, stability,
        label, box, color, txt) — box dalam koordinat piksel relatif ke W,H
        yang DIBERIKAN (jadi kalau rgb sudah di-downscale, W/H juga harus
        ukuran kecil itu; koordinat lalu diskalakan ulang oleh pemanggil)."""
        results = self.hands_proc.process(rgb)

        num_hands = 0
        confidence = 0.0
        stability = 0.0
        label = ""
        box = None
        color = (100, 120, 255)
        txt = ""

        if results.multi_hand_landmarks:
            self.no_hand_streak = 0
            num_hands = len(results.multi_hand_landmarks)

            feats = build_feature_vector(
                results.multi_hand_landmarks, results.multi_handedness
            )
            if len(feats) == TOTAL_FEATURES:
                proba = model.predict_proba([np.asarray(feats)])[0]
                max_conf = float(np.max(proba))
                proba_idx = int(np.argmax(proba))
                # PENTING: proba_idx adalah index kolom predict_proba, bukan
                # otomatis sama dengan kelas_index — terjemahkan lewat MODEL_CLASSES.
                pred_cls = MODEL_CLASSES[proba_idx]

                pred_label = LABELS.get(pred_cls, "?")
                required_hands = REQUIRED_HANDS.get(pred_label, 1)
                hands_mismatch = num_hands != required_hands
                if hands_mismatch:
                    pred_cls = -1
                    max_conf = 0.0

                confidence = max_conf
                stable_cls, stab = self._get_stable(pred_cls, max_conf)
                stability = stab

                ax = [lm.x for hl in results.multi_hand_landmarks for lm in hl.landmark]
                ay = [lm.y for hl in results.multi_hand_landmarks for lm in hl.landmark]
                # Disimpan dalam FRAKSI (0-1), bukan piksel, supaya bisa dipakai
                # lagi di frame ukuran berapa pun (termasuk frame skip berikutnya
                # yang resolusinya sama dengan frame asli, bukan versi kecil ini).
                fx1 = max(0.0, min(ax) - 20 / W)
                fy1 = max(0.0, min(ay) - 20 / H)
                fx2 = min(1.0, max(ax) + 20 / W)
                fy2 = min(1.0, max(ay) + 20 / H)
                box = (fx1, fy1, fx2, fy2)

                if stable_cls != -1:
                    label = LABELS.get(stable_cls, "?")
                    color = (0, 220, 140)
                    txt = f"{label}  {max_conf:.0%}"
                elif hands_mismatch:
                    color = (255, 150, 60)
                    txt = f"Butuh {required_hands} tangan untuk '{pred_label}'"
                else:
                    color = (100, 120, 255)
                    txt = (
                        f"Menunggu... {max_conf:.0%}"
                        if max_conf >= CONFIDENCE_THRESHOLD
                        else f"Tidak dikenali {max_conf:.0%}"
                    )
        else:
            self.no_hand_streak += 1
            if self.no_hand_streak >= NO_HAND_GRACE_FRAMES:
                self.pred_hist.clear()

        return results, num_hands, confidence, stability, label, box, color, txt

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        H, W, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self._frame_count += 1
        do_process = (self._frame_count % self.PROCESS_EVERY_N_FRAMES == 0)

        if do_process:
            # Downscale HANYA untuk input MediaPipe/model — ini yang paling
            # berpengaruh ke performa. Landmark yang dikembalikan MediaPipe
            # berupa koordinat 0-1 (relatif), jadi tetap akurat dipakai/
            # digambar di frame resolusi asli tanpa perlu konversi rumit.
            scale = min(1.0, self.PROCESS_MAX_WIDTH / W)
            small = (
                cv2.resize(rgb, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_LINEAR)
                if scale < 1.0 else rgb
            )
            sh, sw, _ = small.shape
            results, num_hands, confidence, stability, label, box, color, txt = (
                self._run_detection(small, sw, sh)
            )
            self._last_overlay = (results, num_hands, confidence, stability, label, box, color, txt)
        else:
            # Frame ini dilewati (tidak diproses ulang) — pakai hasil deteksi
            # terakhir supaya tampilan tetap konsisten tanpa membebani CPU.
            if self._last_overlay is not None:
                results, num_hands, confidence, stability, label, box, color, txt = self._last_overlay
            else:
                results = None
                num_hands, confidence, stability = 0, 0.0, 0.0
                label, box, txt = "", None, ""
                color = (100, 120, 255)

        if results is not None and results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    rgb,
                    hl,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style(),
                )

        if box is not None:
            fx1, fy1, fx2, fy2 = box
            x1, y1, x2, y2 = int(fx1 * W), int(fy1 * H), int(fx2 * W), int(fy2 * H)
            cv2.rectangle(rgb, (x1, y1), (x2, y2), color, 3)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(rgb, (x1, y1 - th - 14), (x1 + tw + 8, y1), color, -1)
            cv2.putText(
                rgb, txt, (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2, cv2.LINE_AA,
            )

        with shared.lock:
            shared.num_hands = num_hands
            shared.confidence = confidence
            shared.stability = stability
            if label and label != shared.last_stable:
                shared.last_stable = label
                if not shared.history or shared.history[-1] != label:
                    shared.history.append(label)
            elif not label:
                # Reset juga di sini (bukan cuma saat ganti ke label lain) supaya
                # gesture yang sama tapi diulang (tangan turun lalu naik lagi)
                # tetap tercatat di histori.
                shared.last_stable = ""

        out_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return av.VideoFrame.from_ndarray(out_bgr, format="bgr24")


# ──────────────────────────── UI: Halaman Deteksi ─────────────────────
def page_deteksi():
    st.subheader("🎥 Deteksi Bahasa Isyarat Real-time")
    st.caption(
        "Nyalakan kamera, lalu tunjukkan gesture tangan sesuai kamus isyarat. "
        "Izinkan akses kamera di browser saat diminta."
    )

    col_cam, col_panel = st.columns([2, 1], gap="large")

    with col_cam:
        ctx = webrtc_streamer(
            key="sign-language-detector",
            video_processor_factory=SignProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            # Batasi resolusi & frame rate yang diminta dari kamera browser.
            # Sebelumnya tidak dibatasi sama sekali, jadi browser bisa mengirim
            # 720p/1080p @30fps — jauh lebih berat daripada yang dibutuhkan
            # model (yang cuma butuh gambar kecil). Ini penyebab utama "berat".
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 480},
                    "height": {"ideal": 360},
                    "frameRate": {"ideal": 15, "max": 20},
                },
                "audio": False,
            },
            async_processing=True,
            # Antrian frame masuk diperkecil jadi 1: kalau pemrosesan sempat
            # tertinggal (mis. saat tangan baru muncul & bebannya naik),
            # frame lama langsung dibuang dan yang diproses selalu frame
            # TERBARU — bukan menumpuk lalu diproses berurutan (itulah yang
            # sebelumnya terasa seperti "ngelag/freeze").
            video_receiver_size=1,
        )

    with col_panel:
        st.markdown("**Status Deteksi**")
        stat_box = st.empty()

        st.markdown("**📝 Kalimat Terdeteksi**")
        hist_box = st.empty()

        clear_clicked = st.button("🗑️ Hapus Histori", use_container_width=True)
        if clear_clicked:
            with shared.lock:
                shared.history.clear()
                shared.last_stable = ""

        with st.expander("Gesture yang tersedia"):
            st.write(", ".join(LABELS.values()) if LABELS else "—")
        st.caption(f"Model aktif: {MODEL_NAME}")

    if not ctx.state.playing:
        st.info("💡 Tekan **START** di atas untuk mengaktifkan kamera.")

    _render_status_panel(stat_box, hist_box, ctx.state.playing)


@st.fragment(run_every="0.4s")
def _render_status_panel(stat_box, hist_box, is_playing):
    """Refresh panel status & histori secara berkala.

    Sengaja dipisah jadi @st.fragment (bukan st.rerun() di seluruh halaman
    seperti sebelumnya). st.rerun() penuh ikut me-render ulang komponen
    webrtc_streamer setiap 0.4 detik, yang menambah beban CPU/GIL dan bisa
    memicu koneksi kamera "tersendat" — sedangkan fragment cuma me-refresh
    bagian kecil ini saja tanpa mengganggu stream kamera yang sedang jalan.
    """
    with shared.lock:
        num_hands = shared.num_hands
        confidence = shared.confidence
        stability = shared.stability
        history = list(shared.history)

    with stat_box.container():
        c1, c2, c3 = st.columns(3)
        c1.metric("Tangan", num_hands)
        c2.metric("Confidence", f"{confidence:.0%}" if confidence else "—")
        c3.metric("Stabilitas", f"{stability:.0%}" if stability else "—")

    hist_box.text_area(
        "Hasil gesture yang terdeteksi akan muncul di sini",
        value="  ".join(history) if history else "",
        height=140,
        label_visibility="collapsed",
    )

    # Fragment ini tetap ter-jadwal jalan tiap 0.4s oleh Streamlit walau
    # kamera mati; itu tidak masalah karena isinya cuma re-render angka
    # statis (bukan proses berat), jadi cukup berhenti di sini saja.
    return


# ──────────────────────────── UI: Halaman Kamus ────────────────────────
def page_kamus():
    st.subheader("📖 Kamus Bahasa Isyarat")
    st.caption("Pelajari cara melakukan setiap gesture sebelum mencobanya di halaman Deteksi.")

    names = list(DICTIONARY.keys())
    if not names:
        st.warning("Kamus masih kosong.")
        return

    col_list, col_detail = st.columns([1, 2], gap="large")

    with col_list:
        st.markdown("**Pilih Gesture**")
        query = st.text_input("Cari gesture", placeholder="Ketik untuk mencari...", label_visibility="collapsed")
        filtered = [n for n in names if query.lower() in n.lower()] if query else names
        if not filtered:
            st.info("Tidak ditemukan.")
            return
        selected = st.radio("Daftar gesture", filtered, label_visibility="collapsed")

    with col_detail:
        info = DICTIONARY[selected]
        img = load_gesture_image(info.get("gambar", ""))
        img_col, text_col = st.columns([1, 1.4], gap="large")
        with img_col:
            st.image(img, use_container_width=True)
        with text_col:
            st.markdown(f"## {selected}")
            st.markdown(f"`{info.get('kategori', '-')}`  ·  {info.get('jumlah_tangan', 1)} tangan")
            st.markdown("**📋 Deskripsi**")
            st.write(info.get("deskripsi", "-"))
            st.markdown("**🤲 Cara Membuat**")
            st.write(info.get("cara", "-"))


# ──────────────────────────── Main ────────────────────────────────────
def main():
    st.title("🤟 Sign Language Detector")

    tab_deteksi, tab_kamus = st.tabs(["🎥 Deteksi", "📖 Kamus Isyarat"])
    with tab_deteksi:
        page_deteksi()
    with tab_kamus:
        page_kamus()


if __name__ == "__main__":
    main()
