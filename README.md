# Sign Language Detector — Streamlit

Versi web (Streamlit) dari aplikasi Sign Language Detector. UI Tkinter sudah
dihapus sepenuhnya — aplikasi ini dibangun khusus untuk **pengguna akhir**:
mendeteksi gesture lewat kamera browser dan membuka kamus isyarat. Tidak ada
fitur pengumpulan data / training di sini.

## Isi folder

```
streamlit_app.py         # aplikasi utama (deteksi real-time + kamus)
feature_extractor.py      # modul ekstraksi fitur tangan (dipakai saat inferensi)
gesture_dictionary.json   # data kamus: deskripsi, cara, kategori, gambar
model.p                   # model klasifikasi yang sudah dilatih
assets/gestures/          # gambar untuk kamus
requirements.txt
```

## Cara menjalankan

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Buka URL yang muncul di terminal (default `http://localhost:8501`), lalu
izinkan akses kamera saat browser memintanya.

## Fitur

- **Tab 🎥 Deteksi** — kamera aktif langsung di browser (via WebRTC, jadi bisa
  dideploy ke server tanpa perlu kamera fisik di server tersebut). Menampilkan
  jumlah tangan terdeteksi, confidence, stabilitas prediksi, dan histori kata
  yang berhasil dikenali (bisa dihapus dengan tombol "Hapus Histori").
- **Tab 📖 Kamus Isyarat** — daftar seluruh gesture di `gesture_dictionary.json`
  lengkap dengan gambar, deskripsi, dan cara melakukannya. Ada kolom pencarian
  untuk gesture yang jumlahnya banyak.

## Catatan model & kamus

- Model (`model.p`) sudah **dilatih ulang** dari `training/data.pickle`
  memakai versi `scikit-learn` yang sama dengan yang dikunci di
  `requirements.txt`. Model versi sebelumnya dilatih pakai scikit-learn lama
  (1.2.0) dan **crash** ("no attribute 'monotonic_cst'") kalau dijalankan
  dengan scikit-learn versi baru — persis kejadian "kamera freeze begitu
  tangan muncul" yang dilaporkan sebelumnya, karena crash-nya terjadi tepat
  di baris `model.predict_proba(...)`, yang hanya dipanggil saat tangan
  terdeteksi.
- `mediapipe` dikunci ke `0.10.14` karena versi 0.10.30 ke atas menghapus API
  `mediapipe.solutions` yang dipakai app ini.
- **Penting:** install dependency PERSIS sesuai `requirements.txt` (jangan
  di-upgrade manual), supaya kombinasi versi tetap teruji cocok dengan
  `model.p`.
- Kalau ingin menambah gesture baru atau melatih ulang model dari nol, itu
  tetap dilakukan lewat skrip di folder `training/` (di luar aplikasi ini),
  lalu salin `model.p` hasilnya ke folder ini.
- Beberapa entri kamus mereferensikan nama file gambar (mis. `Halo.png`) yang
  mungkin belum ada di `assets/gestures/` pada paket asli (folder itu baru
  berisi `A.png`–`Z.png`). Jika gambar tidak ditemukan, aplikasi otomatis
  menampilkan placeholder abu-abu bertanda "?" — tinggal taruh file gambar
  dengan nama yang sesuai di `assets/gestures/` untuk melengkapinya.

## Deploy gratis ke Streamlit Community Cloud

Karena kamera diakses lewat WebRTC di browser (bukan `cv2.VideoCapture` di
server), aplikasi ini bisa dideploy ke cloud tanpa perlu kamera fisik
terpasang di server. Langkah-langkahnya:

### 1. Push ke GitHub

Buat repo baru (boleh public atau private), lalu upload seluruh isi folder
`streamlit_app/` ini ke repo tersebut (termasuk `requirements.txt`,
`model.p`, `gesture_dictionary.json`, `assets/`). **Jangan** upload
`.streamlit/secrets.toml` (kalau sudah dibuat) — itu tempat kredensial
rahasia, cukup upload `.streamlit/secrets.toml.example` sebagai contoh.

### 2. Buat akun TURN server gratis (WAJIB untuk kamera)

Ini bagian yang sering terlewat: kalau langsung deploy tanpa TURN server,
tombol START kamera biasanya cuma **loading terus / tidak pernah nyambung**
begitu di-hosting online (STUN publik saja tidak cukup di banyak server
cloud, termasuk Streamlit Community Cloud).

1. Daftar gratis di **https://www.metered.ca** (paket free: 20GB
   TURN traffic/bulan, cukup banyak untuk pemakaian wajar).
2. Di dashboard-nya, buat "App" baru — nanti kamu dapat nama app (domain)
   dan API Key.
3. Catat dua nilai ini, dipakai di langkah 4.

### 3. Deploy di Streamlit Community Cloud

1. Buka **https://share.streamlit.io**, login/daftar pakai akun GitHub.
2. Klik **"Create app"** → pilih repo, branch, dan file path
   `streamlit_app.py`.
3. (Opsional) atur subdomain URL-nya di "App URL".
4. Klik **Deploy** — proses build biasanya beberapa menit (install
   mediapipe & opencv agak lama).

### 4. Isi Secrets (TURN credentials)

1. Di halaman app kamu di Community Cloud, buka menu **⋮ → Settings →
   Secrets**.
2. Isi:
   ```toml
   METERED_API_KEY = "api-key-dari-metered.ca"
   METERED_DOMAIN = "nama-app-metered-kamu"
   ```
3. Simpan — app akan otomatis restart dan kamera akan pakai TURN server
   ini.

### Catatan keterbatasan hosting gratis

- Streamlit Community Cloud gratis membatasi resource per app (kira-kira
  ~1GB RAM). Kombinasi mediapipe + opencv + model termasuk cukup berat,
  jadi kalau app terasa lambat/sering "reboot" sendiri di cloud (beda
  dengan di komputer sendiri), itu wajar untuk tier gratis.
- Kalau Community Cloud terasa kurang stabil untuk app ini, alternatif
  gratis lain yang juga umum dipakai untuk app `streamlit-webrtc`:
  **Hugging Face Spaces** (pilih SDK "Streamlit" saat membuat Space) —
  caranya mirip, dan secrets TURN server diisi lewat menu "Settings →
  Repository secrets" di Space tersebut.
- Paket free Metered.ca (20GB/bulan) bisa habis kalau dipakai banyak orang
  terus-menerus; kalau itu terjadi, kamera akan gagal konek lagi sampai
  kuota reset bulan berikutnya, atau bisa upgrade ke paket berbayar mereka.
