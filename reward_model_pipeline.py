# -*- coding: utf-8 -*-
"""
============================================================================
 ÖDÜL MODELİ EĞİTİMİ - Reward Model Pipeline
 Bilgi Teknolojileri / Yapay Zeka Teknikleri - Ödev 1
============================================================================

AMAÇ
----
Bir veri kümesindeki  Soru (S), Düşünme süreci (D) ve Cevap metni (C)
metinlerini birleştirip, metin embedding'lerini GİRDİ; insan tarafından
verilmiş "puan" (1-5) değerini ise ÇIKTI (hedef) olarak kullanan bir
regresyon modeli (ödül modeli) eğitmek.

DENEY TASARIMI
--------------
  * 4 farklı embedding modeli  ×  5+ farklı ML algoritması  = 20 deney
  * Test kümesi:  rastgele seçilmiş 1000 örnek
  * Eğitim kümesi: geri kalan tüm örnekler
  * Başarı ölçütleri (regresyon): MAE, RMSE, R², Spearman, Pearson,
    ve yuvarlanmış tahmin doğruluğu (1-5 sınıf görünümü).

ÇIKTILAR (sonuclar/ klasörü)
----------------------------
  * embeddings/<model>.npy            -> üretilen embedding matrisleri (önbellek)
  * sonuclar/metrikler.csv            -> 20 deneyin tüm metrikleri (ana tablo)
  * sonuclar/pivot_*.csv              -> embedding × algoritma özet tabloları
  * sonuclar/grafikler/*.png          -> ısı haritaları, çubuk grafikler, scatter
  * sonuclar/en_iyi_model.txt         -> en iyi (embedding, algoritma) ikilisi

KULLANIM
--------
  1) requirements.txt içindeki paketleri kurun:
        pip install -r requirements.txt
  2) Veri dosyanızı (xlsx ya da csv) bu klasöre koyun. Varsayılan aranan
     isimler aşağıda VERI_DOSYASI_ADAYLARI listesindedir; farklıysa
     --veri parametresiyle yol verin.
  3) Çalıştırın:
        python reward_model_pipeline.py
     Hızlı denemek için küçük bir örneklemle:
        python reward_model_pipeline.py --ornek 500

NOT: GPU önerilir (embedding üretimi için). CPU'da da çalışır, sadece yavaştır.
============================================================================
"""

import os
import re
import sys
import json
import time
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0) GENEL AYARLAR (KONFİGÜRASYON)
# ---------------------------------------------------------------------------

# Çalışılan klasör (bu .py dosyasının bulunduğu yer).
KOK = os.path.dirname(os.path.abspath(__file__))

# Sonuçların yazılacağı klasörler.
SONUC_DIR     = os.path.join(KOK, "sonuclar")
EMBED_DIR     = os.path.join(KOK, "embeddings")
GRAFIK_DIR    = os.path.join(SONUC_DIR, "grafikler")

# Veri dosyası otomatik aranırken denenecek isimler.
VERI_DOSYASI_ADAYLARI = [
    "veri.xlsx", "veri.csv", "data.xlsx", "data.csv",
    "odev.xlsx", "odev_veri.xlsx", "dataset.xlsx", "puanlar.xlsx",
]

# Test kümesi büyüklüğü (ödev gereği rastgele 1000 örnek).
TEST_BOYUTU = 1000

# Tekrarlanabilirlik için sabit rastgelelik tohumu.
TOHUM = 42

# ---------------------------------------------------------------------------
# 4 EMBEDDING MODELİ (ödevde belirtilenler)
# ---------------------------------------------------------------------------
# Her model için olası anahtarlar:
#   ad               : kısa, dosya/tabloda kullanılacak etiket
#   repo             : HuggingFace model kimliği
#   prefix           : bazı modeller (örn. E5) girdi metnine ön-ek ister
#   pooling          : transformers yedeğinde havuzlama ('mean' | 'lasttoken')
#   st_model_kwargs  : SentenceTransformer'a yükleme sırasında geçilecek argümanlar
#                      (örn. jina için {"default_task": "retrieval"})
#   force_transformers: True -> sentence-transformers atlanır, doğrudan transformers
#                       + manuel pooling kullanılır (MPS'te sorun çıkaran modeller için)
EMBEDDING_MODELLERI = [
    {
        "ad": "turkish-e5-large",
        "repo": "ytu-ce-cosmos/turkish-e5-large",
        # E5 ailesi, cümle gömmesi için "query: " ön-eki bekler.
        "prefix": "query: ",
        "pooling": "mean",
    },
    {
        "ad": "jina-v5-text-small",
        "repo": "jinaai/jina-embeddings-v5-text-small",
        "prefix": "",
        "pooling": "mean",
        # Jina v5 çok-görevli bir modeldir; gömme üretmeden önce görev seçilmeli.
        # Belge/metin temsili için "retrieval" görevi kullanılıyor.
        "st_model_kwargs": {"default_task": "retrieval"},
    },
    {
        "ad": "harrier-oss-0.6b",
        "repo": "microsoft/harrier-oss-v1-0.6b",
        "prefix": "",
        "pooling": "mean",
    },
    {
        "ad": "qwen3-embed-0.6b",
        "repo": "Qwen/Qwen3-Embedding-0.6B",
        "prefix": "",
        "pooling": "lasttoken",   # Qwen3-Embedding son-token havuzlama kullanır
        # Qwen3, sentence-transformers + MPS'te bellek hatası verdiği için
        # doğrudan transformers + float32 yoluyla üretilir.
        "force_transformers": True,
    },
]


# ---------------------------------------------------------------------------
# 1) VERİ YÜKLEME ve HAZIRLAMA
# ---------------------------------------------------------------------------
def veri_dosyasini_bul(verilen_yol=None):
    """Çalışılacak veri dosyasının yolunu bulur.

    Mantık şöyle işler: Kullanıcı bir yol verdiyse doğrudan onu kullanırız.
    Vermediyse, klasörde sık kullanılan isimlere (veri.xlsx, data.csv, ...)
    bakarız; o da yoksa son çare olarak klasördeki ilk xlsx/csv dosyasını
    seçeriz.

    Parametreler:
        verilen_yol: Kullanıcının --veri ile verdiği dosya yolu, yoksa None.

    Döndürür:
        Bulunan dosyanın tam (mutlak) yolu.

    Not:
        Hiçbir dosya bulunamazsa programı açıklayıcı bir mesajla durdurur.
    """
    if verilen_yol:
        if not os.path.exists(verilen_yol):
            sys.exit(f"[HATA] Belirtilen veri dosyası bulunamadı: {verilen_yol}")
        return verilen_yol
    for ad in VERI_DOSYASI_ADAYLARI:
        yol = os.path.join(KOK, ad)
        if os.path.exists(yol):
            return yol
    # Klasördeki herhangi bir xlsx/csv'yi son çare olarak dene.
    for f in os.listdir(KOK):
        if f.lower().endswith((".xlsx", ".xls", ".csv")):
            return os.path.join(KOK, f)
    sys.exit(
        "[HATA] Veri dosyası bulunamadı. xlsx/csv dosyanızı bu klasöre koyun "
        "ya da --veri parametresiyle yolunu verin."
    )


def kolon_bul(df, anahtarlar, haric=None):
    """Başlığında verilen anahtar kelimelerden birini içeren ilk kolonu bulur.

    Veri dosyalarının kolon başlıkları her zaman aynı yazılmaz; kimi zaman
    "Soru", kimi zaman "Sorunuz" ya da sadece "S" olabilir. Bu fonksiyon, küçük
    harfe indirgenmiş başlıkta anahtar kelimeyi arayarak doğru kolonu yakalar.
    Aynı kolonun iki farklı role (örn. hem D hem C) atanmasını önlemek için
    daha önce seçilmiş kolonlar 'haric' ile dışarıda bırakılabilir.

    Parametreler:
        df: İçinde arama yapılacak DataFrame.
        anahtarlar: Aranacak anahtar kelimeler listesi (örn. ["soru", "question"]).
        haric: Atlanacak (zaten seçilmiş) kolon adları.

    Döndürür:
        Eşleşen ilk kolonun adı; eşleşme yoksa None.
    """
    haric = haric or []
    for kol in df.columns:
        if kol in haric:
            continue
        k = str(kol).strip().lower()
        for a in anahtarlar:
            if a in k:
                return kol
    return None


def kolon_tam_harf(df, harf, haric=None):
    """Adı tam olarak tek bir harften ibaret olan kolonu bulur (örn. yalnızca "S").

    Bu, kolon_bul'un güvenli bir yedeğidir. Tek harfle ("c" gibi) arama yapmak
    tehlikelidir; çünkü "CosmosLLM" gibi içinde o harf geçen bir başlık yanlışlıkla
    eşleşebilir. Burada başlığın TAM olarak o harfe eşit olmasını şart koşarız.

    Parametreler:
        df: İçinde arama yapılacak DataFrame.
        harf: Aranan tek harf (örn. "s", "d", "c").
        haric: Atlanacak (zaten seçilmiş) kolon adları.

    Döndürür:
        Adı tam olarak o harf olan kolonun adı; yoksa None.
    """
    haric = haric or []
    for kol in df.columns:
        if kol in haric:
            continue
        if str(kol).strip().lower() == harf.lower():
            return kol
    return None


def veriyi_yukle(yol):
    """Veri dosyasını okuyup eğitime hazır, temiz bir tablo hâline getirir.

    Sırasıyla şunları yapar: dosyayı (xlsx/csv) okur, soru/düşünme/cevap ve puan
    kolonlarını otomatik tespit eder, üç metni tek bir metinde birleştirir, puanı
    sayıya çevirir (gerekirse "çok kötü…çok iyi" etiketlerini 1–5'e eşler) ve
    puanı geçersiz satırları atar. Yol boyunca ne bulduğunu ekrana yazdırır ki
    veride bir terslik varsa hemen fark edilsin.

    Parametreler:
        yol: Okunacak veri dosyasının yolu.

    Döndürür:
        İki kolonlu bir DataFrame:
          - "metin": "Soru: …\\nDüşünme: …\\nCevap: …" biçiminde birleşik metin
          - "puan" : 1–5 arası tamsayı hedef değer
    """
    print(f"[BİLGİ] Veri okunuyor: {yol}")
    if yol.lower().endswith(".csv"):
        df = pd.read_csv(yol)
    else:
        df = pd.read_excel(yol)

    print(f"[BİLGİ] Ham veri boyutu: {df.shape}  |  Kolonlar: {list(df.columns)}")

    # --- Kolonları akıllıca tespit et ---
    # Önce puan kolonunu sabitle ki diğerleriyle karışmasın.
    kol_P = kolon_bul(df, ["puan", "değerlendirme", "degerlendirme", "score",
                           "rating", "ödül", "odul", "reward", "label"])
    secili = [c for c in [kol_P] if c]
    # S = soru. ("Sorunuz", "Soru", "S" ...)
    kol_S = (kolon_bul(df, ["soru", "question", "prompt"], haric=secili)
             or kolon_tam_harf(df, "s", haric=secili))
    secili += [c for c in [kol_S] if c]
    # D = düşünme süreci. ("CosmosLLM düşünme süreci", "Düşünme", "D" ...)
    kol_D = (kolon_bul(df, ["düşün", "dusun", "think", "reason", "cot"], haric=secili)
             or kolon_tam_harf(df, "d", haric=secili))
    secili += [c for c in [kol_D] if c]
    # C = cevap metni. DİKKAT: "cevabı" kelimesi "cevap" içermez (b≠p),
    # bu yüzden "cevab" anahtarını da ekliyoruz.
    kol_C = (kolon_bul(df, ["cevab", "cevap", "answer", "response", "yanıt",
                            "yanit"], haric=secili)
             or kolon_tam_harf(df, "c", haric=secili))

    # Eğer kolonlar tam olarak S, D, C, puan ise (tek harfli), onları kullan.
    kolon_haritasi = {"S": kol_S, "D": kol_D, "C": kol_C, "puan": kol_P}
    print(f"[BİLGİ] Tespit edilen kolonlar -> {kolon_haritasi}")

    eksik = [k for k, v in kolon_haritasi.items() if v is None]
    if eksik:
        sys.exit(
            f"[HATA] Şu kolonlar tespit edilemedi: {eksik}. "
            f"Lütfen kolon adlarını kontrol edin. Mevcut kolonlar: {list(df.columns)}"
        )

    # --- Metinleri güvenli stringe çevir (NaN -> boş string) ---
    def temiz(x):
        return "" if pd.isna(x) else str(x).strip()

    S = df[kol_S].map(temiz)
    D = df[kol_D].map(temiz)
    C = df[kol_C].map(temiz)

    # --- S + D + C metinlerini birleştir (ödev gereği) ---
    # Modelin hangi metnin nerede başladığını ayırt edebilmesi için
    # açıklayıcı etiketler ekliyoruz.
    birlesik = (
        "Soru: " + S + "\n" +
        "Düşünme: " + D + "\n" +
        "Cevap: " + C
    )

    # --- Hedef değişken (puan) sayıya çevrilir ---
    # Bu veri kümesinde puan, sayı yerine sıralı (ordinal) METİN etiketidir:
    #   "çok kötü"=1, "kötü"=2, "orta"=3, "iyi"=4, "çok iyi"=5
    # Önce sayısal okumayı dener; çoğunluk NaN çıkarsa metin eşlemesine geçer.
    ETIKET_PUAN = {
        "çok kötü": 1, "cok kotu": 1, "çok kotu": 1, "cok kötü": 1,
        "kötü": 2, "kotu": 2,
        "orta": 3,
        "iyi": 4,
        "çok iyi": 5, "cok iyi": 5, "çok iyı": 5,
    }
    puan = pd.to_numeric(df[kol_P], errors="coerce")
    if puan.notna().mean() < 0.5:
        # Metin etiketlerini normalize edip eşle.
        norm = df[kol_P].astype(str).str.strip().str.lower()
        norm = norm.str.replace(r"\s+", " ", regex=True)
        puan = norm.map(ETIKET_PUAN)
        eslesmeyen = sorted(set(norm[puan.isna()].unique()))
        if eslesmeyen:
            print(f"[UYARI] Eşlenemeyen puan etiketleri (atılacak): {eslesmeyen}")
        print(f"[BİLGİ] Puan, metin etiketinden sayıya çevrildi "
              f"(çok kötü=1 … çok iyi=5).")

    veri = pd.DataFrame({"metin": birlesik, "puan": puan})

    # Puanı olmayan (NaN) satırları at.
    once = len(veri)
    veri = veri.dropna(subset=["puan"]).reset_index(drop=True)
    print(f"[BİLGİ] Puanı geçersiz {once - len(veri)} satır atıldı.")

    print(f"[BİLGİ] Kullanılabilir örnek sayısı: {len(veri)}")
    print(f"[BİLGİ] Puan dağılımı:\n{veri['puan'].round().astype(int).value_counts().sort_index()}")
    return veri


# ---------------------------------------------------------------------------
# 2) EMBEDDING ÜRETİMİ
# ---------------------------------------------------------------------------
def embedding_uret(metinler, model_cfg, batch=32):
    """Bir metin listesini, seçilen embedding modeliyle sayısal vektörlere çevirir.

    Bu, makine öğrenmesinin "metni anlamasını" sağlayan adımdır: her metin,
    anlamını temsil eden sabit uzunlukta bir vektöre dönüşür. Olabildiğince
    sağlam olması için birden çok strateji sırayla denenir:
      1) sentence-transformers (gerekirse modele özel görev/argümanla, örn.
         jina için default_task="retrieval"),
      2) olmazsa transformers + elle havuzlama (mean ya da last-token), bf16
         yerine float32'ye zorlayarak (MPS bfloat16'yı desteklemez).
    Her strateji önce GPU/MPS'te, hata olursa CPU'da denenir; böylece bir
    donanım/uyumluluk sorununda program çökmek yerine güvenli yola düşer.
    Vektörler son olarak kosinüs uzayına normalize edilir.

    Parametreler:
        metinler: Vektöre çevrilecek metinlerin listesi.
        model_cfg: Model tanımı (repo adı, ön-ek, pooling türü) içeren sözlük.
        batch: Aynı anda işlenecek metin sayısı (bellek/hız dengesi).

    Döndürür:
        (N, boyut) biçiminde float32 numpy dizisi; her satır bir metnin vektörü.
    """
    import torch

    # Cihaz seçimi: NVIDIA GPU (cuda) > Apple Silicon GPU (mps) > CPU.
    # M-serisi Mac'lerde MPS, embedding üretimini belirgin biçimde hızlandırır.
    if torch.cuda.is_available():
        ana_cihaz = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        ana_cihaz = "mps"
    else:
        ana_cihaz = "cpu"

    prefix = model_cfg.get("prefix", "")
    girdiler = [prefix + t for t in metinler]
    MAKS_UZUNLUK = 512   # MPS'te dev bellek tahsisini önlemek için dizi sınırı

    # ----- Yöntem A: sentence-transformers -----
    def yontem_st(cihaz):
        from sentence_transformers import SentenceTransformer
        mk = model_cfg.get("st_model_kwargs")  # örn. jina için default_task
        model = SentenceTransformer(
            model_cfg["repo"], device=cihaz, trust_remote_code=True,
            model_kwargs=mk if mk else None,
        )
        # Çok uzun bağlamlı modellerde (örn. Qwen3) MPS belleğini korumak için
        # dizi uzunluğunu kıs.
        try:
            if getattr(model, "max_seq_length", 0) and model.max_seq_length > MAKS_UZUNLUK:
                model.max_seq_length = MAKS_UZUNLUK
        except Exception:
            pass
        v = model.encode(
            girdiler, batch_size=batch, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        return np.asarray(v, dtype=np.float32)

    # ----- Yöntem B: transformers + manuel havuzlama (yedek) -----
    def yontem_tf(cihaz):
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(model_cfg["repo"], trust_remote_code=True)
        # MPS bfloat16'yı desteklemez; her zaman float32'ye zorla.
        try:
            model = AutoModel.from_pretrained(
                model_cfg["repo"], trust_remote_code=True,
                torch_dtype=torch.float32, attn_implementation="eager")
        except TypeError:
            model = AutoModel.from_pretrained(
                model_cfg["repo"], trust_remote_code=True,
                torch_dtype=torch.float32)
        model = model.to(cihaz).eval()

        def pool(last_hidden, mask, yontem):
            if yontem == "lasttoken":
                uz = mask.sum(dim=1) - 1            # son gerçek token'ın indeksi
                return last_hidden[torch.arange(last_hidden.size(0)), uz]
            m = mask.unsqueeze(-1).float()          # mean pooling (maskeli ortalama)
            return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)

        cikti = []
        with torch.no_grad():
            for i in range(0, len(girdiler), batch):
                parca = girdiler[i:i + batch]
                enc = tok(parca, padding=True, truncation=True,
                          max_length=MAKS_UZUNLUK, return_tensors="pt").to(cihaz)
                out = model(**enc)
                v = pool(out.last_hidden_state, enc["attention_mask"],
                         model_cfg.get("pooling", "mean"))
                v = torch.nn.functional.normalize(v, p=2, dim=1)
                cikti.append(v.float().cpu().numpy().astype(np.float32))
                if (i // batch) % 10 == 0:
                    print(f"      {i+len(parca)}/{len(girdiler)}")
        return np.vstack(cikti)

    # Denenecek yöntemler: force_transformers ise sadece transformers; aksi
    # halde önce sentence-transformers, sonra transformers.
    if model_cfg.get("force_transformers"):
        yontemler = [("transformers", yontem_tf)]
    else:
        yontemler = [("sentence-transformers", yontem_st),
                     ("transformers", yontem_tf)]

    # Cihaz sırası: önce GPU/MPS, hata olursa CPU'ya düş (MPS bazı modellerde
    # kararsız olabiliyor; CPU yavaş ama güvenli).
    cihazlar = [ana_cihaz, "cpu"] if ana_cihaz != "cpu" else ["cpu"]

    son_hata = None
    for ad, fn in yontemler:
        for cihaz in cihazlar:
            try:
                print(f"   -> yöntem={ad}, cihaz={cihaz} ({model_cfg['repo']})")
                return fn(cihaz)
            except Exception as e:
                son_hata = e
                print(f"   [UYARI] {ad}/{cihaz} başarısız: {e}")

    raise RuntimeError(f"Embedding üretilemedi ({model_cfg['ad']}): {son_hata}")


def embedding_al_veya_uret(metinler, model_cfg):
    """Embedding'leri akıllıca yükler: varsa diskten okur, yoksa üretip kaydeder.

    Embedding üretmek pahalı bir işlemdir, bu yüzden bir kez hesapladığımız
    vektörleri diske (.npy) yazıp önbelleğe alırız. Programı tekrar
    çalıştırdığınızda aynı model için baştan hesaplama yapılmaz; satır sayısı
    uyuştuğu sürece doğrudan diskten yüklenir. Sayı uyuşmazsa (veri değişmişse)
    güvenle yeniden üretir.

    Parametreler:
        metinler: Vektöre çevrilecek metinler.
        model_cfg: Model tanımı (önbellek dosyası adı bundan türetilir).

    Döndürür:
        (N, boyut) biçiminde embedding matrisi (numpy dizisi).
    """
    os.makedirs(EMBED_DIR, exist_ok=True)
    yol = os.path.join(EMBED_DIR, f"{model_cfg['ad']}.npy")
    if os.path.exists(yol):
        X = np.load(yol)
        if X.shape[0] == len(metinler):
            print(f"[BİLGİ] Önbellekten yüklendi: {yol}  ({X.shape})")
            return X
        print("[UYARI] Önbellek boyutu uyuşmuyor, yeniden üretiliyor.")
    print(f"[BİLGİ] Embedding üretiliyor: {model_cfg['ad']}")
    t0 = time.time()
    X = embedding_uret(metinler, model_cfg)
    np.save(yol, X)
    print(f"[BİLGİ] Kaydedildi: {yol}  ({X.shape})  süre={time.time()-t0:.1f}s")
    return X


# ---------------------------------------------------------------------------
# 3) ML ALGORİTMALARI (en az 5)
# ---------------------------------------------------------------------------
def ml_algoritmalari():
    """Karşılaştıracağımız regresyon algoritmalarını hazır hâlde döndürür.

    Her algoritma, embedding vektörlerinden 1–5 puanı tahmin etmeye çalışacak.
    Bilerek farklı "aileler" seçtik ki hangi yaklaşımın bu probleme daha uygun
    olduğunu görebilelim: doğrusal (Ridge), ağaç toplulukları (RandomForest,
    GradientBoosting), örnek tabanlı (KNN), çekirdek yöntemi (SVR) ve bir sinir
    ağı (MLP). XGBoost kuruluysa o da eklenir. Ödev en az 5 algoritma ister;
    burada 6–7 tane vardır.

    Döndürür:
        {algoritma_adı: model_nesnesi} biçiminde sözlük. Modeller henüz
        eğitilmemiştir; .fit() çağrısını ana döngü yapar.
    """
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor

    algos = {
        # 1) Doğrusal (regülarizasyonlu) regresyon - hızlı temel model.
        "Ridge": Ridge(alpha=10.0),
        # 2) Rastgele Orman - ağaç tabanlı, doğrusal olmayan ilişkiler.
        "RandomForest": RandomForestRegressor(
            n_estimators=300, n_jobs=-1, random_state=TOHUM),
        # 3) Gradyan Artırma - güçlü ağaç tabanlı boosting.
        "GradientBoosting": GradientBoostingRegressor(random_state=TOHUM),
        # 4) Destek Vektör Regresyonu - RBF çekirdeği.
        "SVR": SVR(C=10.0, kernel="rbf"),
        # 5) En Yakın Komşu - örnek tabanlı.
        "KNN": KNeighborsRegressor(n_neighbors=15, weights="distance"),
        # 6) Çok Katmanlı Algılayıcı (sinir ağı).
        "MLP": MLPRegressor(hidden_layer_sizes=(256, 64), max_iter=300,
                            early_stopping=True, random_state=TOHUM),
    }

    # XGBoost varsa GradientBoosting yerine onu da ekleyelim (opsiyonel).
    try:
        from xgboost import XGBRegressor
        algos["XGBoost"] = XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            random_state=TOHUM, verbosity=0)
    except Exception:
        print("[BİLGİ] xgboost bulunamadı, atlandı (zorunlu değil).")

    return algos


# ---------------------------------------------------------------------------
# 4) DEĞERLENDİRME (REGRESYON METRİKLERİ)
# ---------------------------------------------------------------------------
def metrik_hesapla(y_true, y_pred):
    """Bir modelin tahminlerini, çok yönlü başarı ölçütleriyle değerlendirir.

    Tek bir sayı yanıltıcı olabileceği için problemi birkaç açıdan ölçeriz:
    ortalama hata büyüklüğü (MAE, RMSE), modelin varyansı ne kadar açıkladığı
    (R²), tahmin sıralamasının gerçekle uyumu (Spearman, Pearson) ve puanları
    1–5'e yuvarlayınca tutturma oranı (tam isabet ve ±1 toleranslı). Korelasyon
    hesaplanamazsa (örn. sabit tahmin) NaN döner, program çökmez.

    Parametreler:
        y_true: Gerçek puanlar.
        y_pred: Modelin tahmin ettiği puanlar.

    Döndürür:
        Metrik adlarını değerlerine eşleyen sözlük (MAE, RMSE, R², Spearman,
        Pearson, Acc(yuvarlanmis), Acc(+-1)).
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from scipy.stats import spearmanr, pearsonr

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    try:
        spear = spearmanr(y_true, y_pred).correlation
        pears = pearsonr(y_true, y_pred)[0]
    except Exception:
        spear = pears = np.nan

    # Puanlar 1-5 olduğundan tahminleri yuvarlayıp "doğruluk" da raporlayalım.
    y_round = np.clip(np.round(y_pred), 1, 5)
    acc = float(np.mean(y_round == np.round(y_true)))
    # +-1 puan toleranslı doğruluk (regresyonda anlamlı bir ölçüt).
    acc1 = float(np.mean(np.abs(y_round - np.round(y_true)) <= 1))

    return {
        "MAE": mae, "RMSE": rmse, "R2": r2,
        "Spearman": spear, "Pearson": pears,
        "Acc(yuvarlanmis)": acc, "Acc(+-1)": acc1,
    }


# ---------------------------------------------------------------------------
# 5) ANA DENEY DÖNGÜSÜ
# ---------------------------------------------------------------------------
def deneyleri_calistir(veri, modeller=None, test_boyutu=TEST_BOYUTU):
    """Tüm deneyleri yürüten ana döngü: her embedding × her algoritma.

    İşin kalbi burası. Önce veriyi rastgele bölüp 1000 örneği test, kalanını
    eğitim olarak ayırır (sonuçlar tekrarlanabilsin diye sabit tohumla). Sonra
    her embedding modeli için vektörleri (önbellekten) alır, özellikleri
    standartlaştırır ve sırayla her ML algoritmasını eğitip test eder. Her
    deneyin metriklerini ve süresini tek bir tabloda toplar. Bir model ya da
    algoritma hata verirse o deneyi atlar, gerisini sürdürür.

    Parametreler:
        veri: "metin" ve "puan" kolonlu, hazırlanmış DataFrame.
        modeller: Kullanılacak embedding model tanımları (varsayılan: 4 model).
        test_boyutu: Test kümesindeki örnek sayısı (varsayılan 1000; veri azsa
            orantılı küçültülür).

    Döndürür:
        Her satırı bir (embedding, algoritma) deneyini ve metriklerini içeren
        DataFrame.
    """
    from sklearn.preprocessing import StandardScaler

    if modeller is None:
        modeller = EMBEDDING_MODELLERI

    os.makedirs(SONUC_DIR, exist_ok=True)
    metinler = veri["metin"].tolist()
    y = veri["puan"].values.astype(float)

    # --- Eğitim / Test bölmesi: rastgele 1000 örnek test ---
    rng = np.random.RandomState(TOHUM)
    n = len(veri)
    tboyut = min(test_boyutu, n // 5)  # veri azsa orantılı küçült
    test_idx = rng.choice(n, size=tboyut, replace=False)
    test_maske = np.zeros(n, dtype=bool)
    test_maske[test_idx] = True
    egitim_idx = np.where(~test_maske)[0]
    print(f"\n[BÖLME] Eğitim: {len(egitim_idx)}  |  Test: {tboyut}\n")

    algos = ml_algoritmalari()
    print(f"[BİLGİ] {len(algos)} ML algoritması: {list(algos.keys())}")

    kayitlar = []
    for mcfg in modeller:
        print("\n" + "=" * 70)
        print(f"EMBEDDING MODELİ: {mcfg['ad']}  ({mcfg['repo']})")
        print("=" * 70)
        try:
            X = embedding_al_veya_uret(metinler, mcfg)
        except Exception as e:
            print(f"[HATA] {mcfg['ad']} embedding üretilemedi, atlanıyor: {e}")
            continue

        X_tr, X_te = X[egitim_idx], X[test_idx]
        y_tr, y_te = y[egitim_idx], y[test_idx]

        # Özellikleri ölçekle (özellikle SVR/KNN/MLP için kritik).
        olcek = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = olcek.transform(X_tr), olcek.transform(X_te)

        for ad, model in algos.items():
            t0 = time.time()
            try:
                model.fit(X_tr_s, y_tr)
                y_pred = model.predict(X_te_s)
                m = metrik_hesapla(y_te, y_pred)
                m.update({
                    "embedding": mcfg["ad"],
                    "algoritma": ad,
                    "egitim_n": len(y_tr),
                    "test_n": len(y_te),
                    "boyut": X.shape[1],
                    "sure_s": round(time.time() - t0, 2),
                })
                kayitlar.append(m)
                print(f"  [{mcfg['ad']:18s} | {ad:16s}] "
                      f"RMSE={m['RMSE']:.3f}  MAE={m['MAE']:.3f}  "
                      f"R2={m['R2']:.3f}  rho={m['Spearman']:.3f}  "
                      f"({m['sure_s']}s)")
            except Exception as e:
                print(f"  [HATA] {mcfg['ad']} / {ad}: {e}")

    sonuc = pd.DataFrame(kayitlar)
    # Sütun sırasını düzenle.
    onsutun = ["embedding", "algoritma", "RMSE", "MAE", "R2",
               "Spearman", "Pearson", "Acc(yuvarlanmis)", "Acc(+-1)",
               "boyut", "egitim_n", "test_n", "sure_s"]
    sonuc = sonuc[[c for c in onsutun if c in sonuc.columns]]
    return sonuc


# ---------------------------------------------------------------------------
# 6) TABLOLAR ve GRAFİKLER
# ---------------------------------------------------------------------------
def ciktilari_kaydet(sonuc):
    """Deney sonuçlarını dosyaya döker: tablolar, özet pivotlar ve grafikler.

    Ham sonuç tablosunu CSV olarak kaydeder; ardından embedding × algoritma
    biçiminde özet pivot tablolar üretir. Her önemli metrik için bir ısı haritası,
    R² ve RMSE için de gruplu çubuk grafikler çizer. Son olarak en düşük RMSE'ye
    sahip kazanan kombinasyonu ayrı bir metin dosyasına yazar. Böylece rapor ve
    yorum aşamasında her şey hazır olur.

    Parametreler:
        sonuc: deneyleri_calistir'in döndürdüğü metrik tablosu.

    Döndürür:
        Yok. Tüm çıktıları 'sonuclar/' klasörüne dosya olarak yazar.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(SONUC_DIR, exist_ok=True)
    os.makedirs(GRAFIK_DIR, exist_ok=True)

    # --- Ana metrik tablosu ---
    ana_yol = os.path.join(SONUC_DIR, "metrikler.csv")
    sonuc.to_csv(ana_yol, index=False, encoding="utf-8-sig")
    print(f"\n[KAYIT] Ana tablo: {ana_yol}")

    # --- Pivot tablolar (embedding × algoritma) ---
    for metrik in ["RMSE", "MAE", "R2", "Spearman"]:
        if metrik not in sonuc.columns:
            continue
        piv = sonuc.pivot(index="embedding", columns="algoritma", values=metrik)
        piv.to_csv(os.path.join(SONUC_DIR, f"pivot_{metrik}.csv"),
                   encoding="utf-8-sig")

        # Isı haritası
        fig, ax = plt.subplots(figsize=(1.6 * piv.shape[1] + 2, 4))
        im = ax.imshow(piv.values, cmap="viridis" if metrik in ("R2", "Spearman") else "viridis_r",
                       aspect="auto")
        ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=40, ha="right")
        ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            color="white", fontsize=8)
        ax.set_title(f"{metrik} — Embedding × Algoritma")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(GRAFIK_DIR, f"heatmap_{metrik}.png"), dpi=140)
        plt.close(fig)

    # --- R² çubuk grafiği (gruplu) ---
    if "R2" in sonuc.columns:
        piv = sonuc.pivot(index="algoritma", columns="embedding", values="R2")
        fig, ax = plt.subplots(figsize=(10, 5))
        piv.plot(kind="bar", ax=ax)
        ax.set_ylabel("R²"); ax.set_title("R² — Algoritma bazında embedding karşılaştırması")
        ax.legend(title="Embedding", bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.axhline(0, color="black", lw=0.6)
        fig.tight_layout()
        fig.savefig(os.path.join(GRAFIK_DIR, "bar_R2.png"), dpi=140)
        plt.close(fig)

    # --- RMSE çubuk grafiği ---
    if "RMSE" in sonuc.columns:
        piv = sonuc.pivot(index="algoritma", columns="embedding", values="RMSE")
        fig, ax = plt.subplots(figsize=(10, 5))
        piv.plot(kind="bar", ax=ax)
        ax.set_ylabel("RMSE"); ax.set_title("RMSE — Algoritma bazında embedding karşılaştırması")
        ax.legend(title="Embedding", bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(GRAFIK_DIR, "bar_RMSE.png"), dpi=140)
        plt.close(fig)

    # --- En iyi modeli belirle (en düşük RMSE) ---
    if "RMSE" in sonuc.columns and len(sonuc):
        en_iyi = sonuc.loc[sonuc["RMSE"].idxmin()]
        with open(os.path.join(SONUC_DIR, "en_iyi_model.txt"), "w",
                  encoding="utf-8") as f:
            f.write("EN İYİ MODEL (en düşük RMSE)\n")
            f.write("=" * 40 + "\n")
            f.write(en_iyi.to_string())
        print(f"\n[SONUÇ] En iyi: {en_iyi['embedding']} + {en_iyi['algoritma']} "
              f"(RMSE={en_iyi['RMSE']:.3f}, R2={en_iyi['R2']:.3f})")

    print(f"[KAYIT] Grafikler: {GRAFIK_DIR}")


# ---------------------------------------------------------------------------
# 7) ANA AKIŞ
# ---------------------------------------------------------------------------
def main():
    """Programın giriş noktası: baştan sona tüm akışı çalıştırır.

    Komut satırı parametrelerini okur (--veri, --ornek, --test), veriyi yükler,
    isteğe bağlı olarak hızlı deneme için örneklem alır, tüm deneyleri koşturur,
    çıktıları kaydeder ve özet tabloyu ekrana basar. Doğrudan `python
    reward_model_pipeline.py` ile çağrılır.
    """
    ap = argparse.ArgumentParser(description="Ödül Modeli Eğitimi pipeline")
    ap.add_argument("--veri", default=None, help="Veri dosyası yolu (xlsx/csv)")
    ap.add_argument("--ornek", type=int, default=None,
                    help="Hızlı deneme için kullanılacak örnek sayısı (örn. 500)")
    ap.add_argument("--test", type=int, default=TEST_BOYUTU,
                    help="Test kümesi büyüklüğü (varsayılan 1000)")
    args = ap.parse_args()

    print("=" * 70)
    print(" ÖDÜL MODELİ EĞİTİMİ - PIPELINE BAŞLIYOR")
    print("=" * 70)

    yol = veri_dosyasini_bul(args.veri)
    veri = veriyi_yukle(yol)

    if args.ornek and args.ornek < len(veri):
        veri = veri.sample(args.ornek, random_state=TOHUM).reset_index(drop=True)
        print(f"[BİLGİ] Hızlı deneme: {args.ornek} örnek kullanılıyor.")

    sonuc = deneyleri_calistir(veri, test_boyutu=args.test)
    if len(sonuc) == 0:
        sys.exit("[HATA] Hiç sonuç üretilemedi. Embedding/model yüklemesini kontrol edin.")

    ciktilari_kaydet(sonuc)

    print("\n" + "=" * 70)
    print("ÖZET TABLO (RMSE'ye göre sıralı)")
    print("=" * 70)
    print(sonuc.sort_values("RMSE").to_string(index=False))
    print("\n[BİTTİ] Tüm çıktılar 'sonuclar/' klasörüne yazıldı.")


if __name__ == "__main__":
    main()
