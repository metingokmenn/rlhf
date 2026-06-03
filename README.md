# Ödül Modeli Eğitimi (Reward Model)

**Bilgi Teknolojileri / Yapay Zeka Teknikleri — Ödev 1 (2025/2)**

Bir LLM'in ürettiği cevabın kalitesini, metin embedding'lerinden yola çıkarak
**1–5 arası bir puanla** tahmin eden bir ödül modeli. Çalışma, **4 embedding
modeli × (en az) 5 ML algoritması = 20+ eğitim/test deneyini** karşılaştırır.

---

## İçindekiler
1. [Proje özeti](#1-proje-özeti)
2. [Veri kümesi](#2-veri-kümesi)
3. [Yöntem](#3-yöntem)
4. [Kurulum](#4-kurulum)
5. [Çalıştırma adımları](#5-çalıştırma-adımları)
6. [Çıktılar](#6-çıktılar)
7. [Dosya yapısı](#7-dosya-yapısı)
8. [Donanım ve süre](#8-donanım-ve-süre)
9. [Sık karşılaşılan sorunlar](#9-sık-karşılaşılan-sorunlar)
10. [Teslim kontrol listesi](#10-teslim-kontrol-listesi)

---

## 1. Proje özeti

Amaç, dil modeli hizalamasında kullanılan **reward model** mantığının
basitleştirilmiş bir uygulamasını kurmaktır: bir **soru (S)**, modelin
**düşünme süreci (D)** ve **cevap metni (C)** verildiğinde, bir insanın bu
cevaba vereceği değerlendirme puanını otomatik tahmin etmek.

Akış kısaca: `S + D + C metni → embedding (vektör) → regresyon modeli → puan (1–5)`.

---

## 2. Veri kümesi

`veri.xlsx` (12.134 satır, 4 kolon):

| Kolon | Açıklama | Rol |
|-------|----------|-----|
| `Sorunuz` | Kullanıcının sorusu | S (girdi metni) |
| `CosmosLLM düşünme süreci` | Modelin akıl yürütmesi | D (girdi metni) |
| `CosmosLLM cevabı` | Modelin nihai cevabı | C (girdi metni) |
| `Değerlendirme Puanınız` | İnsan değerlendirmesi | Hedef (çıktı) |

**Puan etiketi sıralı (ordinal) metindir** ve kod tarafından otomatik olarak
sayıya eşlenir:

| Etiket | Sayı | Adet |
|--------|------|------|
| çok kötü | 1 | 626 |
| kötü | 2 | 898 |
| orta | 3 | 1.748 |
| iyi | 4 | 4.309 |
| çok iyi | 5 | 4.553 |

> **Not:** Veri sınıf bakımından **dengesizdir** ("iyi" ve "çok iyi" baskın).
> Bu, RMSE/MAE yorumlanırken akılda tutulmalıdır.

Kod, kolon adlarını otomatik tespit eder; farklı isimlendirilmiş bir veride de
çalışır. Tespit edilemezse dosyayı `--veri` parametresiyle verin.

---

## 3. Yöntem

**Adım adım:**

1. **Metin birleştirme.** S, D ve C tek metinde birleştirilir:
   `"Soru: …\nDüşünme: …\nCevap: …"` (modelin bölümleri ayırt edebilmesi için
   etiketli).
2. **Embedding.** Birleşik metin 4 modelle vektöre çevrilir. Vektörler
   `embeddings/` altında **önbelleğe** alınır (tekrar üretilmez).

   | Embedding modeli | HuggingFace repo |
   |---|---|
   | turkish-e5-large | `ytu-ce-cosmos/turkish-e5-large` |
   | jina-v5-text-small | `jinaai/jina-embeddings-v5-text-small` |
   | harrier-oss-0.6b | `microsoft/harrier-oss-v1-0.6b` |
   | qwen3-embed-0.6b | `Qwen/Qwen3-Embedding-0.6B` |

3. **Bölme.** Veri rastgele bölünür: **1.000 örnek test**, geri kalan (~11.134)
   eğitim. Tekrarlanabilirlik için sabit tohum (`TOHUM=42`).
4. **Ölçekleme.** Özellikler `StandardScaler` ile standartlaştırılır
   (SVR/KNN/MLP için kritik).
5. **Eğitim.** Her embedding, aşağıdaki regresyon algoritmalarına girdi verilir:

   | Algoritma | Tür |
   |---|---|
   | Ridge | Doğrusal (regülarizasyonlu) |
   | RandomForest | Ağaç topluluğu (bagging) |
   | GradientBoosting | Ağaç topluluğu (boosting) |
   | SVR | Destek vektör regresyonu (RBF) |
   | KNN | Örnek tabanlı |
   | MLP | Sinir ağı |
   | XGBoost* | Boosting (*kuruluysa) |

6. **Değerlendirme.** Test kümesinde **MAE, RMSE, R², Spearman, Pearson** ve
   yuvarlanmış tahmin doğruluğu (±0 ve ±1) hesaplanır.
7. **Raporlama.** Tablolar (CSV) ve grafikler (PNG) üretilir.

---

## 4. Kurulum

**Sanal ortam (önerilir):**

```bash
cd /Users/metingokmen/software/python/ml_hw2

python3 -m venv venv
source venv/bin/activate          # macOS/Linux. Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> Terminali her açtığında çalıştırmadan önce `source venv/bin/activate` yap;
> prompt'ta `(venv)` görünmeli. Bitince `deactivate`.
> PyTorch'un standart pip sürümü Mac'te **MPS/Metal** desteğini içerir.

---

## 5. Çalıştırma adımları

```bash
# 1) ÖNCE hızlı doğrulama — tüm akışı küçük örneklemle dene
python reward_model_pipeline.py --ornek 500

# 2) Sorun yoksa TAM çalıştırma (12.134 satır, 20+ deney)
python reward_model_pipeline.py
```

**Faydalı parametreler:**

| Parametre | Açıklama | Varsayılan |
|---|---|---|
| `--veri YOL` | Veri dosyası yolu (xlsx/csv) | otomatik bulunur |
| `--ornek N` | Hızlı deneme için N örnek kullan | tümü |
| `--test N` | Test kümesi büyüklüğü | 1000 |

---

## 6. Çıktılar

`reward_model_pipeline.py` çalıştıktan sonra:

```
sonuclar/
├── metrikler.csv          # 20+ deneyin tüm metrikleri (ana tablo)
├── pivot_RMSE.csv         # embedding × algoritma özeti (ve MAE/R2/Spearman)
├── pivot_R2.csv
├── en_iyi_model.txt       # en düşük RMSE'li kombinasyon
└── grafikler/
    ├── heatmap_RMSE.png   # ısı haritaları
    ├── heatmap_R2.png
    ├── bar_R2.png         # gruplu çubuk grafikler
    └── bar_RMSE.png
embeddings/
└── *.npy                  # embedding önbelleği (yeniden üretilmez)
```

---

## 7. Dosya yapısı

```
ml_hw2/
├── reward_model_pipeline.py   # ANA PIPELINE (veri→embedding→eğitim→metrik→grafik)
├── requirements.txt           # Gerekli paketler
├── README.md                  # Bu dosya
├── .gitignore
└── veri.xlsx                  # Veri kümesi (teslimde ayrıca gönderilir)
```

---

## 8. Donanım ve süre

GPU önerilir ama zorunlu değil. Kod cihazı otomatik seçer:
**NVIDIA (cuda) → Apple Silicon (mps) → CPU**.

**Apple M4 Pro için kaba beklentiler (tam veri):**

| Aşama | Süre (MPS) | Not |
|---|---|---|
| Embedding (4 model × 12k metin) | ~30–90 dk | Darboğaz burası; önbelleğe alınır |
| 20+ ML eğitimi | ~20–40 dk | Tek yavaş olan SVR (RBF, ~11k örnek) |

> `trust_remote_code`'lu modeller (jina/qwen3) MPS yerine CPU'ya düşerse
> embedding süresi uzar ama yine de tamamlanır. Bellek sorun değildir
> (modeller tek tek yüklenir; embedding'ler ~50 MB/model).

---

## 9. Sık karşılaşılan sorunlar

- **Bir embedding modeli yüklenemiyor / repo bulunamadı.** Kod o modeli atlar,
  kalanlarla devam eder. Repo adını HuggingFace'te doğrulayın.
- **`sentence-transformers` ile yükleme hatası.** Kod otomatik olarak
  `transformers` + ortalama (mean) pooling yöntemine geri düşer.
- **Mac'te `attention` uyarısı.** Sorun değil; eager attention kullanılır.
- **Metinler çok uzun.** 512 token'a kırpılır; çok uzun cevapların sonu kesilir
  (ödev için kabul edilebilir, raporda belirtilir).

---

## 10. Teslim kontrol listesi

- [ ] `python reward_model_pipeline.py` tam veriyle çalıştı, `sonuclar/` doldu
- [ ] 5–8 dk video çekildi ve YouTube'a yüklendi, bağlantı rapora eklendi
- [ ] Kod + rapor + video bağlantısı `online.yildiz.edu.tr`'ye yüklendi

---

*Algoritma adımları ve değişken açıklamaları için `reward_model_pipeline.py`
içindeki Türkçe yorumlara bakınız.*
