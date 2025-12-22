# ANN ile ogrenci Sinav Performans Tahmini
# Student Performance Prediction with Artificial Neural Networks (ANN)

## 📌 Proje Tanımı
Bu proje, öğrencilerin akademik performansını etkileyen faktörleri analiz etmek ve bu performansı tahmin etmek amacıyla geliştirilmiştir. Öğrencilerin çalışma süresi, önceki akademik başarıları, uyku süreleri ve ders dışı aktiviteleri gibi değişkenler kullanılarak, yapay sinir ağları (Artificial Neural Networks – ANN) tabanlı bir regresyon modeli oluşturulmuştur.

Eğitim analitiği alanında öğrenci başarısının önceden tahmin edilmesi; erken müdahale, akademik rehberlik ve eğitim politikalarının geliştirilmesi açısından büyük önem taşımaktadır. Bu çalışma, derin öğrenme yöntemlerinin eğitim verileri üzerindeki etkinliğini incelemeyi amaçlamaktadır.

---

## 🎯 Projenin Amacı
- Öğrenci performansını etkileyen temel faktörleri incelemek  
- Akademik başarıyı sayısal olarak tahmin eden bir model geliştirmek  
- Yapay sinir ağlarının regresyon problemlerindeki başarısını değerlendirmek  

---

## 📊 Kullanılan Veri Seti
**Kaynak:** Kaggle  
**Dataset Adı:** Student Performance (Multiple Linear Regression)

Veri seti aşağıdaki değişkenleri içermektedir:

| Değişken | Açıklama |
|--------|---------|
| Hours_Studied | Öğrencinin günlük çalışma süresi |
| Previous_Scores | Önceki sınavlardan alınan notlar |
| Extracurricular_Activities | Ders dışı aktivitelere katılım durumu |
| Sleep_Hours | Günlük uyku süresi |
| Sample_Question_Papers_Practiced | Çözülen örnek soru sayısı |
| Performance_Index | Öğrencinin genel performans puanı (Hedef değişken) |

Bu veri seti, akademik başarıyı doğrudan etkileyen değişkenler içermesi nedeniyle regresyon problemi için uygundur.

---

## 🧠 Kullanılan Yöntem
Bu çalışmada **Artificial Neural Network (ANN)** tabanlı bir regresyon modeli kullanılmıştır.

### Model Özellikleri:
- Çok katmanlı ileri beslemeli yapay sinir ağı
- ReLU aktivasyon fonksiyonu
- Dropout ile overfitting önleme
- Adam optimizasyon algoritması
- Kayıp fonksiyonu: Mean Squared Error (MSE)

ANN yöntemi, klasik doğrusal regresyon yöntemlerine kıyasla değişkenler arasındaki doğrusal olmayan ilişkileri öğrenme yeteneğine sahip olması nedeniyle tercih edilmiştir.

---

## ⚙️ Model Eğitimi
- Veri seti %80 eğitim, %20 test olarak ayrılmıştır  
- Girdi verilerine **StandardScaler** ile normalizasyon uygulanmıştır  
- Mini-batch training yöntemi kullanılmıştır  
- Eğitim süreci boyunca model kaybı düzenli olarak takip edilmiştir  

---

## 📈 Model Performansı
Test verisi üzerinde elde edilen performans sonuçları:

- **MSE:** 5.76  
- **RMSE:** 2.40  
- **MAE:** 1.92  
- **R²:** 0.98  

Bu sonuçlar, modelin öğrenci performansını yüksek doğrulukla tahmin edebildiğini göstermektedir.

---
