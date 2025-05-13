# Derin Öğrenme Tabanlı Müzik Türü Tanıma: GTZAN Veri Kümesi ile Uygulamalı Bir Yaklaşım 

Bu proje, GTZAN veri seti kullanılarak müzik dosyalarının türlerini derin öğrenme tabanlı bir sistem ile sınıflandırmayı amaçlamaktadır. Proje kapsamında sinyal işleme teknikleri, öznitelik çıkarımı ve CNN mimarisi kullanılarak ses verilerinden tür tespiti gerçekleştirilmiştir.

## Veri Seti Bilgileri

**Açıklama:**

- Veri seti, **2000-2001 yılları arasında toplanmış** olan ses verilerinden oluşmaktadır.  
- Her biri **30 saniye uzunluğunda** olan ses dosyaları, **10 farklı müzik türünü** temsil etmektedir:
  - *Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock*
- Her müzik türüne ait **100 örnek** bulunmaktadır ve toplamda **1000 adet ses dosyası** içermektedir.  
- Tüm dosyalar **.wav formatında** olup, benzer süre ve frekans aralığında olduklarından dolayı **modelleme için tutarlı ve uygun bir yapı** sunmaktadır.
--------------------------------------------------------------------------------------------------------------

---

## Kullanılan Kütüphaneler

```python
import warnings
warnings.simplefilter("ignore", UserWarning)

import os
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing

import librosa
import librosa.display
from IPython.display import Audio
import IPython.display as ipd
import scipy

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize, minmax_scale
from sklearn.feature_selection import RFECV, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D as MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.image import resize

from xgboost import XGBClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from tempfile import TemporaryFile
# Jupyter notebook içi grafikler için
%matplotlib inline
```

---

## Proje Süreci

### Ses Verisinin Görselleştirilmesi ve Ön Analiz

Projenin ilk aşamasında ses verileri analiz edilmeden önce görselleştirilmiştir. Bu süreçte:

- Her ses dosyası için dalga formu (waveform) ve spektrum grafikleri çizilerek görsel analiz yapılmıştır.
- Böylece her müzik türünün temel akustik yapısı hakkında sezgisel bilgi edinilmiştir.

>  *Bu adım, veriyle daha iyi sezgisel ilişki kurmak ve veri setindeki örüntüleri anlamak için kritik öneme sahiptir.*
---
### Öznitelik Çıkarımı

Ses verilerinden daha anlamlı ve modellemeye uygun vektörler elde etmek için çeşitli öznitelik çıkarım teknikleri kullanılmıştır:

- `MFCC` (*Mel-Frequency Cepstral Coefficients*)
- `Mel Spectrogram`
- `Spectral Centroid`
- `Spectral Rolloff`
- `Chroma Frequencies`
  
> Bu öznitelikler ile her bir ses örneği, makine öğrenmesi algoritmalarıyla işlenebilecek yapısal forma getirilmiştir.

### Model Eğitimi

Öznitelik çıkarımı tamamlandıktan sonra `CNN` (Convolutional Neural Network) mimarisi ile sınıflandırma gerçekleştirilmiştir.

- **Convolutional Neural Network (CNN)** mimarisi tercih edilmiştir.
- Veriler *train* ve *test* olarak bölünmüş, *Min-Max normalization* uygulanmıştır.
- Model yapısı:
  - 2 adet `Conv2D + MaxPooling` katmanı
  - `Flatten` ve `Dense` katmanlar
  - `Softmax` çıkışlı çok sınıflı sınıflandırma
    
> Eğitim sırasında:
> - **Loss function**: `categorical_crossentropy`
> - **Optimizer**: `Adam`
> - **Callback**: `EarlyStopping` ve `ModelCheckpoint` kullanılmıştır.
---

**Classification Report:**

```
               precision    recall  f1-score   support

       blues       0.92      0.92      0.92       290
   classical       0.95      0.95      0.95       272
     country       0.78      0.96      0.86       296
       disco       0.94      0.95      0.94       263
      hiphop       0.97      0.97      0.97       294
        jazz       0.93      0.95      0.94       269
       metal       0.96      0.93      0.95       289
         pop       0.95      0.91      0.93       296
      reggae       0.93      0.93      0.93       257
        rock       0.94      0.75      0.83       270

    accuracy                           0.92      2796
   macro avg       0.93      0.92      0.92      2796
weighted avg       0.93      0.92      0.92      2796
```
---

## Analiz ve Sonuç

Model, eğitim ve doğrulama sürecinde 0.8-0.9 doğruluk ve düşük kayıp değerleri elde etmiştir. Eğitim/doğrulama metriklerinin birbirine yakın olması, overfitting olmadığını ve iyi genelleme yaptığını gösterir. Kayıpların 25-30 epoch sonra stabilize olması, modelin verimli öğrendiğini ve optimize edildiğini kanıtlar. Sonuç olarak, model kararlı ve güvenilir bir performans sergilemiştir.

**Sonuç:**

**Genel Değerlendirme:**

Modelin sınıflandırma performansı her müzik türü için yüksek seviyededir. Precision, recall ve f1-score değerleri dengelidir. Overfitting gözlenmemiştir. Eğitim sırasında stabil bir öğrenme süreci elde edilmiştir.

---

## Proje Detayları

- Veri Seti: [Kaggle](https://www.kaggle.com/) üzerinden temin edilmiştir.
- Python Sürümü: 3.10.14
- Lisans: Yok

Proje hakkında daha fazla bilgi veya kodlar için [Kaggle](https://www.kaggle.com/code/itselif/m-zik-t-r-alg-lama-ve-s-n-fland-rma-sistemi-dl) adresine göz atabilirsiniz.

