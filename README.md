# ✈️ Sieć neuronowa do predykcji cen biletów lotniczych

Projekt realizowany w ramach zajęć z **Sztucznej Inteligencji**. Celem było stworzenie własnej implementacji **wielowarstwowej sieci neuronowej (MLP)** i wykorzystanie jej do **predykcji ceny biletów lotniczych** na podstawie wybranych cech lotu.

## 🧠 Opis projektu

Model został zaimplementowany od podstaw w języku **Python**, bez użycia bibliotek typu Keras czy PyTorch. Dane wejściowe pochodziły z pliku `loty_clean.xlsx`, zawierającego informacje o locie, takie jak:

- czas wylotu i przylotu
- długość lotu
- liczba przesiadek
- linie lotnicze
- klasa biletu
- typ bagażu
- liczba dni do odlotu

### 🧪 Eksperyment: dobór zmiennych

Częścią projektu była **analiza wpływu poszczególnych zmiennych na skuteczność działania sieci**. Dla każdej zmiennej:

1. Usuwano ją ze zbioru danych
2. Dane były przetwarzane zgodnie ze skryptem preprocessingowym (bez one-hot)
3. Trenowano sieć neuronową z bazowymi parametrami
4. Mierzono metryki: `MSE`, `MAE`, `MAPE`, `R²`

🔍 Dzięki temu udało się określić, które zmienne są **kluczowe**, a które mają **niewielki wpływ** lub wręcz wprowadzają szum.

## 🧩 Struktura projektu

