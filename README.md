# Freqtrade Strategies Collection

<p align="center">
  <img src="https://img.shields.io/badge/strategies-1,072%2B-6C5CE7" alt="Strategies">
  <img src="https://img.shields.io/badge/python-3.8%2B-00CEC9" alt="Python">
  <img src="https://img.shields.io/badge/sources-8-00B894" alt="Sources">
  <img src="https://img.shields.io/badge/ML%20strategies-217%2B-FDCB6E" alt="ML">
</p>

<p align="center">
  <b>Крупнейшая коллекция торговых стратегий для Freqtrade</b><br>
  1 072+ готовых стратегий из 8 источников
</p>

---

## 🚀 Быстрый старт

```bash
git clone https://github.com/geniok1980/Freqtrade-Strategies.git
cd Freqtrade-Strategies
cp peet-crypto/*.py /path/to/freqtrade/user_data/strategies/
freqtrade trade --strategy ClucHAnix_5mTB1
```

## 📦 Состав

| Источник | Стратегий | Описание |
|----------|----------:|----------|
| [peet-crypto](./peet-crypto) | 429 | ML-стратегии (XGBoost, LightGBM, нейросети) |
| [nateemma-strategies](./nateemma-strategies) | 381 | SimpleStrategies, индикаторные стратегии |
| [werkkrew-strategies](./werkkrew-strategies) | 122 | Solipsis, Schism, Cluckie и другие |
| [official-strategies](./official-strategies) | 67 | Официальные стратегии Freqtrade |
| [nostalgia-for-infinity](./nostalgia-for-infinity) | 42 | Стратегии с бектестами |
| [ssssi-strategies](./ssssi-strategies) | 22 | ClucHAnix, Binance-оптимизации |
| [strategies-that-work](./strategies-that-work) | 5 | Проверенные рабочие стратегии |
| [user_data](./user_data) | 4 | Примеры от Freqtrade |
| **Итого** | **1 072+** | |

## 🏷️ Категории

| Категория | Описание | Количество |
|-----------|----------|:----------:|
| 🤖 ML-стратегии | XGBoost, LightGBM, RandomForest, LSTM | 217+ |
| ⚡ Скальпинг | Быстрые стратегии для 1m-5m | 368+ |
| 📈 Трендовые | EMA/SMA, ADX, MACD | 718+ |
| 🔄 Моментум | RSI, Stochastic, CCI | 643+ |
| 🛡️ DCA/Сетки | Мартингейл, усреднение | 25+ |
| 📉 Mean Reversion | Bollinger Bands, контр-тренд | 15+ |
| 🔮 Фьючерсы | Плечо, шорт, хедж | 190+ |
| 📦 Hyperopt | Loss-функции | 40+ |

## 🛠️ Установка

### Требования
- Python 3.8+
- Freqtrade (`pip install freqtrade`)

### Установка всех стратегий

```bash
cd /path/to/freqtrade/user_data/strategies/
git clone https://github.com/geniok1980/Freqtrade-Strategies.git
# Или скопируйте выборочно:
cp /path/to/Freqtrade-Strategies/official-strategies/user_data/strategies/MultiMa.py .
```

### Запуск с бектестом

```bash
freqtrade backtesting --strategy MultiMa --timerange 20240101-20241231
```

## 📚 Лицензия

Стратегии распространяются по лицензиям их авторов (в основном MIT).
Данная сборка — открытая, для личного и коммерческого использования.

## 🤝 Как добавить свою стратегию

1. Сделайте форк репозитория
2. Добавьте свой .py файл в соответствующую папку
3. Создайте Pull Request

## 💎 Готовый продукт

Если вам нужна **упакованная версия с поддержкой, документацией и обновлениями** — посмотрите [Freqtrade Strategy Pack](https://geniok.ru/strategy-pack/).

- 1 072+ стратегий в одном ZIP
- Каталог с фильтрацией по категориям
- Ежемесячные обновления
- Поддержка по email и в Telegram
- **Бесплатно** — 5 стратегий для старта

---

<p align="center">
  Сделано с ❤️ для сообщества Freqtrade<br>
  <a href="https://geniok.ru">geniok.ru</a>
</p>
