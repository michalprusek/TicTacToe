# Robot TicTacToe

Aplikace pro hraní piškvorek s robotickým ramenem uArm Swift Pro. Systém používá počítačové vidění pro detekci herní desky a symbolů, umělou inteligenci pro herní strategii a robotické rameno pro fyzické umisťování symbolů na herní desku.

## Funkce

- Detekce herní desky a symbolů pomocí počítačového vidění
- Ovládání robotického ramene uArm Swift Pro
- Nastavitelná obtížnost AI protivníka
- Grafické uživatelské rozhraní vytvořené pomocí PyQt5
- Podpora pro ladění a vizualizaci detekce

## Požadavky

- Python 3.6+
- uArm Swift Pro s firmwarem 4.0+
- Webkamera
- Závislosti uvedené v `requirements.txt`

## Instalace

1. Klonujte repozitář:https://github.com/michalprusek/TicTacToe.git
   ```
   git clone https://github.com/michalprusek/TicTacToe.git
   cd TicTacToe
   ```

2. Nainstalujte závislosti:
   ```
   pip install -r requirements.txt
   ```

3. Nainstalujte uArm Python SDK:
   ```
   cd uArm-Python-SDK
   python setup.py install
   cd ..
   ```

## Spuštění aplikace

Spusťte hlavní aplikaci s GUI:

```
python -m app.main.main_pyqt
```

### Parametry příkazové řádky

- `--camera INDEX` - Index kamery, který se má použít (výchozí: 0)
- `--debug` - Povolení režimu ladění s dalším protokolováním a vizualizací
- `--difficulty LEVEL` - Počáteční úroveň obtížnosti (0-10, výchozí: 5)

Příklad:
```
python -m app.main.main_pyqt --camera 1 --debug --difficulty 7
```

## Testování

Spusťte testy:

```
python run_tests.py
```

Nebo s pokrytím kódu:

```
python run_tests.py --coverage
```

## Struktura projektu

- `app/` - Hlavní aplikační balíček
  - `main/` - Hlavní aplikační soubory
  - `core/` - Základní komponenty a konfigurace
  - `calibration/` - Kalibrační nástroje a soubory
  - `config/` - Konfigurační soubory
- `docs/` - Dokumentace
- `tests/` - Testy
- `utils/` - Pomocné nástroje
- `weights/` - Váhy modelů pro detekci

## Kalibrace robotického ramene

Před prvním použitím je třeba kalibrovat robotické rameno:

1. Ujistěte se, že rameno je připojeno a zapnuto
2. Spusťte kalibrační nástroj:
   ```
   python -m app.calibration.calibration
   ```
3. Postupujte podle pokynů na obrazovce

## Licence

MIT
