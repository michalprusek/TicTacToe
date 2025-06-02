# Robot TicTacToe

Aplikace pro hraní piškvorek s robotickým ramenem uArm Swift Pro pomocí počítačového vidění a AI.

## Požadavky

- Python 3.11+ (doporučeno)
- uArm Swift Pro (volitelné)
- Webkamera (volitelné)
- Operační systém: Windows, macOS, Linux

## Instalace a spuštění

### Bash (Linux/Mac)

```bash
# Stáhnout z GitLab (semestral branch)
git clone git@gitlab.fit.cvut.cz:BI-PYT/B242/prusemic.git -b semestral
cd prusemic

# Vytvořit virtuální prostředí (doporučeno)
python -m venv venv
source venv/bin/activate

# Nainstalovat závislosti
pip install -r requirements.txt

# Spustit aplikaci
python -m app.main.main_pyqt

# nebo
python run.py
```

### PowerShell (Windows)

```powershell
# Stáhnout z GitLab (semestral branch)
git clone git@gitlab.fit.cvut.cz:BI-PYT/B242/prusemic.git -b semestral
cd prusemic

# Vytvořit virtuální prostředí (doporučeno)
python -m venv venv
venv\Scripts\Activate.ps1

# Nainstalovat závislosti
pip install -r requirements.txt

# Spustit aplikaci
python -m app.main.main_pyqt
```

### Alternativní instalace přes HTTPS

```bash
# Pokud nemáte SSH klíč nastaven
git clone https://gitlab.fit.cvut.cz/BI-PYT/B242/prusemic.git -b semestral
cd prusemic
```

## Volitelné parametry

```bash
# S volbou kamery a debug módem
python -m app.main.main_pyqt --camera 1 --debug --difficulty 7

# Kalibrace robota (pouze s uArm)
python -m app.calibration.calibration

# Spuštění testů
pytest

# Coverage analýza
pytest --cov=app --cov-report=term-missing
pytest --cov=app --cov-report=html

# Spuštění lintingu
pylint app

# Spuštění všech kontrol kvality kódu
pytest && pylint app
```

## Funkce

- Detekce herní desky pomocí počítačového vidění (YOLO)
- AI protivník s nastavitelnou obtížností (1-10)
- Ovládání robotického ramene uArm Swift Pro (volitelné)
- PyQt5 grafické rozhraní s moderním designem
- Cross-platform kompatibilita (Windows, macOS, Linux)
- Automatická kalibrace kamery a robota
- Statistiky her a analýza výkonu

## Architektura

Aplikace je rozdělena do několika modulů:

- `app/main/` - Hlavní aplikační logika a GUI
- `app/core/` - Základní herní logika a algoritmy
- `app/calibration/` - Kalibrace robota a kamery
- `tests/` - Kompletní testovací suite
- `docs/` - Dokumentace architektury a nasazení

## Řešení problémů

### Chyby s kamerou
```bash
# Zkontrolujte dostupné kamery
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
```

### Chyby s robotem
```bash
# Zkontrolujte připojení uArm
python -m app.calibration.calibration
```

### Chyby s PyQt5
```bash
# Reinstalace PyQt5
pip uninstall PyQt5
pip install PyQt5>=5.15.7
```

## Vývoj

Pro vývoj aplikace:

```bash
# Instalace vývojových závislostí
pip install -r requirements.txt

# Spuštění testů s pokrytím
pytest --cov=app --cov-report=html

# Kontrola kvality kódu
pylint app

# Formátování kódu (volitelné)
black app tests
```

## Licence

MIT