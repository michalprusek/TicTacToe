# Kompletní protokol aplikace TicTacToe s robotickou rukou

## Přehled aplikace

Toto je pokročilá aplikace pro hraní piškvorek (TicTacToe) pomocí robotické ruky uArm Swift Pro. Aplikace kombinuje počítačové vidění, umělou inteligenci a řízení robotické ruky pro fyzické hraní piškvorek na papíře.

## 1. Architektura aplikace

### 1.1 Vstupní body
- **Hlavní spuštění**: `python -m app.main.main_pyqt` nebo `python run.py`
- **Parametry příkazové řádky**:
  - `--camera <index>` - Index kamery (výchozí: 0)
  - `--debug` - Zapnutí debug režimu
  - `--difficulty <0-10>` - Obtížnost AI (výchozí: 5)
  - `--max-fps <fps>` - Maximální FPS detekce

### 1.2 Vláknová architektura
Aplikace používá tři hlavní vlákna:

1. **Hlavní vlákno (PyQt5 GUI)**
   - Zpracování uživatelského rozhraní
   - Aktualizace zobrazení herní desky
   - Zpracování výsledků detekce z pracovních vláken
   - Řízení herního stavu

2. **Detekční vlákno (DetectionThread)**
   - Běží na ~2 FPS pro zpracování obrazu
   - Používá YOLO modely pro detekci
   - Výsledky ukládá do fronty pro hlavní vlákno

3. **Vlákno robotické ruky (ArmThread)**
   - Asynchronní zpracování příkazů pro robotickou ruku
   - Fronta příkazů pro pohyby
   - Neblokující API pro ovládání

### 1.3 Komunikace mezi komponentami
Aplikace používá PyQt5 signály pro oddělou komunikaci:
- `GameController.status_changed` → aktualizace stavové lišty
- `CameraController.game_state_updated` → aktualizace herního stavu z detekce
- `CameraController.grid_incomplete` → upozornění na neúplnou mřížku
- `ArmMovementController.arm_connected` → stav připojení ruky

## 2. Systém detekce a rozpoznávání hry

### 2.1 Dvoustupňová detekce

#### Detekce mřížky (Grid Detection)
1. **YOLO Pose Model** detekuje 16 klíčových bodů (4x4 mřížka)
2. **Validace mřížky**:
   - Kontrola vzdáleností mezi body
   - Kontrola úhlů (pravoúhlost)
   - RANSAC pro homografii
3. **Řazení bodů**: Automatické seřazení bodů do kanonického 4x4 pořadí

#### Detekce symbolů (Symbol Detection)
1. **YOLO Object Detection** pro detekci X a O
2. **DŮLEŽITÉ**: V modelu jsou prohozené labely (X je detekováno jako O a naopak)
3. **Mapování do buněk**: Polygon containment test pro přiřazení symbolů

### 2.2 Transformace souřadnic
```
Pixely kamery → Homografie → Normalizovaný prostor mřížky → Mapování buněk
```

### 2.3 Robustnost detekce
- **Sledování ztráty mřížky**: Timeout 2 sekundy
- **Retry logika**: Maximálně 2 pokusy na detekci
- **Fallback**: Přepnutí na tah člověka při selhání

## 3. Ovládání robotické ruky

### 1 Hierarchie ovládání
1. **ArmMovementController** - Vysokoúrovňové pohyby (kreslení X, O)
2. **ArmThread** - Asynchronní zpracování příkazů
3. **ArmController** - Nízkoúrovňové ovládání uArm Swift Pro

### 3.2 Transformace souřadnic kamera → robot

#### Kalibrační matice
Aplikace používá 3x3 perspektivní transformační matici uloženou v `hand_eye_calibration.json`:
```python
[u]   [h11 h12 h13] [x]
[v] = [h21 h22 h23] [y]
[1]   [h31 h32  1 ] [1]
```

#### Proces transformace
1. Detekce pozice v pixelech kamery (u, v)
2. Aplikace inverzní kalibrační matice
3. Získání fyzických souřadnic robota (x, y)
4. Přidání Z souřadnice podle typu pohybu

### 3.3 Typy pohybů
- **Kreslení X**: Dvě diagonální čáry s mezizvednutím
- **Kreslení O**: Kruh kreslený po segmentech (8 bodů)
- **Optimalizované rychlosti**:
  - Cestovní pohyby: 10000 jednotek/s
  - Kreslící pohyby: 3000 jednotek/s

### 3.4 Výškové úrovně
- **Touch Z**: Výška pro dotyk papíru (kreslení)
- **Safe Z**: Bezpečná výška pro přesuny
- **Neutral Z**: Parkovací pozice

## 4. Kalibrační proces

### 4.1 Spuštění kalibrace
```bash
python -m app.calibration.calibration
```

### 4.2 Kroky kalibrace
1. **Detekce mřížky**: Automatická detekce 16 bodů pomocí YOLO
2. **Korekce homografie**: Aplikace RANSAC pro perspektivní korekci
3. **Sběr bodů**: 
   - Uživatel pohybuje rukou na každý bod mřížky
   - Zaznamenání UV (kamera) → XY (robot) korespondencí
4. **Výpočet matice**: Perspektivní transformace metodou nejmenších čtverců
5. **Kalibrace výšek**:
   - Touch Z: Dotyk papíru
   - Safe Z: Bezpečná výška
   - Neutral: Parkovací pozice

### 4.3 Uložení kalibrace
Kalibrační data se ukládají do `app/calibration/hand_eye_calibration.json`:
```json
{
  "calibration_matrix": [[h11, h12, h13], [h21, h22, h23], [h31, h32, 1]],
  "touch_z": -65,
  "safe_z": -30,
  "neutral_position": [200, 0, 20]
}
```

## 5. Herní logika a AI strategie

### 5.1 Správa herního stavu
- **Autoritativní stav**: `GameController.authoritative_board` je jediný zdroj pravdy
- **Detekce tahu**: Lichý počet symbolů = tah robota
- **Koordinace tahů**: Příznaky `arm_move_in_progress` a `waiting_for_detection`

### 5.2 AI Strategie

#### BernoulliStrategySelector
- Obtížnost 0-10 mapována na pravděpodobnost 0.0-1.0
- Výběr mezi náhodnou a minimax strategií
- Vyšší obtížnost = více optimálních tahů

#### Minimax algoritmus
- Alpha-beta prořezávání pro efektivitu
- Heuristiky pro běžné scénáře (střed, rohy)
- Prioritizace rychlých výher, odkládání proher
- Hodnocení pozic: +10 výhra, -10 prohra, 0 remíza

### 5.3 Detekce konce hry
- Kontrola 8 výherních kombinací
- Detekce remízy (plná deska)
- Animace výherní čáry s gradientem

## 6. Uživatelské rozhraní

### 6.1 Hlavní komponenty
- **TicTacToeApp**: Hlavní okno s fullscreen zobrazením
- **TicTacToeBoard**: Display-only widget herní desky
- **StatusManager**: Centralizovaný systém notifikací
- **GameStatisticsWidget**: Sledování výher/proher/remíz

### 6.2 Vizuální prvky
- Tmavé téma s anti-aliased grafikou
- Zvýraznění posledních tahů s fade animací
- Gradientová výherní čára
- Stavové ikony a notifikace

### 6.3 Debug režim
- Zobrazení detekovaných bodů mřížky
- Vizualizace homografie
- Confidence hodnoty detekce
- FPS a latence

## 7. Konfigurace

### 7.1 Struktura konfigurace
```python
AppConfig
├── GameDetectorConfig
│   ├── camera_index
│   ├── grid_model_path
│   ├── symbol_model_path
│   └── confidence_thresholds
├── ArmControllerConfig
│   ├── port
│   ├── speeds
│   └── heights
└── GameConfig
    ├── difficulty
    ├── ui_language
    └── timing_settings
```

### 7.2 Důležité parametry
- **Grid confidence**: 0.6 (minimální spolehlivost detekce bodů)
- **Symbol confidence**: 0.7 (minimální spolehlivost detekce X/O)
- **Detection timeout**: 5 sekund
- **Max retry count**: 2 pokusy

## 8. Řešení problémů a robustnost

### 8.1 Ztráta mřížky
- Automatická pauza hry
- Varování "Grid not fully visible"
- Obnovení po znovudetekci

### 8.2 Selhání detekce tahu
- Timeout 5 sekund na detekci
- 2 pokusy opakování
- Fallback na lidský tah

### 8.3 Race conditions
- Detekční výsledky ve frontě
- Žádné GUI aktualizace z worker vláken
- Atomické změny stavu

## 9. Utility a pomocné skripty

### 9.1 Testování detekce
```bash
python scripts/utils/symbol_tester.py  # Test detekce symbolů
python scripts/utils/webcam_recorder.py  # Nahrávání z kamery
```

### 9.2 Sběr dat pro trénování
```bash
python scripts/utils/save_frames_on_key.py  # Ukládání snímků
python scripts/utils/annotate_frames.py  # Anotace dat
```

## 10. Testování

### 10.1 Spuštění testů
```bash
pytest  # Všechny testy
pytest --cov=app --cov-report=html  # S pokrytím
pytest tests/test_game_logic_pytest.py -v  # Konkrétní modul
```

### 10.2 Testovací architektura
- Pure pytest framework (bez unittest)
- 80%+ pokrytí kritických modulů
- Mockování hardwaru (kamera, robotická ruka)
- Parametrizované testy pro různé scénáře

## 11. Známé problémy a omezení

1. **Prohozené labely v YOLO modelu** - X je detekováno jako O a naopak
2. **Citlivost na osvětlení** - Nejlepší výsledky při rovnoměrném osvětlení
3. **Požadavek na viditelnost celé mřížky** - Všech 16 bodů musí být viditelných
4. **Kalibrace platná pouze pro fixní pozici** - Při přemístění kamery/robota nutná rekalibrace

## 12. Doporučení pro provoz

1. **Osvětlení**: Rovnoměrné, bez ostrých stínů
2. **Papír**: Bílý, matný povrch
3. **Pero**: Tmavé, dobře viditelné
4. **Pozice kamery**: Kolmo nad herní plochou
5. **Kalibrace**: Provést před každým použitím po přemístění

Tato aplikace představuje pokročilé řešení pro fyzické hraní piškvorek s robustní detekcí, adaptivní AI a spolehlivým ovládáním robotické ruky.