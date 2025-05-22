# PyQt Testing Guide

Tato příručka popisuje bezpečný způsob testování PyQt aplikací v projektu TicTacToe s cílem zabránit segmentačním chybám (segmentation faults).

## Obsah

- [Úvod](#úvod)
- [Segmentační chyby při testování PyQt](#segmentační-chyby-při-testování-pyqt)
- [Řešení problému](#řešení-problému)
- [Bezpečný přístup k testování](#bezpečný-přístup-k-testování)
- [Používání PyQtGuiTestCaseSafe](#používání-pyqtguitestcasesafe)
- [Dostupné utility](#dostupné-utility)
- [Příklady testování](#příklady-testování)
- [Nejlepší postupy](#nejlepší-postupy)

## Úvod

Testování PyQt aplikací může být složité kvůli problémům se segmentačními chybami, zejména při spouštění testů v kontinuální integraci nebo bez GUI. V projektu TicTacToe jsme implementovali robustní a bezpečný způsob testování PyQt5 aplikací pomocí pytest a pytest-qt.

## Segmentační chyby při testování PyQt

Segmentační chyby (segmentation faults) jsou běžným problémem při testování PyQt aplikací. Tyto chyby mohou nastat z několika důvodů:

1. **Inicializace QApplication**: Opakovaná inicializace QApplication v různých testech.
2. **Ukončení QApplication**: Volání QApplication.exit() v testech.
3. **Přístup ke grafickým objektům**: Přístup k GUI objektům po jejich zničení.
4. **Renderování**: Problém s grafickým renderováním na systémech bez displeje.

V našem projektu jsme zaznamenali segmentační chyby zejména při spouštění testů třídy TicTacToeApp, která dědí z QMainWindow.

## Řešení problému

Pro řešení těchto problémů jsme implementovali několik opatření:

1. **Offscreen rendering**: Použití platformy "offscreen" pro testování bez zobrazení GUI.
2. **Singleton QApplication**: Vytvoření pouze jedné instance QApplication pro celou testovací relaci.
3. **Mock implementace**: Použití mock třídy, která nedědí z PyQt tříd, pro testování logiky.
4. **pytest-qt**: Použití specializovaného pytest pluginu pro PyQt.

## Bezpečný přístup k testování

Implementovali jsme dva přístupy k bezpečnému testování PyQt aplikací:

### 1. MockTicTacToeAppSafe

Třída `MockTicTacToeAppSafe` v `conftest_qt_safe.py` poskytuje bezpečnou implementaci TicTacToeApp, která:

- Nedědí z žádné PyQt třídy, čímž předchází problémům s PyQt objekty
- Implementuje stejné rozhraní jako TicTacToeApp pro konzistentní testování
- Používá mocky pro všechny UI komponenty
- Neprovádí žádné operace s PyQt widgety

### 2. pytest-qt s offscreen platformou

Pro testy, které skutečně potřebují PyQt funkcionalitu:

- Konfigurace `QT_QPA_PLATFORM=offscreen` v proměnné prostředí nebo pyproject.toml
- Použití fixture `qtbot` z pytest-qt pro interakci s widgety
- Nastavení `WA_DontShowOnScreen` pro widgety, které je třeba testovat

## Používání PyQtGuiTestCaseSafe

Třída `PyQtGuiTestCaseSafe` v `conftest_qt_safe.py` poskytuje metody pro bezpečné vytváření a testování PyQt komponent:

```python
def test_my_pyqt_component(self):
    # Vytvoření aplikace a mock instance
    app, qt_app = PyQtGuiTestCaseSafe.create_test_app()
    
    # Nastavení pro test
    app.some_property = some_value
    
    # Testování funkčnosti
    app.some_method()
    
    # Ověření výsledku
    assert app.some_result == expected_result
```

## Dostupné utility

Spolu s bezpečným testem jsme poskytli několik utilit pro zjednodušení testování:

1. **GameEndCheckTestUtilsSafe**: Utility pro testování konce hry
2. **EventHandlingTestUtilsSafe**: Utility pro testování zpracování událostí
3. **PyQtGuiTestCaseSafe**: Základní třída pro testy PyQt GUI

Tyto utility nabízejí standardizované metody pro běžné úkoly testování.

### Příklad použití GameEndCheckTestUtilsSafe

```python
def test_check_game_end_no_winner(self):
    """Test check_game_end method s žádným vítězem."""
    GameEndCheckTestUtilsSafe.test_check_game_end_no_winner(self.app)
```

### Příklad použití EventHandlingTestUtilsSafe

```python
def test_debug_button_enable(self):
    """Test povolení debug módu přes debug button."""
    EventHandlingTestUtilsSafe.test_debug_button_enable(self.app)
```

## Příklady testování

### Testování s MockTicTacToeAppSafe

Tento přístup je ideální pro testování logiky bez PyQt závislostí:

```python
class TestGameEndCheckSafe:
    """Testy pro kontrolu konce hry v TicTacToeApp pomocí bezpečných testovacích metod."""
    
    def setup_method(self):
        """Nastavení testovacího prostředí."""
        # Vytvoření aplikace
        self.app, self.qt_app = PyQtGuiTestCaseSafe.create_test_app()
        
        # Konfigurace aplikace pro testy
        GameEndCheckTestUtilsSafe.prepare_app_for_testing(self.app)
    
    def test_check_game_end_no_winner(self):
        """Test metody check_game_end bez vítěze."""
        GameEndCheckTestUtilsSafe.test_check_game_end_no_winner(self.app)
```

### Testování s pytest-qt (qtbot)

Pro testy, které potřebují skutečnou interakci s widgety:

```python
def test_with_qtbot_fixture(qtbot, safe_tic_tac_toe_app):
    """Test použití fixture qtbot s naším bezpečným mock aplikace."""
    # Test zpracování událostí
    safe_tic_tac_toe_app.debug_mode = False
    safe_tic_tac_toe_app.handle_debug_button_click()
    
    # Kontrola, že debug_mode byl nastaven na True
    assert safe_tic_tac_toe_app.debug_mode is True
    
    # Kontrola, že text debug_button byl aktualizován
    safe_tic_tac_toe_app.debug_button.setText.assert_called_once_with("Vypnout debug")
```

## Nejlepší postupy

Při testování PyQt aplikací dodržujte následující postupy:

1. **Preferujte MockTicTacToeAppSafe**: Pro většinu testů používejte bezpečnou mock implementaci.

2. **Offscreen platforma**: Nastavte `QT_QPA_PLATFORM=offscreen` v pytest.ini nebo pyproject.toml.

3. **Jedna instance QApplication**: Používejte fixture `qt_app` pro správu QApplication.

4. **Čištění zdrojů**: Ujistěte se, že všechny widgety jsou správně uzavřeny po testu.

5. **Vyhněte se děděni z PyQt tříd**: Mock třídy by neměly dědit z PyQt tříd.

6. **Používejte qtbot**: Pro interakci s widgety použijte fixture `qtbot` místo přímého volání metod.

7. **Vyhněte se QApplication.exit()**: Nepoužívejte QApplication.exit() v testech.

8. **Mockujte externí závislosti**: Mockujte hardware (kamera, robotické rameno).

Dodržováním těchto postupů můžete psát spolehlivé testy PyQt aplikací bez obav ze segmentačních chyb.