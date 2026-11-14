# Установка этапа III в существующий COS1

База: `COS1_STAGE2_COMPLETE_2026-11-14.zip`. Это добавочное обновление, не новый полный релиз. Существующие `.venv`, `.idea`, Sources Root, исходные данные и этапы I–II сохраняются. Ни один существующий файл проекта не требуется заменять.

## Windows / PyCharm

1. Распакуйте `COS1_STAGE3_UPDATE.zip` **в отдельную папку**, например `C:\Users\ВАШЕ_ИМЯ\Downloads\COS1_STAGE3_UPDATE`. Не распаковывайте его поверх проекта.
2. Откройте **существующий** проект в PyCharm. В Terminal из корня проекта выполните (замените путь к распакованному обновлению):

```powershell
.\.venv\Scripts\python.exe "C:\Users\ВАШЕ_ИМЯ\Downloads\COS1_STAGE3_UPDATE\scripts\apply_stage3_update.py" --project . --apply
```

Скрипт сначала проверяет контрольные суммы входов/кода и всего обновления. Если новый файл уже существует с другим содержимым, он останавливается **до копирования**, выводит конкретный конфликт и ничего не перезаписывает. Не удаляйте такой файл автоматически: сохраните свои изменения и согласуйте объединение. Для предварительной проверки используйте ту же команду без `--apply`. Повторная установка идентичного обновления безопасна.

3. Если пакет уже установлен через editable installation, повторная установка не нужна. Если `import cos1` не работает именно в выбранном интерпретаторе, выполните в корне проекта:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

4. Откройте `notebooks/03_normalization.ipynb`, выберите существующий интерпретатор/ядро `.venv\Scripts\python.exe`, нажмите **Run All**. Все ячейки выполняются по порядку; результаты появятся в Notebook и новых каталогах этапа III.

Не меняйте Sources Root и не пересоздавайте `.venv`. Код не использует ручную подстановку `sys.path`. Рабочая папка может быть корнем проекта или `notebooks`: пути берутся из установленного `cos1`.

## Добавляемые файлы

- `configs/normalization_experiments.json` — группы, применимость и параметры дизайна эксперимента.
- `src/cos1/normalization.py` — шесть преобразований и сериализация параметров.
- `src/cos1/normalization_study.py` — загрузка/проверка входов, таблицы, графики и отчёт; отдельный модуль сохраняет Notebook читаемым и исключает копирование математики.
- `notebooks/03_normalization.ipynb` — исследовательский Notebook с выполненными ячейками.
- `tests/test_normalization.py` — автоматические проверки.
- `results/tables/normalization_metrics.csv` — 102 строки, включая явные NA.
- `results/tables/normalization_original_metrics.csv` — исходные характеристики всех 17 обработанных рядов до нормировки.
- `results/tables/normalization_checks.csv` — 89 проверенных преобразований.
- `results/tables/normalization_state_comparisons.csv` — описательные отношения характеристик.
- `results/manifests/normalization_manifest.json` — происхождение входов, группы и сохранённые параметры.
- `results/figures/normalization/` — 19 PNG: `CWRU`, `Paderborn`, `JNU`, `MFPT`, `HUSTbearing-Huazhong`, для каждого суффиксы `_time.png`, `_hist.png`, `_spectrum.png`; дополнительно `MFPT_relative_power.png` и три `MFPT_matched_{time,hist,spectrum}.png`.
- `reports/stage3/NORMALIZATION_REPORT.md` — научный отчёт, пересоздаваемый из вычислений.
- `reports/stage3/CHECKSUM_AUDIT.md`, `BASELINE_SHA256.json`, `VALIDATION.md` — исходное состояние и результаты проверки.
- `scripts/apply_stage3_update.py` — проверяемый установщик добавочных файлов.
- `scripts/check_normalization_notebook.py` — последовательная проверка ячеек в отдельном процессе IPython, когда внешний kernel недоступен; запись выводов только с `--write`.
- `UPDATE_STAGE3_INSTRUCTIONS.md` — эта инструкция.
- `STAGE3_STATE_SHA256.json` — новый снимок содержимого; не подтверждение старого релиза.

`STAGE3_UPDATE_MANIFEST.json` — служебный список файлов пакета и требований к базе; используется установщиком из распакованного обновления. Точный перечень переносимых файлов находится в его поле `payload`.

## Повторное выполнение и контрольные суммы

Run All обновляет только результаты этапа III. Его Notebook-выводы, версии ПО в манифесте и изображения при другой версии библиотек могут менять байтовые хэши. Это не означает изменения входных сигналов. `STAGE3_STATE_SHA256.json` описывает **доставленный проверенный снимок**, автоматически не переписывается. Исторический `RELEASE_SHA256.json` не изменён.

Новые массивы `.npy` не создаются: преобразования воспроизводятся из неизменённых `data/processed` и сохранённых параметров. Для отдельного применения используйте `Parameters.from_dict(...)` и `apply(new_signal, parameters)`; функция `apply` не оценивает параметры заново.

## Тесты

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests scripts/apply_stage3_update.py
```

Для возврата к этапу II достаточно не открывать новый Notebook; старые файлы не изменены. Полное удаление обновления возможно по списку `payload`, но удалять новые файлы с собственными последующими правками без резервной копии не следует.
