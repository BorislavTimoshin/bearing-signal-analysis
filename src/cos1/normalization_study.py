"""Stage III orchestration; reads stage-II arrays and writes only new results."""

import hashlib
import json
from pathlib import Path
import platform

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cos1.analysis import statistics
from cos1.data import ROOT
from cos1.normalization import (
    METHODS,
    apply,
    fit_group_z,
    fit_individual,
    fit_reference,
    mean_square_spectrum,
)
from cos1.spectrum import amplitude_spectrum

LABELS = {
    "none": "Без нормировки",
    "individual_z": "Индивидуальная z",
    "max_abs": "По max |x|",
    "rms": "По RMS",
    "group_z": "Групповая z",
    "reference_rms": "По исправному эталону",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_study(root=ROOT):
    root = Path(root)
    config = json.loads(
        (root / "configs/normalization_experiments.json").read_text(encoding="utf-8")
    )
    meta = pd.read_csv(
        root / "results/tables/resampling_manifest.csv", keep_default_na=False
    )
    arrays = {}
    if len(meta) != 17 or meta.id.duplicated().any():
        raise ValueError("Expected 17 unique stage-II records")
    for row in meta.to_dict("records"):
        path = root / row["output_path"]
        if digest(path) != row["output_sha256"]:
            raise ValueError(f"Processed input checksum mismatch: {row['id']}")
        x = np.load(path, allow_pickle=False)
        if x.ndim != 1 or not np.isfinite(x).all() or len(x) != row["M"]:
            raise ValueError("Invalid stage-II data")
        if row["fs_out"] != config["fs_hz"]:
            raise ValueError("Unexpected sample rate")
        x.flags.writeable = False
        arrays[row["id"]] = x
    return config, meta, arrays


def fit_common(config, meta, arrays):
    """Fit declared groups; channel/condition decisions are in the config."""
    fitted, membership = {}, {}
    for method in ("group_z", "reference_rms"):
        for group in config[method]:
            selected = {i: arrays[i] for i in group["fit_ids"]}
            if method == "reference_rms":
                states = meta.set_index("id").loc[group["fit_ids"], "fault_type"]
                if not states.eq("normal").all():
                    raise ValueError("Reference contains a nonhealthy record")
                params = fit_reference(selected)
            else:
                params = fit_group_z(selected)
            key = method + ":" + group["id"]
            fitted[key] = {"parameters": params, "definition": group}
            for i in group["apply_ids"]:
                if i not in arrays or (i, method) in membership:
                    raise ValueError("Unknown record or overlapping application groups")
                membership[i, method] = key
    if any((i, "group_z") not in membership for i in arrays):
        raise ValueError("Every record needs an explicitly defined group")
    return fitted, membership


def evaluate(config, meta, arrays, fitted, membership):
    """Calculate real arrays, statistics and independent affine/FFT checks."""
    transformed, params_used, records, checks = {}, {}, [], []
    source_rows = meta.set_index("id").to_dict("index")
    for record_id, x in arrays.items():
        row = source_rows[record_id]
        before = statistics(x)
        f, original_power = mean_square_spectrum(x, config["fs_hz"])
        _, original_amplitude = amplitude_spectrum(x, config["fs_hz"])
        for method in METHODS:
            common_key = membership.get((record_id, method))
            base = dict(
                id=record_id,
                dataset=row["dataset"],
                source_filename=row["source_filename"],
                channel=row["channel"],
                fault_type=row["fault_type"],
                recording_group_id=row["recording_group_id"],
                method=method,
                fs_hz=config["fs_hz"],
                n=len(x),
                duration_s=len(x) / config["fs_hz"],
                input_unit=row["signal_unit"] or "unconfirmed_recorded_unit",
                output_unit=(row["signal_unit"] or "unconfirmed_recorded_unit")
                if method == "none"
                else "1 (relative)",
                group_key=common_key or "individual",
                input_sha256=row["output_sha256"],
            )
            if method in ("group_z", "reference_rms") and common_key is None:
                records.append(
                    dict(
                        **base,
                        status="not_applicable",
                        reason=config["reference_not_applicable"].get(
                            row["dataset"], "No justified reference"
                        ),
                    )
                )
                continue
            params = (
                fitted[common_key]["parameters"]
                if common_key
                else fit_individual(x, method, record_id)
            )
            y = apply(x, params)
            transformed[record_id, method] = y
            params_used[record_id, method] = params
            after = statistics(y)
            _, power = mean_square_spectrum(y, config["fs_hz"])
            _, amplitude = amplitude_spectrum(y, config["fs_hz"])
            total = power.sum()
            centroid = float(np.sum(f * power) / total) if total else None
            scale, offset = params.scale, params.offset
            expected_mean = (before["mean"] - offset) / scale
            expected_std = before["std_centered"] / scale
            expected_rms = (
                np.hypot(before["std_centered"], before["mean"] - offset) / scale
            )
            expected_max = (
                max(abs(float(x.min()) - offset), abs(float(x.max()) - offset)) / scale
            )

            def close(a, b):
                return bool(np.isclose(a, b, rtol=1e-10, atol=1e-12))

            flags = dict(
                mean_ok=close(after["mean"], expected_mean),
                std_ok=close(after["std_centered"], expected_std),
                rms_ok=close(after["rms_raw"], expected_rms),
                max_ok=close(after["max_abs"], expected_max),
                excess_ok=close(after["excess"], before["excess"]),
                crest_centered_ok=close(
                    after["crest_centered"], before["crest_centered"]
                ),
                crest_raw_ok=offset != 0
                or close(after["crest_raw"], before["crest_raw"]),
                parseval_ok=close(total, after["std_centered"] ** 2),
            )
            amp_error = float(
                np.max(abs(amplitude - original_amplitude / scale))
                / max(float(np.max(original_amplitude / scale)), 1e-300)
            )
            shape_error = float(
                np.max(abs(power / total - original_power / original_power.sum()))
            )
            flags.update(
                amplitude_scale_ok=amp_error < 1e-10,
                spectral_shape_ok=shape_error < 1e-10,
            )
            checks.append(
                dict(
                    id=record_id,
                    method=method,
                    **flags,
                    amplitude_relative_error=amp_error,
                    normalized_power_max_error=shape_error,
                    passed=all(flags.values()),
                )
            )
            records.append(
                dict(
                    **base,
                    status="ok",
                    reason="",
                    offset=offset,
                    scale=scale,
                    **after,
                    centered_mean_square=float(total),
                    spectral_centroid_hz=centroid,
                )
            )
    return transformed, params_used, pd.DataFrame(records), pd.DataFrame(checks)


def compare_states(metrics):
    """Descriptive ratios, not independent statistical observations."""
    healthy = {
        "CWRU": "cwru_normal",
        "Paderborn": "pu_K001",
        "JNU": "jnu_normal",
        "MFPT": "mfpt_baseline_1",
    }
    result = []
    for dataset, normal in healthy.items():
        sub = metrics[metrics.dataset.eq(dataset)]
        for method in METHODS:
            h = sub[sub.id.eq(normal) & sub.method.eq(method)].iloc[0]
            for _, d in sub[~sub.id.eq(normal) & sub.method.eq(method)].iterrows():
                status = "ok" if h.status == d.status == "ok" else "not_applicable"
                warning = (
                    "different_native_bandwidth"
                    if dataset == "CWRU"
                    else "different_load_and_native_fs"
                    if d.id == "mfpt_InnerRaceFault_vload_1"
                    else "incomplete_load_and_units"
                    if dataset == "JNU"
                    else "unconfirmed_units"
                    if dataset == "Paderborn"
                    else "single_record_per_state"
                )
                row = dict(
                    dataset=dataset,
                    healthy_id=normal,
                    fault_id=d.id,
                    method=method,
                    status=status,
                    limitation=warning,
                    shared_parameters=bool(
                        h.group_key == d.group_key
                        and method in ("group_z", "reference_rms")
                    ),
                )
                for col in ("rms_raw", "std_centered", "max_abs"):
                    row[col + "_fault_over_healthy"] = (
                        float(d[col] / h[col]) if status == "ok" else None
                    )
                result.append(row)
    return pd.DataFrame(result)


def comparison_figure(meta, transformed, dataset, kind="time", fs=51200):
    """Each panel shares its axes across records; methods have different units."""
    rows = meta[meta.dataset.eq(dataset)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), layout="constrained")
    fig.suptitle(
        f"{dataset} — "
        + {
            "time": "осциллограммы: первые 40 мс",
            "hist": "плотности распределений: весь ряд",
            "spectrum": "амплитудные спектры: весь ряд",
        }[kind],
        fontsize=15,
    )
    colors = plt.get_cmap("tab10")
    for ax, method in zip(axes.flat, METHODS):
        available = [
            (row, transformed[row.id, method])
            for row in rows.itertuples()
            if (row.id, method) in transformed
        ]
        ax.set_title(
            LABELS[method]
            + (
                " (разные группы)"
                if method == "group_z"
                and (
                    dataset in ("CWRU", "HUSTbearing-Huazhong")
                    or (dataset == "MFPT" and len(rows) > 2)
                )
                else ""
            )
        )
        if not available:
            ax.text(
                0.5,
                0.5,
                "Неприменимо:\nнет обоснованного исправного эталона",
                ha="center",
                va="center",
                transform=ax.transAxes,
                wrap=True,
            )
            ax.set_axis_off()
            continue
        units = (
            (str(rows.iloc[0].signal_unit) or "ед. записи (не подтверждены)")
            if method == "none"
            else "отн. ед."
        )
        bins = np.linspace(
            min(y.min() for _, y in available), max(y.max() for _, y in available), 101
        )
        for j, (row, y) in enumerate(available):
            color = colors(list(rows.id).index(row.id))
            label = (
                row.id.replace("hustbearing_medium_ball_65", "ball65")
                .replace("mfpt_", "")
                .replace("cwru_", "")
                .replace("jnu_", "")
                .replace("pu_", "")
            )
            if dataset == "MFPT" and "InnerRace" in row.id:
                label += " (0 lb)"
            if kind == "time":
                count = min(len(y), int(0.04 * fs))
                ax.plot(
                    np.arange(count) / fs * 1000,
                    y[:count],
                    lw=0.8,
                    alpha=0.85,
                    label=label,
                    color=color,
                )
            elif kind == "hist":
                ax.hist(
                    y,
                    bins=bins,
                    density=True,
                    histtype="step",
                    lw=1.2,
                    label=label,
                    color=color,
                )
            else:
                f, a = amplitude_spectrum(y, fs)
                ax.plot(f / 1000, a, lw=0.7, alpha=0.8, label=label, color=color)
        ax.set_xlabel(
            "Время, мс"
            if kind == "time"
            else f"Амплитуда, {units}"
            if kind == "hist"
            else "Частота, кГц"
        )
        ax.set_ylabel(
            "Плотность, 1/(ед. амплитуды)" if kind == "hist" else f"Амплитуда, {units}"
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, loc="upper right")
        if kind == "spectrum":
            ax.set_xlim(0, fs / 2000)
            if dataset == "CWRU":
                ax.axvspan(6, fs / 2000, color="grey", alpha=0.15)
                ax.text(
                    0.98,
                    0.5,
                    "Исходные полосы\nразличаются",
                    ha="right",
                    transform=ax.transAxes,
                    fontsize=8,
                )
    return fig


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, metadata={"Software": "COS1 Stage III"})


def write_results(
    root,
    config,
    meta,
    arrays,
    fitted,
    membership,
    params_used,
    metrics,
    checks,
    comparisons,
):
    """Write only explicitly named stage-III destinations; no old checksums updated."""
    root = Path(root)
    tables = root / "results/tables"
    tables.mkdir(parents=True, exist_ok=True)
    for name, table in [
        ("normalization_metrics", metrics),
        ("normalization_original_metrics", metrics[metrics.method.eq("none")]),
        ("normalization_checks", checks),
        ("normalization_state_comparisons", comparisons),
    ]:
        table.to_csv(tables / (name + ".csv"), index=False, float_format="%.17g")
    groups = {
        key: {
            "parameters": value["parameters"].to_dict(),
            "definition": value["definition"],
        }
        for key, value in fitted.items()
    }
    manifest = dict(
        schema=1,
        scope="descriptive_only; not a training/test evaluation",
        fs_hz=config["fs_hz"],
        config_sha256=digest(root / "configs/normalization_experiments.json"),
        stage2_manifest_sha256=digest(root / "results/tables/resampling_manifest.csv"),
        software=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            pandas=pd.__version__,
            matplotlib=matplotlib.__version__,
        ),
        groups=groups,
        source_records=meta.to_dict("records"),
        transformations=[
            dict(
                id=i, method=m, group_key=membership.get((i, m)), parameters=p.to_dict()
            )
            for (i, m), p in params_used.items()
        ],
        unavailable=metrics[metrics.status.ne("ok")][
            ["id", "method", "reason"]
        ].to_dict("records"),
        outputs_saved_as_arrays=False,
        reuse="Apply saved parameters only to compatible channel, units, measurement conditions and preprocessing. Current fit is descriptive, not train-only.",
        hashes={
            name: digest(root / name)
            for name in ["src/cos1/normalization.py", "src/cos1/normalization_study.py"]
        },
    )
    path = root / "results/manifests/normalization_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return manifest


def report_text(metrics, checks, comparisons):
    """Numbers in the report are computed, never handwritten experimental values."""
    lines = [
        "# COS1. Этап III: описательное исследование нормировки",
        "",
        "Источник: COS1_STAGE2_COMPLETE_2026-11-14.zip. Использованы только готовые массивы этапа II при 51 200 Гц; ресемплинг не повторялся.",
        "",
        f"17 рядов × 6 методов = {len(metrics)} строк: применимо {metrics.status.eq('ok').sum()}, неприменимо {metrics.status.eq('not_applicable').sum()}. Теоретические проверки: {checks.passed.sum()}/{len(checks)}.",
        "",
        "## Определения и гарантированные свойства",
        "",
        r"$\mu=N^{-1}\sum x_i$, $\sigma^2=N^{-1}\sum(x_i-\mu)^2$, $R=\sqrt{N^{-1}\sum x_i^2}$; делитель СКО — N, без поправки Бесселя.",
        r"Пик-фактор $C=\max|x|/R$; дополнительно центрированный $C_c=\max|x-\mu|/\sigma$. Эксцесс $K=N^{-1}\sum((x_i-\mu)/\sigma)^4-3$, без поправки на смещение оценки.",
        "",
        r"Все методы имеют вид $y=(x-c)/s$, $s>0$. Тогда $\mu_y=(\mu-c)/s$, $\sigma_y=\sigma/s$, $R_y=\sqrt{\sigma^2+(\mu-c)^2}/s$ и $\max|y|=\max_i|x_i-c|/s$.",
        r"При c=0 сохраняется обычный пик-фактор; при сдвиге это не гарантируется. Центрированный пик-фактор и эксцесс инвариантны при ненулевом СКО. Для $y=ax+b$, $a\ne0$, стандартизованные отсчёты отличаются только знаком, исчезающим в четвёртой степени.",
        "",
        r"Для 6x: $\mu_{6x}=6\mu_x$, $\sigma_{6x}=6\sigma_x$, поэтому $(6x-6\mu_x)/(6\sigma_x)=(x-\mu_x)/\sigma_x$ при $\sigma_x>0$.",
        "",
        "Методы: none; individual_z (c=μ, s=σ); max_abs (c=0, s=max|x|); rms (c=0, s=R); group_z (c=μ группы, s=σ группы); reference_rms (c=0, s=медиана RMS исправных эталонов). Нулевой знаменатель вызывает ValueError, а не подменяется epsilon. Для постоянных сигналов эксцесс и центрированный пик-фактор не определены.",
        "",
        "## Группы и эталоны",
        "",
        "В общей z-нормировке каждая включённая запись получает одинаковый вес: μ_D=mean(μ_j), σ_D²=mean[σ_j²+(μ_j−μ_D)²]. Поэтому учитываются и внутрисигнальная вариация, и различия средних. Это не простое среднее СКО и не склейка отсчётов.",
        "CWRU: отдельные группы 48 кГц (исправная) и 12 кГц (три дефекта); MFPT: отдельные группы 270 lb и 0 lb. Paderborn: пара K001/KA01. JNU: четыре записи 600 об/мин, условно, при неполных метаданных. HUSTbearing: каждый полный канал отдельно; вложенный X получает параметры полного X, но не участвует повторно в их оценке. Одночленные группы иллюстрируют метод, но не дают межэкспериментальной оценки масштаба.",
        "Исправный RMS-эталон используется только для MFPT baseline_1→baseline_1/OuterRaceFault_1 и условно для Paderborn K001→K001/KA01 (тот же режим/канал, единицы неизвестны). В каждой группе пока одна исправная запись, поэтому медиана равна её RMS и не обеспечивает устойчивости по повторениям.",
        "Для CWRU (разные полосы), JNU (неподтверждённая нагрузка/калибровка), MFPT inner при 0 lb (нет подходящего контроля), HUSTbearing (нет исправного контроля) reference_rms отмечен как неприменимый. Все параметры и исходные IDs сохранены в normalization_manifest.json.",
        "",
        "## Измеренные отношения неисправный / исправный",
        "",
        "Числа ниже — отношения RMS доступных рядов, а не оценки точности классификации. Неприменимые методы обозначены NA.",
        "",
        "| Набор / неисправная запись | Без | z индивидуальная | max | RMS | z групповая | Исправный эталон |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (dataset, fault), block in comparisons.groupby(
        ["dataset", "fault_id"], sort=False
    ):
        vals = []
        for method in METHODS:
            row = block[block.method.eq(method)].iloc[0]
            vals.append(
                f"{row.rms_raw_fault_over_healthy:.6g}" if row.status == "ok" else "NA"
            )
        lines.append("| " + dataset + " / " + fault + " | " + " | ".join(vals) + " |")
    lines += [
        "",
        "Отношение после RMS-нормировки равно 1 по определению; после индивидуальной z-нормировки RMS также равен 1. Это устранение выбранного масштаба, а не обнаруженный успех диагностики. После max-нормировки RMS может отличаться: он равен обратному исходному пик-фактору. Общий исправный масштаб сохраняет исходные отношения RMS, СКО и максимумов. Групповая z сохраняет отношения СКО внутри одной группы, но общий сдвиг может изменить отношения RMS и максимумов.",
        "",
        "## Распределения и спектры",
        "",
        "При y=(x−c)/s распределение переносится и растягивается: f_y(v)=s f_x(sv+c). Стандартизованная форма и эксцесс сохраняются, хотя масштабы, положения и визуальные гистограммы меняются. Отсчёты коррелированы; гистограммы не дают независимых повторений.",
        "Амплитудный спектр рассчитан существующей функцией этапа II: из копии сигнала удаляется среднее, применяется периодическое окно Ханна, FFT делится на сумму окна; внутренние односторонние частоты удваиваются, DC и чётный Nyquist — нет. Нормировка окна корректна для изолированной гармоники точно на частотной сетке; высота любого пика не обязана быть несмещённой оценкой амплитуды.",
        "Для всех рассматриваемых преобразований центрированный спектр масштабируется как 1/s, относительная форма сохраняется. Для нецентрированной FFT сдвиг меняет DC. Окно используется только при отображении спектра, а не для нормировки временного ряда.",
        "Спектральное среднее квадрата отдельно считается без окна: P_k=w_k|FFT(x−μ)/N|². Его сумма равна σ² (Parseval). P_k имеет единицы амплитуды² на бин; плотность получается делением на Δf=fs/N. Квадрат удвоенного амплитудного спектра Ханна не подменяет эту величину. Это среднее квадрата ускорения/отсчётов, не механическая мощность в ваттах.",
        f"Максимальная относительная ошибка масштабирования амплитудного спектра: {checks.amplitude_relative_error.max():.3g}; максимальная абсолютная ошибка нормированной формы спектрального среднего квадрата: {checks.normalized_power_max_error.max():.3g}.",
        "",
        "## Ограничения и выводы",
        "",
        "1. Индивидуальные методы удаляют информацию об общем множителе амплитуды, но не делают любые два разных сигнала одинаковыми. z дополнительно удаляет постоянное смещение; max/RMS сохраняют относительное смещение.",
        "2. Общий положительный масштаб перспективен, когда нужна информация об относительных амплитудах и имеются сопоставимые измерения. Индивидуальная нормировка перспективна, если усиление датчика — помеха, но может удалить полезную тяжесть вибрации. Сейчас это гипотезы, а не установленное преимущество.",
        "3. MFPT outer и Paderborn — основные пары; JNU — условное сравнение. CWRU имеет различную исходную полосу: общий fs после ресемплинга не делает широкополосные RMS сопоставимыми. MFPT inner имеет другую нагрузку. Эти строки оставлены как иллюстрации, не контролируемый эксперимент.",
        "4. HUSTbearing содержит только дефект. X/Y/Z и вложенный фрагмент — не независимые подшипники. Физические единицы HUSTbearing/JNU/Paderborn не подтверждены. Даже одинаковые единицы g не гарантируют межстендовую сопоставимость.",
        "5. Нет достаточных независимых повторений на состояние, поэтому нет ROC-AUC, тестов значимости и выбора лучшего метода. Для продолжения нужны физические IDs, повторения, согласованные режимы/полосы/каналы, единицы и калибровка, исправный HUSTbearing, сопоставимые CWRU и MFPT inner.",
        "6. Общие параметры здесь оценены по описательной выборке, включая дефекты. В будущем fit выполняется только на обучающей части с группировкой по физическому подшипнику/эксперименту. Сохранённые параметры можно применить без переоценки через Parameters.from_dict и apply; применимость к новым условиям не доказана.",
        "",
        "Графики: results/figures/normalization/ (по три рисунка для каждого датасета: time, hist, spectrum). Все сравниваемые записи внутри панели имеют общую шкалу. Длительности — 1 с для фрагментов и 10,24 с для полных HUST; для осциллограмм показаны первые 40 мс, статистики и спектры рассчитаны по полным доступным рядам.",
        "",
        "## Воспроизводимость и происхождение",
        "",
        "Notebook 03_normalization.ipynb загружает реальные data/processed, проверяет их SHA-256 по текущему манифесту этапа II, последовательно оценивает параметры и вычисляет таблицы/графики. Метаданные условий перенесены из архива, новая проверка авторских источников не проводилась. Первичные URL и исходные хэши находятся в source_records манифеста этапа III.",
        "Исторический RELEASE_SHA256.json оставлен без изменений. Его несоответствие текущему архиву описано отдельно в CHECKSUM_AUDIT.md. Новый STAGE3_STATE_SHA256.json описывает проверенное состояние, не подтверждает исторический релиз.",
    ]
    baseline = metrics[metrics.method.eq("none")].set_index("id")
    mfpt_h, mfpt_d = (
        baseline.loc["mfpt_baseline_1"],
        baseline.loc["mfpt_OuterRaceFault_1"],
    )
    pu_h, pu_d = baseline.loc["pu_K001"], baseline.loc["pu_KA01"]
    hx, hfull = (
        baseline.loc["hustbearing_medium_ball_65"],
        baseline.loc["hustbearing_medium_ball_65_full_X"],
    )
    observations = [
        "",
        "## Наблюдения по реальным массивам",
        "",
        f"MFPT, сопоставимая пара: RMS исправного фрагмента {mfpt_h.rms_raw:.6g} g, наружного дефекта {mfpt_d.rms_raw:.6g} g (отношение {mfpt_d.rms_raw / mfpt_h.rms_raw:.6g}). В этой паре дефектный фрагмент имеет меньший RMS, но немного больший максимум ({mfpt_d.max_abs / mfpt_h.max_abs:.6g} от исправного) и больший пик-фактор ({mfpt_d.crest_raw:.6g} против {mfpt_h.crest_raw:.6g}). Поэтому утверждение «дефект всегда увеличивает RMS» не подтверждается даже этими двумя фрагментами. Различие может зависеть от динамики стенда и выбранного фрагмента; причинный механизм здесь не установлен.",
        f"Paderborn: RMS KA01/K001 = {pu_d.rms_raw / pu_h.rms_raw:.6g}; эксцесс {pu_d.excess:.6g} против {pu_h.excess:.6g}. Центр спектрального среднего квадрата {pu_d.spectral_centroid_hz:.6g} Гц против {pu_h.spectral_centroid_hz:.6g} Гц. Это измеренное различие распределения энергии по частотам в выбранных рядах, не оценка частоты дефекта. После индивидуальной z исчезает различие RMS, но сохраняются эксцесс и относительная форма спектра.",
        "JNU: у всех трёх дефектных фрагментов RMS больше, чем у исправного (точные отношения в таблице). Их большой эксцесс указывает на существенный вклад редких больших отклонений в четвёртый момент; он не доказывает конкретную причину импульсов. Длина фрагмента, неизвестные условия и отсутствие повторений ограничивают вывод.",
        f"HUSTbearing X: RMS фрагмента {hx.rms_raw:.6g}, полного ряда {hfull.rms_raw:.6g}; максимумы {hx.max_abs:.6g} и {hfull.max_abs:.6g} в неподтверждённых единицах записи. Более длинная запись включает больше событий: различие максимумов нельзя объявлять изменением состояния подшипника. Вложенные данные не считаются независимыми.",
        "CWRU: численно амплитудные характеристики дефектных фрагментов выше исправного, но исходные полосы различаются. Нормировка сама по себе не устраняет это смешение факторов; отдельная согласованная полоса потребовала бы другого, пока не выполняемого эксперимента.",
        "",
        "Дополнительный спектральный центр определён как sum(f_k P_k)/sum(P_k) для центрированного прямоугольного спектра. Он характеризует положение распределения среднего квадрата по частотам и инвариантен к рассмотренным нормировкам при ненулевой дисперсии. Вместе с проверкой Parseval он служит контролем формы, а не диагностическим порогом.",
        "",
        "Для наглядности основная пара MFPT вынесена дополнительно на отдельные три рисунка MFPT_matched_{time,hist,spectrum}.png: сильные пики записи inner при другой нагрузке не скрывают сравнение baseline/outer. Всего сохранено 19 рисунков.",
    ]
    lines.extend(observations)
    return "\n".join(lines) + "\n"
