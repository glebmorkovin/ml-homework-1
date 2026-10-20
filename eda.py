"""Разведочный анализ полного Bank Marketing Dataset без обучения модели."""

from pathlib import Path
import hashlib
import os

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font="DejaVu Sans", font_scale=0.95)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 180)
pd.set_option("display.float_format", "{:.3f}".format)
figures_dir = BASE_DIR / "figures"
figures_dir.mkdir(exist_ok=True)


def save_figure(name):
    plt.tight_layout()
    plt.savefig(figures_dir / name, dpi=160, bbox_inches="tight")
    plt.close()


df = pd.read_csv(BASE_DIR / "data" / "bank-full.csv", sep=";")
assert df.shape == (45211, 17), "Ожидается полный bank-full.csv"
assert set(df["y"].unique()) == {"yes", "no"}
numeric_cols = df.select_dtypes(include="number").columns.tolist()
categorical_cols = df.select_dtypes(include="object").columns.drop("y").tolist()
target = df["y"].eq("yes").astype(int)

print("РАЗМЕР И СТРУКТУРА:", df.shape)
print("SHA256:", hashlib.sha256((BASE_DIR / "data/bank-full.csv").read_bytes()).hexdigest())
print(pd.DataFrame({"type": df.dtypes, "unique": df.nunique(), "missing": df.isna().sum()}))
print("\nПервые строки:\n", df.head().to_string(index=False))
print("\nЧИСЛОВЫЕ СТАТИСТИКИ:\n", df[numeric_cols].describe().T)
print("\nКАТЕГОРИАЛЬНЫЕ СТАТИСТИКИ:\n", df.select_dtypes(include="object").describe().T)

unknown_counts = df[categorical_cols].eq("unknown").sum()
unknown_counts = unknown_counts[unknown_counts > 0]
print("\nUNKNOWN:\n", pd.DataFrame({"count": unknown_counts, "percent": unknown_counts / len(df) * 100}))
print("Строк хотя бы с одним unknown:", df[categorical_cols].eq("unknown").any(axis=1).sum())
print("Пустых строк:", df[categorical_cols + ["y"]].eq("").sum().sum())
print("Полных дубликатов:", df.duplicated().sum())
print("Повторов входных признаков:", df.drop(columns="y").duplicated().sum())

target_counts = df["y"].value_counts().reindex(["no", "yes"])
print("\nЦЕЛЕВАЯ ПЕРЕМЕННАЯ:\n", pd.DataFrame({"count": target_counts, "percent": target_counts / len(df) * 100}))
print("Отношение no/yes:", target_counts["no"] / target_counts["yes"])

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(target_counts.index, target_counts, color=["#587a9e", "#c06b3e"])
for i, count in enumerate(target_counts):
    ax.text(i, count + 600, f"{count:,}\n{count / len(df):.2%}", ha="center")
ax.set(title="Оформление срочного депозита", xlabel="Целевая переменная y", ylabel="Число наблюдений", ylim=(0, 45500))
save_figure("01_target.png")

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for ax, col in zip(axes.flat, numeric_cols):
    values = df.loc[df[col] != -1, col] if col == "pdays" else df[col]
    bins = np.arange(values.min(), values.max() + 2) - 0.5 if col in ["age", "day", "campaign"] else 45
    sns.histplot(values, bins=bins, ax=ax, color="#587a9e")
    ax.set_yscale("log")
    ax.set(title=col + (" (только ранее контактировавшие)" if col == "pdays" else ""), xlabel=col, ylabel="Число наблюдений (лог. шкала)")
# Преобразование используется только для показа баланса, исходный CSV не меняется.
balance_log = np.sign(df["balance"]) * np.log1p(df["balance"].abs())
sns.histplot(balance_log, bins=60, ax=axes.flat[7], color="#587a9e")
axes.flat[7].set(title="Баланс: подробности около нуля", xlabel="sign(balance) × ln(1 + |balance|)", ylabel="Число наблюдений")
axes.flat[8].axis("off")
save_figure("02_numeric_distributions.png")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, col in zip(axes.flat, numeric_cols):
    values = df.loc[df[col] != -1, col] if col == "pdays" else df[col]
    sns.boxplot(y=values, ax=ax, color="#8ba7bd", fliersize=2)
    if col in ["balance", "duration", "campaign", "previous"]:
        ax.set_yscale("symlog", linthresh=1)
    if col != "balance":
        ax.set_ylim(bottom=0)
    ax.set(title=col + (" (без −1)" if col == "pdays" else ""), ylabel=col)
axes.flat[-1].axis("off")
save_figure("03_numeric_boxplots.png")

outliers = []
for col in numeric_cols:
    values = df.loc[df[col] != -1, col] if col == "pdays" else df[col]
    q1, q3 = values.quantile([0.25, 0.75])
    lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    count = ((values < lower) | (values > upper)).sum()
    outliers.append([col, len(values), lower, upper, count, count / len(values) * 100])
print("\nВЫБРОСЫ ПО ПРАВИЛУ 1.5 IQR:\n", pd.DataFrame(outliers, columns=["feature", "n", "lower", "upper", "count", "percent"]).to_string(index=False))
print("\nДОПОЛНИТЕЛЬНЫЕ ПРОВЕРКИ:")
checks = {
    "balance < 0": (df["balance"] < 0).sum(),
    "balance == 0": df["balance"].eq(0).sum(),
    "duration == 0": df["duration"].eq(0).sum(),
    "age < 18": (df["age"] < 18).sum(),
    "day outside 1..31": (~df["day"].between(1, 31)).sum(),
    "campaign < 1": (df["campaign"] < 1).sum(),
    "previous < 0": (df["previous"] < 0).sum(),
    "pdays < -1": (df["pdays"] < -1).sum(),
    "pdays == -1 and previous > 0": (df["pdays"].eq(-1) & df["previous"].gt(0)).sum(),
    "pdays >= 0 and previous == 0": (df["pdays"].ge(0) & df["previous"].eq(0)).sum(),
}
print(pd.Series(checks).to_string())
print("\nМаксимальные previous:\n", df.nlargest(5, "previous")[["previous", "campaign", "pdays", "poutcome", "y"]].to_string(index=False))

fig, axes = plt.subplots(3, 3, figsize=(16, 15))
for ax, col in zip(axes.flat, categorical_cols):
    counts = df[col].value_counts().sort_values()
    ax.barh(counts.index, counts, color="#587a9e")
    ax.set(title=col, xlabel="Число наблюдений")
save_figure("04_categorical_counts.png")

fig, axes = plt.subplots(3, 3, figsize=(17, 16))
for ax, col in zip(axes.flat, categorical_cols):
    table = df.assign(positive=target).groupby(col)["positive"].agg(["size", "sum", "mean"]).sort_values("mean")
    print(f"\nДОЛЯ YES ПО {col}:\n", table.to_string())
    labels = [f"{name} (n={n:,})" for name, n in zip(table.index, table["size"])]
    ax.barh(labels, table["mean"], color="#587a9e")
    ax.axvline(target.mean(), color="#c06b3e", linestyle="--")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set(title=col, xlabel="Доля y=yes внутри категории", xlim=(0, 0.75))
fig.suptitle("Пунктир: общая доля yes. Наблюдаемые связи не доказывают причинность", fontsize=13)
save_figure("05_categorical_target.png")

print("\nЧИСЛОВЫЕ ПРИЗНАКИ ПО КЛАССАМ (медианы):\n", df.groupby("y")[numeric_cols].median())
groups = pd.DataFrame({
    "Возраст": pd.cut(df["age"], [0, 30, 40, 50, 60, 100], labels=["≤30", "31–40", "41–50", "51–60", ">60"]),
    "Баланс (квинтили)": pd.qcut(df["balance"], 5),
    "Контакты в текущей кампании": pd.cut(df["campaign"], [0, 1, 2, 3, 5, 10, np.inf], labels=["1", "2", "3", "4–5", "6–10", ">10"]),
    "Давность предыдущего контакта": pd.cut(df["pdays"], [-2, -1, 30, 180, 365, np.inf], labels=["не было", "0–30", "31–180", "181–365", ">365"]),
    "Контакты до кампании": pd.cut(df["previous"], [-1, 0, 1, 3, 10, np.inf], labels=["0", "1", "2–3", "4–10", ">10"]),
    "Длительность (только диагностика)": pd.cut(df["duration"], [-1, 60, 180, 300, 600, np.inf], labels=["0–60", "61–180", "181–300", "301–600", ">600"]),
})
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for ax, col in zip(axes.flat, groups.columns):
    table = target.groupby(groups[col], observed=False).agg(["size", "sum", "mean"])
    print(f"\nДОЛЯ YES: {col}:\n", table.to_string())
    labels = [f"{name}\nn={n:,}" for name, n in zip(table.index, table["size"])]
    ax.bar(labels, table["mean"], color="#c06b3e" if "Длительность" in col else "#587a9e")
    ax.axhline(target.mean(), color="black", linestyle="--", linewidth=1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(title=col, ylabel="Доля y=yes", ylim=(0, 0.6))
    ax.tick_params(axis="x", labelsize=8, rotation=25)
save_figure("06_numeric_target.png")

history = df.assign(previously_contacted=df["pdays"].ne(-1))
print("\nИСТОРИЯ КОНТАКТОВ:\n", pd.crosstab(history["previously_contacted"], history["y"]))
print("\nPDAYS БЕЗ -1:\n", df.loc[df["pdays"] != -1, "pdays"].describe())
print("\nИСТОРИЯ И POUTCOME:\n", pd.crosstab(history["previously_contacted"], history["poutcome"]))
print("\nКОНТАКТ И МЕСЯЦ:\n", pd.crosstab(df["month"], df["contact"]))

months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
month_table = df.assign(positive=target).groupby("month")["positive"].agg(["size", "mean"]).reindex(months)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].bar(months, month_table["size"], color="#587a9e")
axes[0].set(title="Наблюдения по месяцам (все годы вместе)", xlabel="Месяц последнего контакта", ylabel="Число наблюдений")
axes[1].plot(months, month_table["mean"], marker="o", color="#587a9e")
axes[1].axhline(target.mean(), color="#c06b3e", linestyle="--")
axes[1].yaxis.set_major_formatter(PercentFormatter(1))
axes[1].set(title="Доля yes по месяцам", xlabel="Месяц последнего контакта", ylabel="Доля y=yes")
save_figure("07_month.png")

correlation_data = df[numeric_cols].copy()
correlation_data["pdays"] = correlation_data["pdays"].replace(-1, np.nan)
correlation_data["y_yes"] = target
correlations = correlation_data.corr(method="spearman")
print("\nКОРРЕЛЯЦИИ СПИРМЕНА:\n", correlations)
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(correlations, mask=np.triu(np.ones_like(correlations, dtype=bool), k=1), annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title("Корреляции Спирмена; pdays без −1\nduration показан для диагностики утечки")
save_figure("08_correlations.png")

# Порядок строк по документации хронологический, но полной даты в CSV нет.
blocks = df.assign(block=np.arange(len(df)) // 5000 + 1, positive=target).groupby("block")["positive"].agg(["size", "mean"])
print("\nБЛОКИ ПО ПОРЯДКУ СТРОК:\n", blocks)
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar([f"{i}\nn={n}" for i, n in zip(blocks.index, blocks["size"])], blocks["mean"], color="#587a9e")
ax.axhline(target.mean(), color="#c06b3e", linestyle="--")
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set(title="Доля yes в последовательных блоках по 5 000 строк", xlabel="Блок в порядке исходного файла (последний неполный)", ylabel="Доля y=yes")
save_figure("09_row_order.png")
print(f"\nАнализ завершён. Сохранено 9 графиков в {figures_dir}")
