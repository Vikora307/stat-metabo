# ══════════════════════════════════════════════════════════════
#  METABOLOMICS STATISTICAL PIPELINE (LC-HRMS)
#  — Parallel multi-CSV processing —
#
#  Automatically detects the experimental design and runs:
#    • 2 factors (F1 × F2)  →  Two-way ANOVA
#    • 1 factor, >2 levels  →  One-way ANOVA
#    • 1 factor, 2 levels   →  Volcano (t-test + log2FC)
#
#  Outputs per CSV: PCA, PLS-DA, VIP, heatmaps, barplots, Venn
# ══════════════════════════════════════════════════════════════

# ==============================================================
# LIBRARIES
# ==============================================================
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for parallel)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import multiprocessing
from sklearn.decomposition import PCA
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind, chi2
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from matplotlib_venn import venn3_unweighted
from sklearn.impute import SimpleImputer
import warnings
import traceback
import time

warnings.filterwarnings("ignore")


# ╔═════════════════════════════════════════════════════════════╗
# ║               USER CONFIGURATION — EDIT HERE               ║
# ╠═════════════════════════════════════════════════════════════╣
# ║  Modify the parameters below to suit your experiment.      ║
# ║  Everything else runs automatically.                       ║
# ╚═════════════════════════════════════════════════════════════╝

# --- Input / Output ---
INPUT_FOLDER = r"D:\PROJET\20260216-MetSP-FILFRUI-Stress-Gel-Hydrique\ESIpos\STATSbis"

# --- Statistical thresholds ---
ALPHA = 0.05                  # significance level
USE_FDR = False              # True = filter on FDR-adjusted p-values
                              # False = filter on raw p-values (FDR still computed & saved)

# --- CSV layout (row indices, 0-based) ---
ROW_SAMPLE_NAMES = 0          # row containing sample names
ROW_FACTOR_1     = 1          # row containing Factor 1 levels
ROW_FACTOR_2     = 2          # row containing Factor 2 levels
ROW_FACTOR_3     = 3          # row containing Factor 3 levels (optional, kept in metadata)
ROW_DATA_START   = 4          # first row of abundance data
CSV_SEPARATOR    = ";"        # delimiter used in the CSV files

# --- Plot settings ---
DPI              = 300        # resolution of saved figures
PCA_FIGSIZE      = (7, 7)     # PCA / PLS-DA figure size
HEATMAP_WIDTH    = 12         # heatmap figure width
BARPLOT_FIGSIZE  = (13, 4)    # barplot figure size
POINT_SIZE       = 60         # scatter point size (PCA / PLS-DA)
ELLIPSE_CI       = 0.95       # confidence interval for ellipses (0.0–1.0)

# --- Feature selection ---
VIP_TOP_N        = 30         # number of top VIP features to plot
HEATMAP_TOP_N    = 50        # number of top features for "TopN" heatmaps

# --- Parallelism ---
MAX_WORKERS      = None       # None = use all available CPUs
                              # set to 1 for sequential (easier debugging)

# --- Transformation ---
LOG_BASE         = 10         # log transformation base (10 or 2; use None to skip log)
AUTOSCALE        = True       # True = mean-center + divide by SD per molecule


# ╔═════════════════════════════════════════════════════════════╗
# ║           END OF USER CONFIGURATION                        ║
# ╚═════════════════════════════════════════════════════════════╝


# ==============================================================
# FULL PIPELINE (runs for ONE CSV file)
# ==============================================================
def run_pipeline(fichier_csv):
    """
    Complete metabolomics statistical pipeline for a single CSV file.
    All outputs are saved in a dedicated subfolder named after the CSV.
    """
    csv_basename = os.path.splitext(os.path.basename(fichier_csv))[0]
    print(f"\n{'=' * 60}")
    print(f"  STARTING : {csv_basename}")
    print(f"  File     : {fichier_csv}")
    print(f"  FDR mode : {'ON (filtering on adjusted p)' if USE_FDR else 'OFF (filtering on raw p)'}")
    print(f"{'=' * 60}")
    t0 = time.time()

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    def progress(step, total, msg):
        print(f"  [{csv_basename}] [{int(step / total * 100):3d}%] {msg}")

    def p_col(base):
        """Return the column name to filter on, depending on USE_FDR."""
        return f"{base}_FDR" if USE_FDR else base

    def p_label():
        """Label for axis / file names."""
        return "p_FDR" if USE_FDR else "p"

    # ----------------------------------------------------------
    # Output directories
    # ----------------------------------------------------------
    out = os.path.join(os.path.dirname(fichier_csv), f"PIPELINE_STATS_{csv_basename}")
    bar_dir = os.path.join(out, "BARPLOTS")
    os.makedirs(out, exist_ok=True)
    os.makedirs(bar_dir, exist_ok=True)

    # ==============================================================
    # 1. READ CSV
    # ==============================================================
    progress(1, 14, "Reading CSV...")
    df = pd.read_csv(fichier_csv, sep=CSV_SEPARATOR, header=None)

    facteur1 = df.iloc[ROW_FACTOR_1, 1:]
    facteur2 = df.iloc[ROW_FACTOR_2, 1:]
    facteur3 = df.iloc[ROW_FACTOR_3, 1:]
    nom_f1 = df.iloc[ROW_FACTOR_1, 0]
    nom_f2 = df.iloc[ROW_FACTOR_2, 0]

    sample_names = df.iloc[ROW_SAMPLE_NAMES, 1:].astype(str).tolist()
    molecules = df.iloc[ROW_DATA_START:, 0].values

    data = df.iloc[ROW_DATA_START:, 1:].astype(float)
    data.index = molecules
    data.columns = sample_names

    metadata = pd.DataFrame({
        "col": data.columns,
        "F1": facteur1.values,
        "F2": facteur2.values,
        "F3": facteur3.values
    })
    metadata["group"] = metadata["F1"].astype(str) + "_" + metadata["F2"].astype(str)
    n_f1 = metadata["F1"].nunique()
    n_f2 = metadata["F2"].nunique()

    progress(1, 14, f"  → {len(molecules)} molecules × {len(sample_names)} samples")
    progress(1, 14, f"  → Factor 1 ({nom_f1}): {n_f1} levels | Factor 2 ({nom_f2}): {n_f2} levels")

    # ==============================================================
    # 2. DATA TRANSFORMATION
    # ==============================================================
    progress(2, 14, "Transforming data...")

    data_transformed = data.copy()

    if LOG_BASE is not None:
        min_val = data_transformed[data_transformed > 0].min().min()
        data_transformed = data_transformed.replace(0, min_val)
        if LOG_BASE == 10:
            data_transformed = np.log10(data_transformed)
        elif LOG_BASE == 2:
            data_transformed = np.log2(data_transformed)
        else:
            data_transformed = np.log(data_transformed) / np.log(LOG_BASE)
        progress(2, 14, f"  → log{LOG_BASE} applied (min replacement = {min_val:.2e})")

    if AUTOSCALE:
        row_mean = data_transformed.mean(axis=1)
        row_std = data_transformed.std(axis=1)
        # avoid division by zero for constant rows
        row_std = row_std.replace(0, np.nan)
        data_scaled = data_transformed.sub(row_mean, axis=0).div(row_std, axis=0)
        progress(2, 14, "  → Autoscaled (mean-centered, unit variance)")
    else:
        data_scaled = data_transformed
        progress(2, 14, "  → Autoscaling skipped")

    # ==============================================================
    # HELPER FUNCTIONS
    # ==============================================================

    def mean_by_group(mat):
        g = metadata.set_index("col").loc[mat.columns, "group"]
        return mat.groupby(g, axis=1).mean()

    # ----------------------------------------------------------
    # PLS-DA SCORE PLOT
    # ----------------------------------------------------------
    def plsda_plot(mat, labels, name, show_labels=False, use_f2=True, n_components=2):
        X = mat.T.values
        if np.isnan(X).any():
            X = SimpleImputer(strategy="mean").fit_transform(X)

        df_sc = pd.DataFrame(index=mat.columns)
        df_sc["label"] = labels.values if hasattr(labels, "values") else np.array(labels)

        parts = df_sc["label"].astype(str).str.split("_", n=1, expand=True)
        df_sc["F1"] = parts[0]
        df_sc["F2"] = parts[1] if parts.shape[1] > 1 else "NA"

        f1_levels = df_sc["F1"].unique()
        f2_levels = df_sc["F2"].unique()

        df_sc["group"] = df_sc["F1"] if not use_f2 else df_sc["F1"] + "_" + df_sc["F2"]

        classes = df_sc["group"].values
        uniq = np.unique(classes)
        if len(uniq) < 2:
            return

        ncomp = int(min(n_components, 2, X.shape[0] - 1, X.shape[1]))
        if ncomp < 2:
            return

        le = LabelEncoder()
        y_int = le.fit_transform(classes)
        Y = np.eye(len(le.classes_))[y_int]

        pls = PLSRegression(n_components=ncomp)
        pls.fit(X, Y)

        T = pls.x_scores_[:, :2]
        df_sc["LV1"] = T[:, 0]
        df_sc["LV2"] = T[:, 1]

        palette = sns.color_palette("tab20", len(f1_levels))
        color_map = dict(zip(f1_levels, palette))

        show_f2_flag = use_f2 and len(f2_levels) > 1
        marker_list = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">"]
        if show_f2_flag:
            marker_map = dict(zip(f2_levels, marker_list[:len(f2_levels)]))
        else:
            marker_map = {f2_levels[0]: "o"} if len(f2_levels) else {"NA": "o"}

        fig, ax = plt.subplots(figsize=PCA_FIGSIZE)

        for grp, g in df_sc.groupby("group"):
            x, y = g["LV1"].values, g["LV2"].values
            f1 = g["F1"].iloc[0]
            mk = marker_map.get(g["F2"].iloc[0], "o") if show_f2_flag else "o"

            ax.scatter(x, y, s=POINT_SIZE, c=[color_map[f1]], marker=mk,
                       edgecolors="black", linewidths=0.8, alpha=0.9)

            if len(x) > 2:
                cov = np.cov(x, y)
                vals, vecs = np.linalg.eig(cov)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                w, h = 2 * np.sqrt(vals * chi2.ppf(ELLIPSE_CI, 2))
                ell = Ellipse(xy=(np.mean(x), np.mean(y)), width=w, height=h,
                              angle=theta, edgecolor=color_map[f1], lw=1.5,
                              ls="--", facecolor="none", alpha=0.7)
                ax.add_patch(ell)

            if show_labels:
                for xi, yi, lab in zip(x, y, g["label"].values):
                    ax.text(xi, yi, lab, fontsize=7, alpha=0.8)

        ax.set_xlabel("LV1")
        ax.set_ylabel("LV2")
        ax.set_title(f"PLS-DA: {name}")

        handles_f1 = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[f],
                   markeredgecolor="black", markersize=9, label=str(f))
            for f in f1_levels
        ]
        leg1 = ax.legend(handles=handles_f1, loc="upper left",
                         bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=10)
        ax.add_artist(leg1)

        if show_f2_flag:
            handles_f2 = [
                Line2D([0], [0], marker=marker_map[f], color="black",
                       linestyle="", markersize=9, label=str(f))
                for f in f2_levels
            ]
            ax.legend(handles=handles_f2, loc="upper left",
                      bbox_to_anchor=(1.02, 0.55), frameon=False, fontsize=10)

        plt.subplots_adjust(right=0.75)
        plt.savefig(os.path.join(out, f"PLSDA_{name}.png"), dpi=DPI)
        plt.close()

    # ----------------------------------------------------------
    # PLS-DA VIP COMPUTATION
    # ----------------------------------------------------------
    def plsda_vip_top_features(mat, labels, use_f2=True, n_components=2, top_n=20):
        X = mat.T.values
        feature_names = mat.index.to_numpy()

        all_nan = np.isnan(X).all(axis=0)
        if all_nan.any():
            X = X[:, ~all_nan]
            feature_names = feature_names[~all_nan]

        if np.isnan(X).any():
            X = SimpleImputer(strategy="mean").fit_transform(X)

        if X.shape[1] == 0:
            raise ValueError("VIP: 0 features remaining after NaN removal.")

        lab = pd.Series(labels.values if hasattr(labels, "values") else labels).astype(str)
        parts = lab.str.split("_", n=1, expand=True)
        F1 = parts[0]
        F2 = parts[1] if parts.shape[1] > 1 else "NA"
        groups = (F1 + "_" + F2).values if use_f2 else F1.values

        uniq = np.unique(groups)
        if len(uniq) < 2:
            raise ValueError("VIP: only 1 class detected — need at least 2.")

        le = LabelEncoder()
        y_int = le.fit_transform(groups)
        Y = np.eye(len(le.classes_))[y_int]

        ncomp = int(min(n_components, X.shape[0] - 1, X.shape[1]))
        if ncomp < 1:
            raise ValueError("VIP: not enough samples/features for PLS.")

        pls = PLSRegression(n_components=ncomp)
        pls.fit(X, Y)

        T, W, Q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
        p = X.shape[1]
        SSY = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)
        SSY_total = np.sum(SSY)

        if SSY_total == 0:
            raise ValueError("VIP: explained variance is zero.")

        vip = np.sqrt(p * np.sum(SSY * (W ** 2), axis=1) / SSY_total)

        vip_df = (pd.DataFrame({"molecule": feature_names, "VIP": vip})
                    .sort_values("VIP", ascending=False))
        return vip_df, vip_df["molecule"].head(top_n).tolist()

    # ----------------------------------------------------------
    # VIP BAR PLOT
    # ----------------------------------------------------------
    def vip_plot(vip_df, name, top_n=VIP_TOP_N):
        if vip_df.empty:
            return
        top = vip_df.sort_values("VIP", ascending=False).head(top_n).iloc[::-1]
        plt.figure(figsize=(8, max(5, 0.35 * len(top))))
        plt.barh(top["molecule"], top["VIP"])
        plt.axvline(1.0, color="black", linestyle="--", linewidth=1)
        plt.xlabel("VIP score (PLS-DA)")
        plt.ylabel("Molecule")
        plt.title(f"Top {top_n} VIP — PLS-DA ({name})")
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"PLSDA_{name}_VIP_Top{top_n}.png"), dpi=DPI)
        plt.close()

    # ----------------------------------------------------------
    # PCA SCORE PLOT
    # ----------------------------------------------------------
    def pca_plot(mat, labels, name, show_labels=False, use_f2=True):
        X = mat.T.values
        if np.isnan(X).any():
            X = SimpleImputer(strategy="mean").fit_transform(X)

        pca = PCA(n_components=2)
        scores = pca.fit_transform(X)

        df_sc = pd.DataFrame(scores, columns=["PC1", "PC2"])
        df_sc["label"] = labels.values if hasattr(labels, "values") else np.array(labels)

        parts = df_sc["label"].astype(str).str.split("_", n=1, expand=True)
        df_sc["F1"] = parts[0]
        df_sc["F2"] = parts[1] if parts.shape[1] > 1 else "NA"

        f1_levels = df_sc["F1"].unique()
        f2_levels = df_sc["F2"].unique()

        df_sc["group"] = df_sc["F1"] if not use_f2 else df_sc["F1"] + "_" + df_sc["F2"]

        palette = sns.color_palette("tab20", len(f1_levels))
        color_map = dict(zip(f1_levels, palette))

        show_f2_flag = use_f2 and len(f2_levels) > 1
        marker_list = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">"]
        if show_f2_flag:
            marker_map = dict(zip(f2_levels, marker_list[:len(f2_levels)]))
        else:
            marker_map = {f2_levels[0]: "o"}

        fig, ax = plt.subplots(figsize=PCA_FIGSIZE)

        for grp, g in df_sc.groupby("group"):
            x, y = g["PC1"].values, g["PC2"].values
            f1 = g["F1"].iloc[0]
            mk = marker_map.get(g["F2"].iloc[0], "o") if show_f2_flag else "o"

            ax.scatter(x, y, s=POINT_SIZE, c=[color_map[f1]], marker=mk,
                       edgecolors="black", linewidths=0.8, alpha=0.9)

            if len(x) > 2:
                cov = np.cov(x, y)
                vals, vecs = np.linalg.eig(cov)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                w, h = 2 * np.sqrt(vals * chi2.ppf(ELLIPSE_CI, 2))
                ell = Ellipse(xy=(np.mean(x), np.mean(y)), width=w, height=h,
                              angle=theta, edgecolor=color_map[f1], lw=1.5,
                              ls="--", facecolor="none", alpha=0.7)
                ax.add_patch(ell)

            if show_labels:
                for xi, yi, lab in zip(x, y, g["label"].values):
                    ax.text(xi, yi, lab, fontsize=7, alpha=0.8)

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
        ax.set_title(name)

        handles_f1 = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[f],
                   markeredgecolor="black", markersize=9, label=str(f))
            for f in f1_levels
        ]
        leg1 = ax.legend(handles=handles_f1, loc="upper left",
                         bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=10)
        ax.add_artist(leg1)

        if show_f2_flag:
            handles_f2 = [
                Line2D([0], [0], marker=marker_map[f], color="black",
                       linestyle="", markersize=9, label=str(f))
                for f in f2_levels
            ]
            ax.legend(handles=handles_f2, loc="upper left",
                      bbox_to_anchor=(1.02, 0.55), frameon=False, fontsize=10)

        plt.subplots_adjust(right=0.75)
        plt.savefig(os.path.join(out, f"PCA_{name}.png"), dpi=DPI)
        plt.close()

    # ----------------------------------------------------------
    # CLUSTERED HEATMAP
    # ----------------------------------------------------------
    def heatmap_plot(mat, name, replicat=True):
        if mat.empty:
            print(f"  [{csv_basename}] ⚠️ Heatmap '{name}' is empty — skipped")
            return
        if mat.shape[0] < 2 or mat.shape[1] < 2:
            print(f"  [{csv_basename}] ⚠️ Heatmap '{name}' too small {mat.shape} — skipped")
            return

        if replicat:
            meta = metadata.set_index("col").loc[mat.columns].reset_index()
            new_cols = meta["F1"].astype(str) + "_" + meta["F2"].astype(str)
            mat = mat.copy()
            mat.columns = new_cols

            f1_labels = meta["F1"].astype(str).values
            f2_labels = meta["F2"].astype(str).values
            f1_levels = pd.unique(f1_labels)
            f2_levels = pd.unique(f2_labels)
            show_f2 = meta["F2"].astype(str).nunique() > 1

            f1_palette = sns.color_palette("tab20", len(f1_levels))
            f1_map = dict(zip(f1_levels, f1_palette))

            col_colors = pd.DataFrame({"F1": [f1_map[x] for x in f1_labels]}, index=mat.columns)

            if show_f2:
                f2_palette = sns.color_palette("tab10", len(f2_levels))
                f2_map = dict(zip(f2_levels, f2_palette))
                col_colors["F2"] = [f2_map[x] for x in f2_labels]

            g = sns.clustermap(mat, cmap="vlag", center=0, col_colors=col_colors,
                               xticklabels=True, yticklabels=True,
                               figsize=(HEATMAP_WIDTH, max(6, len(mat) / 4)),
                               dendrogram_ratio=(0.15, 0.12), colors_ratio=0.022)

            g.ax_heatmap.xaxis.set_ticks_position("bottom")
            g.ax_heatmap.xaxis.set_label_position("bottom")
            plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, ha="center")
            g.ax_heatmap.yaxis.tick_right()
            g.ax_heatmap.tick_params(axis="y", labelright=True, labelleft=False)

            g.fig.subplots_adjust(right=0.58)
            legend_x = 0.74
            g.cax.set_position([legend_x, 0.78, 0.022, 0.18])
            g.cax.set_title("value", fontsize=10, pad=6)

            extra_artists = []

            if show_f2:
                handles_f2 = [
                    Line2D([0], [0], marker="s", linestyle="", markersize=9,
                           markerfacecolor=f2_map[l], markeredgecolor="none", label=str(l))
                    for l in f2_levels
                ]
                leg_f2 = g.fig.legend(handles_f2, [str(l) for l in f2_levels], title="F2",
                                      loc="upper left", bbox_to_anchor=(legend_x, 0.68),
                                      frameon=False, borderaxespad=0, labelspacing=0.25,
                                      handletextpad=0.4)
                g.fig.add_artist(leg_f2)
                extra_artists.append(leg_f2)

            handles_f1 = [
                Line2D([0], [0], marker="s", linestyle="", markersize=9,
                       markerfacecolor=f1_map[l], markeredgecolor="none", label=str(l))
                for l in f1_levels
            ]
            y_f1 = 0.50 if show_f2 else 0.68
            leg_f1 = g.fig.legend(handles_f1, [str(l) for l in f1_levels], title="F1",
                                  loc="upper left", bbox_to_anchor=(legend_x, y_f1),
                                  frameon=False, borderaxespad=0, labelspacing=0.25,
                                  handletextpad=0.4)
            extra_artists.append(leg_f1)

            plt.savefig(os.path.join(out, f"Heatmap_{name}.png"), dpi=DPI,
                        bbox_inches="tight", bbox_extra_artists=extra_artists)
            plt.close()

        else:
            g = sns.clustermap(mat, cmap="vlag", center=0, xticklabels=True,
                               yticklabels=True,
                               figsize=(HEATMAP_WIDTH, max(6, len(mat) / 4)),
                               dendrogram_ratio=(0.15, 0.12))
            g.ax_heatmap.xaxis.set_ticks_position("bottom")
            g.ax_heatmap.xaxis.set_label_position("bottom")
            plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, ha="center")
            g.ax_heatmap.yaxis.tick_right()
            g.ax_heatmap.tick_params(axis="y", labelright=True, labelleft=False)
            g.fig.subplots_adjust(right=0.58)
            g.cax.set_position([0.74, 0.78, 0.022, 0.18])
            g.cax.set_title("value", fontsize=10, pad=6)
            plt.savefig(os.path.join(out, f"Heatmap_{name}.png"), dpi=DPI, bbox_inches="tight")
            plt.close()

    # ----------------------------------------------------------
    # BARPLOT PER MOLECULE
    # ----------------------------------------------------------
    def barplot_molecule(mol, subdir):
        def auto_legend_style(n):
            if n <= 6:   return 11, 1
            if n <= 10:  return 10, 2
            if n <= 16:  return 9, 2
            if n <= 24:  return 8, 3
            if n <= 32:  return 7, 4
            return 6, 4

        fig, ax = plt.subplots(figsize=BARPLOT_FIGSIZE)
        pos = 0

        f1_levels = list(metadata["F1"].unique())
        palette = sns.color_palette("tab20", n_colors=len(f1_levels))
        colors = dict(zip(f1_levels, palette))

        f2_levels = list(metadata["F2"].unique())
        hatch_list = ["", "//", "xx", "\\\\", "..", "++", "--", "oo", "**"]
        hatches_map = dict(zip(f2_levels, (hatch_list * 30)[:len(f2_levels)]))

        for (f1, f2), g in metadata.groupby(["F1", "F2"]):
            v = data.loc[mol, g["col"]].dropna()
            if v.empty:
                continue
            m, s = v.mean(), v.std()
            ax.bar(pos, m, color=colors[f1], hatch=hatches_map[f2], edgecolor="black")
            ax.errorbar(pos, m, yerr=[[0], [s]], fmt="none", ecolor="black", capsize=4)
            pos += 1

        f1_patches = [Line2D([0], [0], color=colors[f], lw=10) for f in f1_levels]
        f2_patches = [
            plt.Rectangle((0, 0), 1, 1, facecolor="white",
                           hatch=hatches_map[f], edgecolor="black")
            for f in f2_levels
        ]

        fs_f1, ncol_f1 = auto_legend_style(len(f1_levels))
        fs_f2, ncol_f2 = auto_legend_style(len(f2_levels))

        fig.subplots_adjust(right=0.70)
        leg1 = ax.legend(f1_patches, f1_levels, bbox_to_anchor=(1.02, 1.0),
                         loc="upper left", frameon=False, fontsize=fs_f1,
                         ncol=ncol_f1, labelspacing=0.25, handletextpad=0.6,
                         columnspacing=0.8)
        ax.add_artist(leg1)

        nrows_f1 = int(np.ceil(len(f1_levels) / ncol_f1))
        y_f2 = max(0.10, 1.00 - (0.07 + nrows_f1 * 0.065))
        leg2 = ax.legend(f2_patches, f2_levels, bbox_to_anchor=(1.02, y_f2),
                         loc="upper left", frameon=False, fontsize=fs_f2,
                         ncol=ncol_f2, labelspacing=0.35, handletextpad=0.6,
                         columnspacing=0.8)

        ax.set_ylabel("Relative abundance")
        ax.set_xticks([])
        ax.set_title(mol)

        os.makedirs(subdir, exist_ok=True)
        fig.savefig(os.path.join(subdir, f"{mol}.png"), dpi=DPI,
                    bbox_inches="tight", bbox_extra_artists=[leg1, leg2])
        plt.close(fig)

    # ==============================================================
    # 3. PCA (always generated)
    # ==============================================================
    progress(5, 14, "Generating PCA plots...")
    pca_plot(data_scaled, metadata["group"], "Replicates_no_labels", show_labels=False)
    pca_plot(data_scaled, metadata["group"], "Replicates_labels", show_labels=True)

    mean_mat = mean_by_group(data_scaled)
    pca_plot(mean_mat, mean_mat.columns, "Mean_no_labels", show_labels=False)
    pca_plot(mean_mat, mean_mat.columns, "Mean_labels", show_labels=True)
    progress(6, 14, "✅ PCA plots done")

    # ==============================================================
    # 4A. TWO-WAY ANOVA (n_f1 > 1 AND n_f2 > 1)
    # ==============================================================
    if n_f1 > 1 and n_f2 > 1:
        progress(7, 14, "Running two-way ANOVA...")

        res = []
        for mol in molecules:
            df_aov = metadata.copy()
            df_aov["value"] = data_scaled.loc[mol].reindex(df_aov["col"]).to_numpy()
            model = ols("value ~ C(F1) + C(F2) + C(F1):C(F2)", df_aov).fit()
            aov = sm.stats.anova_lm(model, typ=2)
            res.append({
                "molecule": mol,
                "F1_F": aov.loc["C(F1)", "F"],
                "F2_F": aov.loc["C(F2)", "F"],
                "INT_F": aov.loc["C(F1):C(F2)", "F"],
                "pF1": aov.loc["C(F1)", "PR(>F)"],
                "pF2": aov.loc["C(F2)", "PR(>F)"],
                "pINT": aov.loc["C(F1):C(F2)", "PR(>F)"]
            })

        stats = pd.DataFrame(res)

        # FDR correction (always computed)
        for c in ["pF1", "pF2", "pINT"]:
            stats[c + "_FDR"] = multipletests(stats[c], method="fdr_bh")[1]

        # Significance comments (respects USE_FDR toggle)
        for base, comment_col in [("pF1", "Comment_pF1"),
                                   ("pF2", "Comment_pF2"),
                                   ("pINT", "Comment_pINT")]:
            col = p_col(base)
            stats[comment_col] = np.where(stats[col] < ALPHA, "significant", "not significant")

        stats.to_csv(os.path.join(out, "ANOVA2_results.tsv"), sep="\t", index=False)

        # Significant sets
        sigF1 = set(stats.loc[stats[p_col("pF1")] < ALPHA, "molecule"])
        sigF2 = set(stats.loc[stats[p_col("pF2")] < ALPHA, "molecule"])
        sigINT = set(stats.loc[stats[p_col("pINT")] < ALPHA, "molecule"])

        progress(7, 14, f"  → F1: {len(sigF1)} sig | F2: {len(sigF2)} sig | Interaction: {len(sigINT)} sig")

        # PCA & PLS-DA
        labels = metadata["F1"].astype(str) + "_" + metadata["F2"].astype(str)
        pca_plot(data_scaled, labels, "ANOVA2", use_f2=True)
        plsda_plot(data_scaled, labels, "ANOVA2", use_f2=True)

        # VIP
        vip_df, _ = plsda_vip_top_features(data_scaled, labels, use_f2=True, top_n=2000)
        vip_df.to_csv(os.path.join(out, "PLSDA_ANOVA2_VIP.tsv"), sep="\t", index=False)

        sig_any = set(stats.loc[
            (stats[p_col("pF1")] < ALPHA) |
            (stats[p_col("pF2")] < ALPHA) |
            (stats[p_col("pINT")] < ALPHA),
            "molecule"
        ])

        vip_sig = vip_df[vip_df["molecule"].isin(sig_any)].copy()
        vip_sig.to_csv(os.path.join(out, "PLSDA_ANOVA2_VIP_sigOnly.tsv"), sep="\t", index=False)

        top20_sig = vip_sig.sort_values("VIP", ascending=False).head(VIP_TOP_N)
        top20_sig.to_csv(os.path.join(out, f"PLSDA_ANOVA2_VIP_top{VIP_TOP_N}_sigOnly.tsv"),
                         sep="\t", index=False)

        vip_plot(top20_sig, f"ANOVA2_top{VIP_TOP_N}_sigOnly", top_n=VIP_TOP_N)
        vip_plot(vip_df, "ANOVA2", top_n=VIP_TOP_N)

        # Heatmaps
        fdr_tag = "_FDR" if USE_FDR else ""
        for name_anova, sig_set, pcol, fcol in [
            ("F1", sigF1, p_col("pF1"), "F1_F"),
            ("F2", sigF2, p_col("pF2"), "F2_F"),
            ("INT", sigINT, p_col("pINT"), "INT_F")
        ]:
            if sig_set:
                sig_list = list(sig_set)
                heatmap_plot(data_scaled.loc[sig_list], f"ANOVA2_{name_anova}_Replicates", True)
                heatmap_plot(mean_by_group(data_scaled).loc[sig_list],
                             f"ANOVA2_{name_anova}_Mean", False)

            top_n_sig = (
                stats.loc[stats[pcol] < ALPHA]
                     .nlargest(HEATMAP_TOP_N, fcol)["molecule"]
                     .tolist()
            )
            if top_n_sig:
                heatmap_plot(data_scaled.loc[top_n_sig],
                             f"Top{HEATMAP_TOP_N}_ANOVA2_{name_anova}{fdr_tag}_sig_Replicates", True)
                heatmap_plot(mean_by_group(data_scaled).loc[top_n_sig],
                             f"Top{HEATMAP_TOP_N}_ANOVA2_{name_anova}{fdr_tag}_sig_Mean", False)
            else:
                print(f"  [{csv_basename}] ⚠️ ANOVA2 {name_anova}: no Top{HEATMAP_TOP_N} "
                      f"with {pcol} < {ALPHA}")

        # Barplots
        progress(10, 14, "Generating ANOVA2 barplots...")
        anova2_dir = os.path.join(bar_dir, "ANOVA2")
        subdirs = {
            "F1": os.path.join(anova2_dir, "F1"),
            "F2": os.path.join(anova2_dir, "F2"),
            "INT": os.path.join(anova2_dir, "Interaction"),
        }
        for d in subdirs.values():
            os.makedirs(d, exist_ok=True)
        for mol in sigF1:
            barplot_molecule(mol, subdirs["F1"])
        for mol in sigF2:
            barplot_molecule(mol, subdirs["F2"])
        for mol in sigINT:
            barplot_molecule(mol, subdirs["INT"])

        # Venn diagram
        progress(12, 14, "Generating Venn diagram...")
        plt.figure(figsize=(8, 6))
        v = venn3_unweighted([sigF1, sigF2, sigINT],
                             set_labels=(nom_f1, nom_f2, "Interaction"))
        for idx, color in zip(
            ["100", "010", "001", "110", "101", "011", "111"],
            ["#FF9999", "#99FF99", "#9999FF", "#FFCC99", "#FF99FF", "#99FFFF", "#CCCCCC"]
        ):
            patch = v.get_patch_by_id(idx)
            if patch:
                patch.set_facecolor(color)
                patch.set_alpha(0.5)

        plt.title(f"Venn — Two-way ANOVA ({p_label()} < {ALPHA})")
        summary_venn = {nom_f1: len(sigF1), nom_f2: len(sigF2), "Interaction": len(sigINT)}
        colors_venn = {nom_f1: "#FF9999", nom_f2: "#99FF99", "Interaction": "#9999FF"}
        ax = plt.gca()
        for i, (label, count) in enumerate(summary_venn.items()):
            ax.text(1.05, 0.7 - i * 0.1, f"{label}: {count}",
                    transform=ax.transAxes, fontsize=11, fontweight="bold",
                    color=colors_venn[label], va="center")
        plt.subplots_adjust(right=0.75)
        plt.savefig(os.path.join(out, "Venn_ANOVA2.png"), dpi=DPI)
        plt.close()

    # ==============================================================
    # 4B. ONE-WAY ANOVA (n_f1 > 2, n_f2 == 1)
    # ==============================================================
    elif n_f1 > 2 and n_f2 == 1:
        progress(7, 14, "Running one-way ANOVA...")

        res = []
        for mol in molecules:
            df_aov = metadata.copy()
            df_aov["value"] = pd.to_numeric(data_scaled.loc[mol].values, errors="coerce")
            if df_aov["value"].notna().sum() < 2:
                res.append({"molecule": mol, "F": np.nan, "p": np.nan})
                continue
            try:
                model = ols("value ~ C(F1)", df_aov).fit()
                aov = sm.stats.anova_lm(model, typ=2)
                Fv = aov.loc["C(F1)", "F"]
                pv = aov.loc["C(F1)", "PR(>F)"]
            except Exception:
                Fv, pv = np.nan, np.nan
            res.append({"molecule": mol, "F": Fv, "p": pv})

        stats = pd.DataFrame(res)
        stats["p"] = pd.to_numeric(stats["p"], errors="coerce")
        stats["F"] = pd.to_numeric(stats["F"], errors="coerce")

        mask = stats["p"].notna()
        stats["p_FDR"] = np.nan
        if mask.sum() > 0:
            stats.loc[mask, "p_FDR"] = multipletests(stats.loc[mask, "p"], method="fdr_bh")[1]

        pc = p_col("p")
        stats["Comment_p"] = np.where(stats[pc] < ALPHA, "significant", "not significant")
        stats.to_csv(os.path.join(out, "ANOVA1_results.tsv"), sep="\t", index=False)

        # F vs -log10(p) plot
        df_plot = stats.dropna(subset=["F", pc]).copy()
        df_plot["status"] = np.where(df_plot[pc] < ALPHA, "SIG", "NS")
        summary_a1 = df_plot["status"].value_counts().reindex(["SIG", "NS"], fill_value=0)

        progress(7, 14, f"  → Significant: {summary_a1['SIG']} | Not significant: {summary_a1['NS']}")

        plt.figure(figsize=(7, 6))
        plt.scatter(df_plot["F"], -np.log10(df_plot[pc].clip(lower=1e-300)),
                    c=df_plot["status"].map({"SIG": "red", "NS": "lightgrey"}), s=30)
        plt.axhline(-np.log10(ALPHA), ls="--", c="black")
        plt.xlabel("F ratio")
        plt.ylabel(f"-log10({p_label()})")
        plt.title(f"One-way ANOVA: F ratio vs -log10({p_label()})")
        recap = f"Significant: {summary_a1['SIG']}\nNot significant: {summary_a1['NS']}"
        plt.text(0.98, 0.98, recap, transform=plt.gca().transAxes, ha="right", va="top",
                 fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        plt.tight_layout()
        fdr_tag = "_FDR" if USE_FDR else ""
        plt.savefig(os.path.join(out, f"ANOVA1_F_vs_{p_label()}.png"), dpi=DPI)
        plt.close()

        sig = stats.loc[stats[pc] < ALPHA, "molecule"]

        labels = metadata["F1"].astype(str)
        pca_plot(data_scaled, labels, "ANOVA1", use_f2=False)
        plsda_plot(data_scaled, labels, "ANOVA1", use_f2=False)

        # VIP
        vip_df, _ = plsda_vip_top_features(data_scaled, labels, use_f2=False, top_n=2000)
        vip_df.to_csv(os.path.join(out, "PLSDA_ANOVA1_VIP.tsv"), sep="\t", index=False)

        sig_set = set(sig)
        vip_sig = vip_df[vip_df["molecule"].isin(sig_set)].copy()
        vip_sig.to_csv(os.path.join(out, "PLSDA_ANOVA1_VIP_sigOnly.tsv"), sep="\t", index=False)

        top20_sig = vip_sig.sort_values("VIP", ascending=False).head(VIP_TOP_N)
        top20_sig.to_csv(os.path.join(out, f"PLSDA_ANOVA1_VIP_top{VIP_TOP_N}_sigOnly.tsv"),
                         sep="\t", index=False)

        if len(top20_sig) > 0:
            vip_plot(top20_sig, f"ANOVA1_top{VIP_TOP_N}_sigOnly", top_n=VIP_TOP_N)
        else:
            print(f"  [{csv_basename}] ⚠️ No significant molecules — VIP top{VIP_TOP_N} skipped")

        vip_plot(vip_df, "ANOVA1", top_n=VIP_TOP_N)

        # Heatmaps
        if len(sig) > 0:
            heatmap_plot(data_scaled.loc[sig], "ANOVA1_Replicates", True)
            heatmap_plot(mean_by_group(data_scaled).loc[sig], "ANOVA1_Mean", False)

        top_n_sig = (
            stats.loc[stats[pc] < ALPHA]
                 .dropna(subset=["F"])
                 .nlargest(HEATMAP_TOP_N, "F")["molecule"]
                 .tolist()
        )
        if top_n_sig:
            heatmap_plot(data_scaled.loc[top_n_sig],
                         f"Top{HEATMAP_TOP_N}_ANOVA1_F{fdr_tag}_sig_Replicates", True)
            heatmap_plot(mean_by_group(data_scaled).loc[top_n_sig],
                         f"Top{HEATMAP_TOP_N}_ANOVA1_F{fdr_tag}_sig_Mean", False)
        else:
            print(f"  [{csv_basename}] ⚠️ No Top{HEATMAP_TOP_N} with {p_label()} < {ALPHA}")

        # Barplots
        progress(10, 14, "Generating ANOVA1 barplots...")
        for mol in sig:
            barplot_molecule(mol, os.path.join(bar_dir, "ANOVA1"))

    # ==============================================================
    # 4C. VOLCANO (n_f1 == 2, n_f2 == 1)
    # ==============================================================
    elif n_f1 == 2 and n_f2 == 1:
        progress(7, 14, "Running Volcano (t-test + log2FC)...")

        groups = metadata.groupby("F1")["col"].apply(list)
        g1, g2 = groups.iloc[0], groups.iloc[1]

        res = []
        for mol in molecules:
            v1 = data.loc[mol, g1]
            v2 = data.loc[mol, g2]
            fc = (v2.mean() + 1e-9) / (v1.mean() + 1e-9)
            t, p_val = ttest_ind(v1, v2, nan_policy="omit")
            res.append({"molecule": mol, "log2FC": np.log2(fc), "p": p_val})

        stats = pd.DataFrame(res)
        stats["p_FDR"] = multipletests(stats["p"], method="fdr_bh")[1]

        pc = p_col("p")

        stats["Regulation"] = "Not significant"
        stats.loc[(stats[pc] < ALPHA) & (stats["log2FC"] > 0), "Regulation"] = "Up"
        stats.loc[(stats[pc] < ALPHA) & (stats["log2FC"] < 0), "Regulation"] = "Down"
        stats.to_csv(os.path.join(out, "Volcano_results.tsv"), sep="\t", index=False)

        stats["status"] = "NS"
        stats.loc[(stats[pc] < ALPHA) & (stats["log2FC"] > 0), "status"] = "UP"
        stats.loc[(stats[pc] < ALPHA) & (stats["log2FC"] < 0), "status"] = "DOWN"

        summary_v = stats["status"].value_counts().reindex(["UP", "DOWN", "NS"], fill_value=0)
        progress(7, 14, f"  → UP: {summary_v['UP']} | DOWN: {summary_v['DOWN']} | NS: {summary_v['NS']}")

        # Volcano plot
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(stats["log2FC"], -np.log10(stats[pc].clip(lower=1e-300)),
                   c=stats["status"].map({"UP": "red", "DOWN": "blue", "NS": "lightgrey"}), s=30)
        ax.axhline(-np.log10(ALPHA), ls="--", c="black")
        ax.axvline(0, ls="--", c="black")
        ax.set_xlabel("log2(Fold Change)")
        ax.set_ylabel(f"-log10({p_label()})")
        ax.set_title(f"Volcano plot ({p_label()} < {ALPHA})")

        for i, (label, color) in enumerate([("UP", "red"), ("DOWN", "blue"), ("NS", "grey")]):
            ax.text(1.02, 0.8 - i * 0.08, f"{label}: {summary_v[label]}",
                    transform=ax.transAxes, fontsize=11, fontweight="bold",
                    color=color, va="center")

        plt.subplots_adjust(right=0.78)
        plt.savefig(os.path.join(out, "Volcano_UP_DOWN.png"), dpi=DPI)
        plt.close()

        sig = stats.loc[stats[pc] < ALPHA, "molecule"]

        labels = metadata["F1"].astype(str)
        pca_plot(data_scaled, labels, "VOLCANO", use_f2=False)
        plsda_plot(data_scaled, labels, "VOLCANO", use_f2=False)

        # Heatmaps
        fdr_tag = "_FDR" if USE_FDR else ""
        if len(sig) > 0:
            heatmap_plot(data_scaled.loc[sig], "Volcano_Replicates", True)
            heatmap_plot(mean_by_group(data_scaled).loc[sig], "Volcano_Mean", False)

        top_n_sig = (
            stats.loc[stats[pc] < ALPHA]
                 .assign(absFC=lambda d: d["log2FC"].abs())
                 .nlargest(HEATMAP_TOP_N, "absFC")["molecule"]
                 .tolist()
        )
        if top_n_sig:
            heatmap_plot(data_scaled.loc[top_n_sig],
                         f"Top{HEATMAP_TOP_N}_Volcano_absFC{fdr_tag}_sig_Replicates", True)
            heatmap_plot(mean_by_group(data_scaled).loc[top_n_sig],
                         f"Top{HEATMAP_TOP_N}_Volcano_absFC{fdr_tag}_sig_Mean", False)
        else:
            print(f"  [{csv_basename}] ⚠️ No Top{HEATMAP_TOP_N} with {p_label()} < {ALPHA}")

        # Barplots
        progress(10, 14, "Generating Volcano barplots...")
        volcano_dir = os.path.join(bar_dir, "VOLCANO")
        up_dir = os.path.join(volcano_dir, "UP")
        down_dir = os.path.join(volcano_dir, "DOWN")
        os.makedirs(up_dir, exist_ok=True)
        os.makedirs(down_dir, exist_ok=True)

        for mol in stats.loc[stats["status"] == "UP", "molecule"]:
            barplot_molecule(mol, up_dir)
        for mol in stats.loc[stats["status"] == "DOWN", "molecule"]:
            barplot_molecule(mol, down_dir)

    # ==============================================================
    # DONE
    # ==============================================================
    elapsed = time.time() - t0
    progress(14, 14, f"PIPELINE COMPLETE 🎉  ({elapsed:.1f}s)")
    return csv_basename, True


# ==============================================================
# SAFE WRAPPER (catches exceptions per file)
# ==============================================================
def run_pipeline_safe(fichier_csv):
    """Wrapper so one failing file does not kill the whole pool."""
    try:
        return run_pipeline(fichier_csv)
    except Exception:
        basename = os.path.splitext(os.path.basename(fichier_csv))[0]
        print(f"\n❌ ERROR for {basename}:\n{traceback.format_exc()}")
        return basename, False


# ==============================================================
# MAIN — DISCOVER CSV FILES & RUN IN PARALLEL
# ==============================================================
if __name__ == "__main__":

    csv_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.csv")))

    print("=" * 60)
    print(f"  INPUT FOLDER   : {INPUT_FOLDER}")
    print(f"  CSV FILES FOUND: {len(csv_files)}")
    print(f"  ALPHA          : {ALPHA}")
    print(f"  USE FDR        : {USE_FDR}")
    print(f"  LOG BASE       : {LOG_BASE}")
    print(f"  AUTOSCALE      : {AUTOSCALE}")
    print(f"  VIP TOP-N      : {VIP_TOP_N}")
    print(f"  HEATMAP TOP-N  : {HEATMAP_TOP_N}")
    print("=" * 60)

    if len(csv_files) == 0:
        print("⚠️  No .csv files found in the input folder. Nothing to do.")
    else:
        for i, f in enumerate(csv_files, 1):
            print(f"  {i}. {os.path.basename(f)}")
        print("=" * 60)

        n_workers = min(
            len(csv_files),
            MAX_WORKERS if MAX_WORKERS is not None else max(1, multiprocessing.cpu_count())
        )
        print(f"  Launching {n_workers} parallel worker(s)...\n")

        with multiprocessing.Pool(processes=n_workers) as pool:
            results = pool.map(run_pipeline_safe, csv_files)

        # Summary
        print("\n" + "=" * 60)
        print("  SUMMARY")
        print("=" * 60)
        for name, success in results:
            status = "✅ OK" if success else "❌ FAILED"
            print(f"  {status}  →  {name}")

        n_ok = sum(1 for _, s in results if s)
        n_fail = sum(1 for _, s in results if not s)
        print(f"\n  Total: {n_ok} succeeded, {n_fail} failed out of {len(csv_files)} files.")
        print("=" * 60)