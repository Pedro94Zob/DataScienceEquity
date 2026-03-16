import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter, MinuteLocator

# =============================================================================
# 1. CHARGEMENT ET NETTOYAGE
# =============================================================================

df = pd.read_excel("Complete_Data_Venues.xlsx", engine="openpyxl")

# Les vrais headers sont à la ligne 3 du fichier Excel
df.columns = df.iloc[3]
df.columns = [col.replace('\n', '') if isinstance(col, str) else col for col in df.columns]

# Renommage des colonnes dupliquées (chacune apparaît 2 fois : Fartouch puis Mid)
columns = df.columns.tolist()
doublons = {
    'P&L Bps': ('P&L Bps vs Fartouch', 'P&L Bps vs Mid'),
    'P&L per Share €': ('P&L per Share € vs Fartouch', 'P&L per Share € vs Mid'),
    'P&L €': ('P&L € vs Fartouch', 'P&L € vs Mid'),
}
for col_name, (nom1, nom2) in doublons.items():
    indices = [i for i, col in enumerate(columns) if col == col_name]
    if len(indices) >= 2:
        columns[indices[0]] = nom1
        columns[indices[1]] = nom2
df.columns = columns

# On supprime les lignes de header (0 à 3)
df = df.iloc[4:].reset_index(drop=True)

# =============================================================================
# 2. FILTRE : on ne garde que les ordres (Care OU Market) exécutés en ALGO
# =============================================================================

df = df[
    ((df['PM Instruction'] == 'Care') | (df['PM Instruction'] == 'Market'))
    & (df['Exec Mode'] == 'ALGO')
]
print(f"Fills après filtre : {len(df)} | Order Id uniques : {df['Order Id'].nunique()}")

# =============================================================================
# 3. CONVERSIONS DE TYPES
# =============================================================================

colonnes_numeriques = [
    'Exec Qty', 'Exec Price', 'Order Qty', 'Gross Amount €',
    '% ADV', 'P&L Bps vs Fartouch', 'P&L Bps vs Mid',
    'P&L per Share € vs Fartouch', 'P&L per Share € vs Mid',
    'P&L € vs Fartouch', 'P&L € vs Mid',
    '% Perf vs Spread', 'Spread Capture', 'Spread Bps',
    'Far Touch €', 'Mid Spread €', 'Spread €'
]
for col in colonnes_numeriques:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

for col in ['Exec Timestamp (UTC)', 'Placement Timestamp', 'Pick Up Timestamp']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Poids de pondération : valeur absolue du montant brut en EUR
# Intègre quantité × prix × taux de change — reflète le vrai poids économique de chaque fill
df['Poids_EUR'] = df['Gross Amount €'].abs()

# =============================================================================
# 4. PHASE DE MARCHÉ (au niveau fill)
# Le Spread Capture n'a de sens économique que pendant le trading continu
# (quand il y a un vrai spread bid-ask). On classe chaque fill pour filtrer.
# =============================================================================

def classer_market_phase(row):
    """Détermine la phase de marché d'un fill à partir de son heure et de son type."""
    label = str(row.get('Liquidity Indicator Label', ''))
    if 'Auction' in label or 'auction' in label:
        return 'Auction'
    ts = row['Exec Timestamp (UTC)']
    if pd.isna(ts):
        return 'Unknown'
    heure = ts.hour + ts.minute / 60
    if 7.5 <= heure < 8.0:
        return 'Pre_Market'
    elif 8.0 <= heure < 17.0:
        return 'Continuous'
    elif 17.0 <= heure < 17.5:
        return 'Post_Market'
    else:
        return 'Off_Hours'

df['Market_Phase'] = df.apply(classer_market_phase, axis=1)
print(f"\nRépartition Market_Phase :\n{df['Market_Phase'].value_counts()}")

# =============================================================================
# 5. ENRICHISSEMENT FILL-LEVEL
# On garde chaque fill mais on ajoute : rang dans l'ordre, contribution au PnL,
# flag good/bad, et on force Spread Capture = NaN hors Continuous.
# =============================================================================

print("\nEnrichissement fill-level...")

# Tri chronologique par ordre puis par timestamp
df = df.sort_values(['Order Id', 'Exec Timestamp (UTC)']).reset_index(drop=True)

# Rang du fill dans l'ordre (1er fill, 2e fill, etc.)
df['Fill_Rank'] = df.groupby('Order Id').cumcount() + 1

# Nombre total de fills par ordre (pour contexte)
df['Nb_Fills_Ordre'] = df.groupby('Order Id')['Fill_Rank'].transform('max')

# Spread Capture = NaN si pas en Continuous (pas de spread bid-ask réel)
df['Spread_Capture_Cont'] = df['Spread Capture'].where(df['Market_Phase'] == 'Continuous', np.nan)

# Contribution de chaque fill au PnL total du trade
# Normalisé par Σ|PnL €| pour être borné entre -1 et +1
# Un fill à -0.30 = ce fill a dégradé le PnL du trade de 30%
def calc_contribution(group):
    pnl = group['P&L € vs Mid']
    total_abs = pnl.abs().sum()
    if total_abs > 0:
        return pnl / total_abs
    return pd.Series(np.nan, index=group.index)

df['PnL_Contribution_Mid'] = df.groupby('Order Id', group_keys=False).apply(
    lambda g: calc_contribution(g)
).reset_index(drop=True)

# Flag : ce fill a-t-il amélioré (+) ou dégradé (-) la TCA ?
# Good = exécuté mieux que le mid-spread (PnL > 0)
# Bad = exécuté moins bien que le mid-spread (PnL < 0)
# Neutral = exactement au mid (PnL = 0)
df['Fill_Quality'] = np.where(
    df['P&L Bps vs Mid'] > 0, 'Good',
    np.where(df['P&L Bps vs Mid'] < 0, 'Bad', 'Neutral')
)

print(f"  Fill Quality : {df['Fill_Quality'].value_counts().to_dict()}")

# =============================================================================
# 6. FONCTIONS UTILITAIRES
# =============================================================================

def wavg(group, col, poids='Poids_EUR'):
    """Moyenne pondérée par Gross Amount € (valeur absolue).
    Reflète le coût réel par euro investi, pas juste par action."""
    valid = group[[col, poids]].dropna()
    if valid[poids].sum() > 0:
        return (valid[col] * valid[poids]).sum() / valid[poids].sum()
    return np.nan

def build_segments(g, group_col):
    """Regroupe les fills consécutifs ayant la même valeur de group_col en segments.
    Retourne une liste de dicts avec toutes les infos nécessaires pour timeline + plot."""
    g_sorted = g.sort_values('Exec Timestamp (UTC)')

    segments = []
    current_val = None
    current_start = None
    current_end = None
    current_gross = 0
    current_aggr_qty = 0
    current_passive_qty = 0
    current_total_qty = 0
    current_good = 0
    current_bad = 0
    current_neutral = 0
    current_n_fills = 0
    current_pnl_sum = 0       # somme PnL € vs Mid pour wavg
    current_pnl_poids = 0     # somme Poids_EUR des fills avec PnL non-NaN
    current_brokers = {}      # broker_name -> Poids_EUR cumulé

    def flush_segment():
        """Sauvegarde le segment courant."""
        # PnL moyen pondéré du segment
        seg_pnl_mid = current_pnl_sum / current_pnl_poids if current_pnl_poids > 0 else np.nan
        segments.append({
            'label': current_val,
            'start': current_start,
            'end': current_end,
            'gross': current_gross,
            'aggr_qty': current_aggr_qty,
            'passive_qty': current_passive_qty,
            'total_qty': current_total_qty,
            'nb_fills': current_n_fills,
            'nb_good': current_good,
            'nb_bad': current_bad,
            'nb_neutral': current_neutral,
            'pnl_mid_wavg': seg_pnl_mid,
            'brokers': dict(current_brokers),
        })

    for _, row in g_sorted.iterrows():
        val = row[group_col]
        if pd.isna(val):
            val = 'Unknown'

        if val != current_val:
            if current_val is not None:
                flush_segment()
            current_val = val
            current_start = row['Exec Timestamp (UTC)']
            current_gross = 0
            current_aggr_qty = 0
            current_passive_qty = 0
            current_total_qty = 0
            current_good = 0
            current_bad = 0
            current_neutral = 0
            current_n_fills = 0
            current_pnl_sum = 0
            current_pnl_poids = 0
            current_brokers = {}

        current_end = row['Exec Timestamp (UTC)']
        poids = row['Poids_EUR'] if pd.notna(row['Poids_EUR']) else 0
        current_gross += poids
        qty = row['Exec Qty'] if pd.notna(row['Exec Qty']) else 0
        current_total_qty += qty
        current_n_fills += 1

        flag = row.get('Aggressive Passive Flag', '')
        if flag == 'Aggressive':
            current_aggr_qty += qty
        elif flag == 'Passive':
            current_passive_qty += qty

        pnl_bps = row['P&L Bps vs Mid']
        if pd.notna(pnl_bps):
            current_pnl_sum += pnl_bps * poids
            current_pnl_poids += poids
            if pnl_bps > 0:
                current_good += 1
            elif pnl_bps < 0:
                current_bad += 1
            else:
                current_neutral += 1

        broker = row.get('Broker Name', 'Unknown')
        if pd.isna(broker):
            broker = 'Unknown'
        current_brokers[broker] = current_brokers.get(broker, 0) + poids

    if current_val is not None:
        flush_segment()

    return segments

def build_timeline(g, group_col):
    """Construit une timeline lisible.
    Format : "Lit 35% [Aggr:30%] (08:12-08:31) → SI 50% [Aggr:95%] (08:31-09:15)"
    """
    segments = build_segments(g, group_col)
    total_gross = sum(s['gross'] for s in segments)
    if total_gross == 0:
        return ''

    parts = []
    for s in segments:
        pct = s['gross'] / total_gross * 100
        aggr_pct = s['aggr_qty'] / s['total_qty'] * 100 if s['total_qty'] > 0 else 0
        t_start = s['start'].strftime('%H:%M') if pd.notna(s['start']) else '?'
        t_end = s['end'].strftime('%H:%M') if pd.notna(s['end']) else '?'
        parts.append(f"{s['label']} {pct:.0f}% [Aggr:{aggr_pct:.0f}%] ({t_start}-{t_end})")

    return ' \u2192 '.join(parts)

# =============================================================================
# 7. VISUALISATION TIMELINE D'UN ORDRE
# Appeler plot_order_timeline(df, order_id) avec un Order Id spécifique.
# Si order_id=None, ne fait rien.
# =============================================================================

# Couleurs par type de venue
VENUE_COLORS = {
    'SI - Systematic Internaliser': '#e74c3c',  # rouge
    'Dark': '#2c3e50',                           # bleu foncé
    'Lit': '#27ae60',                            # vert
    'Primary Exchange': '#3498db',               # bleu clair
    'Periodic Auction (Lit)': '#f39c12',         # orange
    'OTC': '#95a5a6',                            # gris
    'Unknown': '#bdc3c7',                        # gris clair
}

# Noms courts pour l'affichage
VENUE_SHORT = {
    'SI - Systematic Internaliser': 'SI',
    'Primary Exchange': 'Primary',
    'Periodic Auction (Lit)': 'Auction',
}

def format_gross(val):
    """Formate un montant en k€ ou M€ lisible."""
    if val >= 1_000_000:
        return f"{val / 1_000_000:.1f}M€"
    elif val >= 1_000:
        return f"{val / 1_000:.0f}k€"
    else:
        return f"{val:.0f}€"

def plot_order_timeline(df, order_id=None):
    """Trace la timeline d'un ordre : une rangée de blocs par venue, avec toutes les infos.

    Chaque bloc = un segment de venue consécutif, contenant :
    - Nom de la venue, nb de fills
    - Gross Amount € + % du trade total
    - Brokers impliqués (avec % chacun)
    - Ratio d'agressivité / passivité
    - Ratio Good / Bad fills
    - PnL moyen pondéré vs Mid

    Axe X = timestamps réels. Largeur proportionnelle au temps.
    Si order_id=None, ne fait rien.
    """
    if order_id is None:
        return

    fills = df[df['Order Id'] == order_id].sort_values('Exec Timestamp (UTC)')
    if len(fills) == 0:
        print(f"Order Id '{order_id}' non trouvé.")
        return

    # Infos de l'ordre
    side = fills['Side'].iloc[0]
    instrument = fills['Instrument Name'].iloc[0]
    isin = fills['ISIN'].iloc[0]
    n_fills = len(fills)
    total_gross_order = fills['Poids_EUR'].sum()

    # Segments par venue
    venue_segs = build_segments(fills, 'Venue Category')

    # Timestamps de référence
    t0 = fills['Exec Timestamp (UTC)'].min()
    t_end_order = fills['Exec Timestamp (UTC)'].max()
    total_duration_s = max((t_end_order - t0).total_seconds(), 1)

    # --- Calcul des positions X en timestamps réels ---
    # On utilise les timestamps directement (pas des minutes relatives)
    fig, ax = plt.subplots(figsize=(18, 5))

    fig.suptitle(
        f"{side} {instrument} ({isin}) — Order {order_id} — {n_fills} fills — {format_gross(total_gross_order)} total",
        fontsize=13, fontweight='bold', y=0.98
    )

    bar_height = 0.8
    n_segs = len(venue_segs)

    for i, seg in enumerate(venue_segs):
        # Position X : secondes depuis t0
        x_start = (seg['start'] - t0).total_seconds()
        x_end = (seg['end'] - t0).total_seconds()
        width = max(x_end - x_start, total_duration_s * 0.03)  # min 3% pour visibilité

        # Couleur par venue
        color = VENUE_COLORS.get(seg['label'], '#bdc3c7')
        ax.barh(0, width, left=x_start, height=bar_height, color=color,
                edgecolor='white', linewidth=2, alpha=0.9)

        # --- Texte dans le bloc ---
        label_short = VENUE_SHORT.get(seg['label'], seg['label'])

        # % du trade total
        pct_trade = seg['gross'] / total_gross_order * 100 if total_gross_order > 0 else 0

        # Brokers : top brokers avec %
        total_seg_gross = seg['gross']
        broker_strs = []
        for bk, bk_gross in sorted(seg['brokers'].items(), key=lambda x: -x[1]):
            bk_pct = bk_gross / total_seg_gross * 100 if total_seg_gross > 0 else 0
            broker_strs.append(f"{bk} {bk_pct:.0f}%")
        brokers_txt = " / ".join(broker_strs[:3])  # max 3 brokers affichés

        # Agressivité
        aggr_pct = seg['aggr_qty'] / seg['total_qty'] * 100 if seg['total_qty'] > 0 else 0
        pass_pct = seg['passive_qty'] / seg['total_qty'] * 100 if seg['total_qty'] > 0 else 0

        # Good / Bad
        n_quality = seg['nb_good'] + seg['nb_bad'] + seg['nb_neutral']
        good_r = seg['nb_good'] / n_quality * 100 if n_quality > 0 else 0
        bad_r = seg['nb_bad'] / n_quality * 100 if n_quality > 0 else 0

        # PnL
        pnl_txt = f"{seg['pnl_mid_wavg']:+.1f}bps" if pd.notna(seg['pnl_mid_wavg']) else "N/A"

        # Assemblage du texte
        txt = (
            f"{label_short}\n"
            f"{seg['nb_fills']} fills\n"
            f"{format_gross(seg['gross'])} ({pct_trade:.0f}%)\n"
            f"{brokers_txt}\n"
            f"Aggr:{aggr_pct:.0f}% Pas:{pass_pct:.0f}%\n"
            f"G:{good_r:.0f}% B:{bad_r:.0f}%\n"
            f"PnL Mid: {pnl_txt}"
        )

        # Taille de police adaptée à la largeur du bloc
        fontsize = 7 if width / total_duration_s > 0.15 else 6
        ax.text(x_start + width / 2, 0, txt, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white',
                linespacing=1.3)

        # Heure sous le bloc
        t_label = seg['start'].strftime('%H:%M:%S') if pd.notna(seg['start']) else '?'
        ax.text(x_start, -bar_height / 2 - 0.08, t_label, ha='left', va='top',
                fontsize=7, color='#333333', rotation=0)

    # Heure de fin du dernier segment
    last_end = venue_segs[-1]['end']
    if pd.notna(last_end):
        x_last = (last_end - t0).total_seconds()
        ax.text(x_last, -bar_height / 2 - 0.08, last_end.strftime('%H:%M:%S'),
                ha='right', va='top', fontsize=7, color='#333333')

    # Mise en forme
    ax.set_yticks([])
    ax.set_ylim(-0.7, 0.7)
    ax.set_xlabel('Temps (secondes depuis le 1er fill)', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Légende venue
    venue_patches = [mpatches.Patch(color=c, label=VENUE_SHORT.get(l, l))
                     for l, c in VENUE_COLORS.items()
                     if any(s['label'] == l for s in venue_segs)]
    ax.legend(handles=venue_patches, loc='upper right', fontsize=8, ncol=2,
              framealpha=0.8)

    plt.tight_layout()
    plt.savefig(f"timeline_order_{order_id}.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Plot sauvé : timeline_order_{order_id}.png")

# =============================================================================
# 8. CONSTRUCTION DU DATASET TRADE-LEVEL ENRICHI
# 1 ligne = 1 Order Id, pondéré par Gross Amount € (Poids_EUR)
# Inclut les timelines venue et broker
# =============================================================================

print(f"\nConstruction du trade-level enrichi...")
results = []

for order_id, g in df.groupby('Order Id'):

    total_qty = g['Exec Qty'].sum()
    total_poids = g['Poids_EUR'].sum()
    n_fills = len(g)

    # --- Fonction locale de moyenne pondérée par Poids_EUR ---
    def wavg_local(col):
        valid = g[[col, 'Poids_EUR']].dropna()
        if valid['Poids_EUR'].sum() > 0:
            return (valid[col] * valid['Poids_EUR']).sum() / valid['Poids_EUR'].sum()
        return np.nan

    # --- Identifiants et contexte ---
    side = g['Side'].iloc[0]
    isin = g['ISIN'].iloc[0]
    instrument = g['Instrument Name'].iloc[0]
    area = g['Area'].iloc[0]
    quotation_country = g['Quotation Country'].iloc[0]
    order_qty = g['Order Qty'].iloc[0]

    # --- VWAP (toujours pondéré par Exec Qty, c'est sa définition) ---
    vwap_valid = g[['Exec Price', 'Exec Qty']].dropna()
    vwap = (vwap_valid['Exec Price'] * vwap_valid['Exec Qty']).sum() / vwap_valid['Exec Qty'].sum() if vwap_valid['Exec Qty'].sum() > 0 else np.nan

    # --- PnL pondéré par Gross Amount € ---
    wpnl_fartouch = wavg_local('P&L Bps vs Fartouch')
    wpnl_mid = wavg_local('P&L Bps vs Mid')

    # --- Spread Capture : uniquement sur fills Continuous ---
    g_cont = g[g['Market_Phase'] == 'Continuous']
    cont_poids = g_cont['Poids_EUR'].sum()
    if cont_poids > 0:
        wavg_spread_capture = wavg(g_cont, 'Spread Capture')
        wavg_perf_vs_spread = wavg(g_cont, '% Perf vs Spread')
        wavg_spread_bps = wavg(g_cont, 'Spread Bps')
    else:
        wavg_spread_capture = np.nan
        wavg_perf_vs_spread = np.nan
        wavg_spread_bps = np.nan

    # --- Timing ---
    ts_min = g['Exec Timestamp (UTC)'].min()
    ts_max = g['Exec Timestamp (UTC)'].max()
    duration_s = (ts_max - ts_min).total_seconds() if pd.notna(ts_min) and pd.notna(ts_max) else 0

    h = int(duration_s // 3600)
    m = int((duration_s % 3600) // 60)
    s = int(duration_s % 60)
    ms = int((duration_s % 1) * 1000)
    duration_fmt = f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    placement_ts = g['Placement Timestamp'].min() if 'Placement Timestamp' in g.columns else pd.NaT
    pickup_ts = g['Pick Up Timestamp'].min() if 'Pick Up Timestamp' in g.columns else pd.NaT
    delay_placement = (ts_min - placement_ts).total_seconds() if pd.notna(ts_min) and pd.notna(placement_ts) else np.nan
    delay_pickup = (ts_min - pickup_ts).total_seconds() if pd.notna(ts_min) and pd.notna(pickup_ts) else np.nan
    exec_intensity = total_qty / (duration_s / 60) if duration_s > 0 else np.nan

    # --- Brokers ---
    broker_volumes = g.groupby('Broker Name')['Poids_EUR'].sum()
    num_brokers = len(broker_volumes)
    broker_max_share = broker_volumes.max() / total_poids if total_poids > 0 else np.nan
    broker_shares = broker_volumes / total_poids if total_poids > 0 else broker_volumes
    broker_hhi = (broker_shares ** 2).sum() if total_poids > 0 else np.nan

    # --- Ratios de venue (pondérés par Poids_EUR) ---
    venue_volumes = g.groupby('Venue Category')['Poids_EUR'].sum()
    si_ratio = venue_volumes.get('SI - Systematic Internaliser', 0) / total_poids if total_poids > 0 else 0
    dark_ratio = venue_volumes.get('Dark', 0) / total_poids if total_poids > 0 else 0
    lit_ratio = venue_volumes.get('Lit', 0) / total_poids if total_poids > 0 else 0
    primary_ratio = venue_volumes.get('Primary Exchange', 0) / total_poids if total_poids > 0 else 0
    auction_ratio = venue_volumes.get('Periodic Auction (Lit)', 0) / total_poids if total_poids > 0 else 0
    otc_ratio = venue_volumes.get('OTC', 0) / total_poids if total_poids > 0 else 0
    venue_count = g['Venue (MIC)'].nunique()

    # --- Ratios Aggressive / Passive (pondérés par Poids_EUR) ---
    flag_volumes = g.groupby('Aggressive Passive Flag')['Poids_EUR'].sum()
    aggressive_ratio = flag_volumes.get('Aggressive', 0) / total_poids if total_poids > 0 else 0
    passive_ratio = flag_volumes.get('Passive', 0) / total_poids if total_poids > 0 else 0
    neutral_ratio = flag_volumes.get('Neutral', 0) / total_poids if total_poids > 0 else 0

    # --- Répartition du volume par phase de marché ---
    phase_volumes = g.groupby('Market_Phase')['Poids_EUR'].sum()
    pct_continuous = phase_volumes.get('Continuous', 0) / total_poids if total_poids > 0 else 0
    pct_auction = phase_volumes.get('Auction', 0) / total_poids if total_poids > 0 else 0
    pct_off = (phase_volumes.get('Off_Hours', 0) + phase_volumes.get('Pre_Market', 0) + phase_volumes.get('Post_Market', 0)) / total_poids if total_poids > 0 else 0

    # --- Timelines ---
    timeline_venue = build_timeline(g, 'Venue Category')
    timeline_broker = build_timeline(g, 'Broker Name')

    # --- Qualité des fills : ratios Good / Bad / Neutral ---
    nb_good = (g['P&L Bps vs Mid'] > 0).sum()
    nb_bad = (g['P&L Bps vs Mid'] < 0).sum()
    nb_neutral_fills = n_fills - nb_good - nb_bad
    good_fill_ratio = nb_good / n_fills if n_fills > 0 else 0
    bad_fill_ratio = nb_bad / n_fills if n_fills > 0 else 0
    neutral_fill_ratio = nb_neutral_fills / n_fills if n_fills > 0 else 0

    # --- Assemblage ---
    result = {
        'Order Id': order_id,
        'Side': side,
        'ISIN': isin,
        'Instrument Name': instrument,
        'Area': area,
        'Quotation Country': quotation_country,
        'Order_Qty': order_qty,
        'Total_Exec_Qty': total_qty,
        'Total_Gross_EUR': total_poids,
        'Num_Fills': n_fills,

        # Prix et PnL (pondérés par Gross Amount €)
        'VWAP': vwap,
        'WAvg_PnL_bps_VS_Fartouch': wpnl_fartouch,
        'WAvg_PnL_bps_VS_Mid': wpnl_mid,

        # Spread (Continuous uniquement, pondéré par Gross Amount €)
        'WAvg_Spread_Capture_Cont': wavg_spread_capture,
        'WAvg_Perf_vs_Spread_Cont': wavg_perf_vs_spread,
        'WAvg_Spread_Bps_Cont': wavg_spread_bps,

        # Timing
        'Exec_Start': ts_min,
        'Exec_End': ts_max,
        'Duration_Seconds': duration_s,
        'Duration_Formatted': duration_fmt,
        'Placement_Timestamp': placement_ts,
        'Pickup_Timestamp': pickup_ts,
        'Delay_Placement_to_Exec_s': delay_placement,
        'Delay_Pickup_to_Exec_s': delay_pickup,
        'Exec_Intensity_Qty_per_Min': exec_intensity,

        # Brokers (pondérés par Gross Amount €)
        'Num_Brokers': num_brokers,
        'Broker_Max_Share': broker_max_share,
        'Broker_HHI': broker_hhi,

        # Venues (pondérés par Gross Amount €)
        'SI_Ratio': si_ratio,
        'Dark_Ratio': dark_ratio,
        'Lit_Ratio': lit_ratio,
        'Primary_Ratio': primary_ratio,
        'Auction_Ratio': auction_ratio,
        'OTC_Ratio': otc_ratio,
        'Venue_Count': venue_count,

        # Aggressive / Passive (pondérés par Gross Amount €)
        'Aggressive_Ratio': aggressive_ratio,
        'Passive_Ratio': passive_ratio,
        'Neutral_Ratio': neutral_ratio,

        # Phases de marché
        'Pct_Volume_Continuous': pct_continuous,
        'Pct_Volume_Auction': pct_auction,
        'Pct_Volume_Off_Hours': pct_off,

        # Timelines
        'Timeline_Venue': timeline_venue,
        'Timeline_Broker': timeline_broker,

        # Qualité des fills (ratios)
        'Good_Fill_Ratio': good_fill_ratio,
        'Bad_Fill_Ratio': bad_fill_ratio,
        'Neutral_Fill_Ratio': neutral_fill_ratio,
    }
    results.append(result)

# =============================================================================
# 9. ANALYSES DESCRIPTIVES
# Toutes les moyennes pondérées utilisent Gross Amount € (Poids_EUR)
# =============================================================================

trade_level = pd.DataFrame(results)
print(f"\nTrade-level construit : {len(trade_level)} ordres, {len(trade_level.columns)} colonnes")

print("\n=== Analyses descriptives ===")

def wavg_fill(group, col, poids='Poids_EUR'):
    """Moyenne pondérée d'une colonne par Gross Amount €, en ignorant les NaN."""
    valid = group[[col, poids]].dropna()
    if valid[poids].sum() > 0:
        return (valid[col] * valid[poids]).sum() / valid[poids].sum()
    return np.nan

onglets = {}

# ---- 1. Stats générales ----
cols_stats = [
    'WAvg_PnL_bps_VS_Fartouch', 'WAvg_PnL_bps_VS_Mid',
    'WAvg_Spread_Capture_Cont', 'WAvg_Perf_vs_Spread_Cont', 'WAvg_Spread_Bps_Cont',
    'Duration_Seconds', 'Num_Fills', 'Total_Exec_Qty', 'Total_Gross_EUR',
    'Aggressive_Ratio', 'Passive_Ratio', 'Neutral_Ratio',
    'SI_Ratio', 'Dark_Ratio', 'Lit_Ratio',
    'Num_Brokers', 'Broker_Max_Share',
    'Pct_Volume_Continuous', 'Pct_Volume_Auction',
    'Good_Fill_Ratio', 'Bad_Fill_Ratio', 'Neutral_Fill_Ratio',
]
onglets['Stats_Generales'] = trade_level[cols_stats].describe().T
print("  Stats générales OK")

# ---- 2. Par broker ----
broker_stats = df.groupby('Broker Name').apply(
    lambda x: pd.Series({
        'Nb_Fills': len(x),
        'Nb_Orders': x['Order Id'].nunique(),
        'Total_Gross_EUR': x['Poids_EUR'].sum(),
        'W_PnL_Fartouch': wavg_fill(x, 'P&L Bps vs Fartouch'),
        'W_PnL_Mid': wavg_fill(x, 'P&L Bps vs Mid'),
        'W_Spread_Capture': wavg_fill(x, 'Spread Capture'),
        'Avg_PnL_Fartouch': x['P&L Bps vs Fartouch'].mean(),
        'Avg_PnL_Mid': x['P&L Bps vs Mid'].mean(),
        'Avg_Spread_Capture': x['Spread Capture'].mean(),
    })
).sort_values('Total_Gross_EUR', ascending=False)
onglets['Par_Broker'] = broker_stats
print("  Par broker OK")

# ---- 3. Par venue category ----
venue_stats = df.groupby('Venue Category').apply(
    lambda x: pd.Series({
        'Nb_Fills': len(x),
        'Total_Gross_EUR': x['Poids_EUR'].sum(),
        'W_PnL_Fartouch': wavg_fill(x, 'P&L Bps vs Fartouch'),
        'W_PnL_Mid': wavg_fill(x, 'P&L Bps vs Mid'),
        'W_Spread_Capture': wavg_fill(x, 'Spread Capture'),
        'W_Spread_Bps': wavg_fill(x, 'Spread Bps'),
        'Avg_PnL_Fartouch': x['P&L Bps vs Fartouch'].mean(),
        'Avg_PnL_Mid': x['P&L Bps vs Mid'].mean(),
        'Avg_Spread_Capture': x['Spread Capture'].mean(),
        'Avg_Spread_Bps': x['Spread Bps'].mean(),
    })
).sort_values('Total_Gross_EUR', ascending=False)
onglets['Par_Venue'] = venue_stats
print("  Par venue OK")

# ---- 4. Par bucket d'agressivité ----
trade_level['Bucket_Agressivite'] = pd.cut(
    trade_level['Aggressive_Ratio'],
    bins=[0, 0.25, 0.50, 0.75, 1.0],
    labels=['0-25%', '25-50%', '50-75%', '75-100%'],
    include_lowest=True
)
bucket_agg = trade_level.groupby('Bucket_Agressivite', observed=False).agg(
    Nb_Ordres=('Order Id', 'count'),
    Avg_PnL_Fartouch=('WAvg_PnL_bps_VS_Fartouch', 'mean'),
    Avg_PnL_Mid=('WAvg_PnL_bps_VS_Mid', 'mean'),
    Avg_Spread_Capture=('WAvg_Spread_Capture_Cont', 'mean'),
    Avg_Duration_s=('Duration_Seconds', 'mean'),
    Avg_Num_Fills=('Num_Fills', 'mean'),
    Avg_Good_Fill_Ratio=('Good_Fill_Ratio', 'mean'),
    Avg_Bad_Fill_Ratio=('Bad_Fill_Ratio', 'mean'),
)
onglets['Par_Bucket_Agressivite'] = bucket_agg
print("  Par bucket agressivité OK")

# ---- 5. Par bucket de durée ----
trade_level['Bucket_Duree'] = pd.cut(
    trade_level['Duration_Seconds'],
    bins=[-0.001, 0, 60, 600, 3600, trade_level['Duration_Seconds'].max() + 1],
    labels=['Instantané (0s)', 'Très court (<1min)', 'Court (1-10min)', 'Moyen (10-60min)', 'Long (>1h)'],
)
bucket_duree = trade_level.groupby('Bucket_Duree', observed=False).agg(
    Nb_Ordres=('Order Id', 'count'),
    Avg_PnL_Fartouch=('WAvg_PnL_bps_VS_Fartouch', 'mean'),
    Avg_PnL_Mid=('WAvg_PnL_bps_VS_Mid', 'mean'),
    Avg_Spread_Capture=('WAvg_Spread_Capture_Cont', 'mean'),
    Avg_Aggressive_Ratio=('Aggressive_Ratio', 'mean'),
    Avg_Num_Fills=('Num_Fills', 'mean'),
    Avg_Good_Fill_Ratio=('Good_Fill_Ratio', 'mean'),
    Avg_Bad_Fill_Ratio=('Bad_Fill_Ratio', 'mean'),
)
onglets['Par_Bucket_Duree'] = bucket_duree
print("  Par bucket durée OK")

# ---- 6. Par venue dominante ----
venue_cols = ['SI_Ratio', 'Dark_Ratio', 'Lit_Ratio', 'Primary_Ratio', 'Auction_Ratio', 'OTC_Ratio']
trade_level['Venue_Dominante'] = trade_level[venue_cols].idxmax(axis=1).str.replace('_Ratio', '')
bucket_venue = trade_level.groupby('Venue_Dominante', observed=False).agg(
    Nb_Ordres=('Order Id', 'count'),
    Avg_PnL_Fartouch=('WAvg_PnL_bps_VS_Fartouch', 'mean'),
    Avg_PnL_Mid=('WAvg_PnL_bps_VS_Mid', 'mean'),
    Avg_Spread_Capture=('WAvg_Spread_Capture_Cont', 'mean'),
    Avg_Aggressive_Ratio=('Aggressive_Ratio', 'mean'),
    Avg_Good_Fill_Ratio=('Good_Fill_Ratio', 'mean'),
    Avg_Bad_Fill_Ratio=('Bad_Fill_Ratio', 'mean'),
).sort_values('Nb_Ordres', ascending=False)
onglets['Par_Venue_Dominante'] = bucket_venue
print("  Par venue dominante OK")

# ---- 7. Broker × Agressivité ----
broker_agg = df.groupby('Broker Name').apply(
    lambda x: pd.Series({
        'Nb_Fills': len(x),
        'Total_Gross_EUR': x['Poids_EUR'].sum(),
        'W_Aggressive_Ratio': (x.loc[x['Aggressive Passive Flag'] == 'Aggressive', 'Poids_EUR'].sum()) / x['Poids_EUR'].sum() if x['Poids_EUR'].sum() > 0 else 0,
        'W_Passive_Ratio': (x.loc[x['Aggressive Passive Flag'] == 'Passive', 'Poids_EUR'].sum()) / x['Poids_EUR'].sum() if x['Poids_EUR'].sum() > 0 else 0,
        'Aggressive_Ratio_Fills': (x['Aggressive Passive Flag'] == 'Aggressive').sum() / len(x),
        'Passive_Ratio_Fills': (x['Aggressive Passive Flag'] == 'Passive').sum() / len(x),
        'W_PnL_Fartouch': wavg_fill(x, 'P&L Bps vs Fartouch'),
        'W_PnL_Mid': wavg_fill(x, 'P&L Bps vs Mid'),
        'W_Spread_Capture': wavg_fill(x, 'Spread Capture'),
        'Avg_PnL_Fartouch': x['P&L Bps vs Fartouch'].mean(),
        'Avg_PnL_Mid': x['P&L Bps vs Mid'].mean(),
        'Avg_Spread_Capture': x['Spread Capture'].mean(),
    })
).sort_values('Total_Gross_EUR', ascending=False)
onglets['Broker_Agressivite'] = broker_agg
print("  Broker × Agressivité OK")

# ---- 8. Venue × Agressivité ----
venue_agg = df.groupby('Venue Category').apply(
    lambda x: pd.Series({
        'Nb_Fills': len(x),
        'Total_Gross_EUR': x['Poids_EUR'].sum(),
        'W_Aggressive_Ratio': (x.loc[x['Aggressive Passive Flag'] == 'Aggressive', 'Poids_EUR'].sum()) / x['Poids_EUR'].sum() if x['Poids_EUR'].sum() > 0 else 0,
        'W_Passive_Ratio': (x.loc[x['Aggressive Passive Flag'] == 'Passive', 'Poids_EUR'].sum()) / x['Poids_EUR'].sum() if x['Poids_EUR'].sum() > 0 else 0,
        'W_Neutral_Ratio': (x.loc[x['Aggressive Passive Flag'] == 'Neutral', 'Poids_EUR'].sum()) / x['Poids_EUR'].sum() if x['Poids_EUR'].sum() > 0 else 0,
        'Aggressive_Ratio_Fills': (x['Aggressive Passive Flag'] == 'Aggressive').sum() / len(x),
        'Passive_Ratio_Fills': (x['Aggressive Passive Flag'] == 'Passive').sum() / len(x),
        'Neutral_Ratio_Fills': (x['Aggressive Passive Flag'] == 'Neutral').sum() / len(x),
        'W_PnL_Fartouch': wavg_fill(x, 'P&L Bps vs Fartouch'),
        'W_PnL_Mid': wavg_fill(x, 'P&L Bps vs Mid'),
        'W_Spread_Capture': wavg_fill(x, 'Spread Capture'),
        'Avg_PnL_Fartouch': x['P&L Bps vs Fartouch'].mean(),
        'Avg_PnL_Mid': x['P&L Bps vs Mid'].mean(),
        'Avg_Spread_Capture': x['Spread Capture'].mean(),
    })
).sort_values('Total_Gross_EUR', ascending=False)
onglets['Venue_Agressivite'] = venue_agg
print("  Venue × Agressivité OK")

# =============================================================================
# 10. EXPORT UNIQUE — results_TCA_Analysis.xlsx
# Toutes les feuilles dans un seul fichier
# =============================================================================

print("\n=== Export final ===")
with pd.ExcelWriter("results_TCA_Analysis.xlsx", engine="openpyxl") as writer:

    # Feuille 1 : Fills enrichis (1 ligne = 1 fill)
    cols_export_fill = [
        'Order Id', 'Fill_Rank', 'Nb_Fills_Ordre',
        'Exec Timestamp (UTC)', 'Market_Phase',
        'Side', 'ISIN', 'Instrument Name',
        'Exec Qty', 'Exec Price', 'Exec Ccy', 'Gross Amount €', 'Poids_EUR',
        'Venue Category', 'Venue (MIC)', 'Venue Name',
        'Broker Name', 'Broker Group',
        'Aggressive Passive Flag',
        'P&L Bps vs Fartouch', 'P&L Bps vs Mid',
        'P&L € vs Fartouch', 'P&L € vs Mid',
        'Spread_Capture_Cont', 'Spread Bps', '% Perf vs Spread',
        'PnL_Contribution_Mid', 'Fill_Quality',
    ]
    cols_export_fill = [c for c in cols_export_fill if c in df.columns]
    df[cols_export_fill].to_excel(writer, sheet_name='Fills_Enriched', index=False)

    # Feuille 2 : Trade-level enrichi (1 ligne = 1 ordre)
    trade_level.to_excel(writer, sheet_name='Trade_Level', index=False)

    # Feuilles 3-10 : Analyses descriptives
    for nom_onglet, dataframe in onglets.items():
        dataframe.to_excel(writer, sheet_name=nom_onglet)

print(f"Export results_TCA_Analysis.xlsx OK")
print(f"  - Fills_Enriched  : {len(df)} fills")
print(f"  - Trade_Level     : {len(trade_level)} ordres")
print(f"  - + 8 onglets d'analyses descriptives")

# =============================================================================
# 11. PLOT TIMELINE (optionnel)
# Décommenter la ligne ci-dessous avec un Order Id pour visualiser un trade
# =============================================================================

# plot_order_timeline(df, order_id='XXXXXXX')  # <-- Remplacer par un vrai Order Id
