import pandas as pd
import numpy as np

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
df['Fill_Quality'] = np.where(
    df['P&L Bps vs Mid'] > 0, 'Good',
    np.where(df['P&L Bps vs Mid'] < 0, 'Bad', 'Neutral')
)

# Export fill-level enrichi
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
    'Spread Capture_Cont', 'Spread Bps', '% Perf vs Spread',
    'PnL_Contribution_Mid', 'Fill_Quality',
]
# On ne garde que les colonnes qui existent (au cas où)
cols_export_fill = [c for c in cols_export_fill if c in df.columns]
df[cols_export_fill].to_excel("fills_enriched.xlsx", index=False)
print(f"Export fills_enriched.xlsx OK ({len(df)} fills)")

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

def build_timeline(g, group_col):
    """Construit une timeline lisible pour un groupe de fills triés par timestamp.

    Regroupe les fills consécutifs ayant la même valeur de group_col en segments.
    Format : "Lit 35% [Aggr:30%] (08:12–08:31) → SI 50% [Aggr:95%] (08:31–09:15)"

    Le % = part du Gross Amount € total du trade sur ce segment.
    Aggr = part du volume agressif sur ce segment.
    """
    g_sorted = g.sort_values('Exec Timestamp (UTC)')

    segments = []
    current_val = None
    current_start = None
    current_end = None
    current_gross = 0
    current_aggr_qty = 0
    current_total_qty = 0

    for _, row in g_sorted.iterrows():
        val = row[group_col]
        if pd.isna(val):
            val = 'Unknown'

        if val != current_val:
            # On sauvegarde le segment précédent
            if current_val is not None:
                segments.append({
                    'label': current_val,
                    'start': current_start,
                    'end': current_end,
                    'gross': current_gross,
                    'aggr_qty': current_aggr_qty,
                    'total_qty': current_total_qty,
                })
            # Nouveau segment
            current_val = val
            current_start = row['Exec Timestamp (UTC)']
            current_gross = 0
            current_aggr_qty = 0
            current_total_qty = 0

        current_end = row['Exec Timestamp (UTC)']
        current_gross += row['Poids_EUR'] if pd.notna(row['Poids_EUR']) else 0
        qty = row['Exec Qty'] if pd.notna(row['Exec Qty']) else 0
        current_total_qty += qty
        if row.get('Aggressive Passive Flag') == 'Aggressive':
            current_aggr_qty += qty

    # Dernier segment
    if current_val is not None:
        segments.append({
            'label': current_val,
            'start': current_start,
            'end': current_end,
            'gross': current_gross,
            'aggr_qty': current_aggr_qty,
            'total_qty': current_total_qty,
        })

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

    return ' → '.join(parts)

# =============================================================================
# 7. CONSTRUCTION DU DATASET TRADE-LEVEL ENRICHI
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
    # Venue : "Lit 35% [Aggr:30%] (08:12-08:31) → SI 50% [Aggr:95%] (08:31-09:15)"
    timeline_venue = build_timeline(g, 'Venue Category')
    # Broker : "Citi 60% [Aggr:25%] (08:12-09:00) → MS 40% [Aggr:80%] (09:00-09:28)"
    timeline_broker = build_timeline(g, 'Broker Name')

    # --- PnL contribution : nb de fills good vs bad ---
    nb_good = (g['P&L Bps vs Mid'] > 0).sum()
    nb_bad = (g['P&L Bps vs Mid'] < 0).sum()

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

        # Qualité des fills
        'Nb_Good_Fills': nb_good,
        'Nb_Bad_Fills': nb_bad,
    }
    results.append(result)

# =============================================================================
# 8. EXPORT TRADE-LEVEL
# =============================================================================

trade_level = pd.DataFrame(results)
print(f"\nTrade-level construit : {len(trade_level)} ordres, {len(trade_level.columns)} colonnes")

trade_level.to_excel("trade_level_enriched.xlsx", index=False)
print("Export trade_level_enriched.xlsx OK")

# =============================================================================
# 9. ANALYSES DESCRIPTIVES
# Export dans analyse_descriptive.xlsx (un onglet par analyse)
# Toutes les moyennes pondérées utilisent Gross Amount € (Poids_EUR)
# =============================================================================

print("\n=== Analyses descriptives ===")

# Fonction utilitaire globale pour les groupby sur le fill-level
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
]
onglets['Stats_Generales'] = trade_level[cols_stats].describe().T
print("  Stats générales OK")

# ---- 2. Par broker (pondéré par Gross Amount € + simple) ----
broker_stats = df.groupby('Broker Name').apply(
    lambda x: pd.Series({
        'Nb_Fills': len(x),
        'Nb_Orders': x['Order Id'].nunique(),
        'Total_Gross_EUR': x['Poids_EUR'].sum(),
        # Pondéré par Gross Amount €
        'W_PnL_Fartouch': wavg_fill(x, 'P&L Bps vs Fartouch'),
        'W_PnL_Mid': wavg_fill(x, 'P&L Bps vs Mid'),
        'W_Spread_Capture': wavg_fill(x, 'Spread Capture'),
        # Non pondéré (pour comparaison)
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
).sort_values('Nb_Ordres', ascending=False)
onglets['Par_Venue_Dominante'] = bucket_venue
print("  Par venue dominante OK")

# ---- 7. Broker × Agressivité ----
broker_agg = df.groupby('Broker Name').apply(
    lambda x: pd.Series({
        'Nb_Fills': len(x),
        'Total_Gross_EUR': x['Poids_EUR'].sum(),
        # Ratios pondérés par Gross Amount €
        'W_Aggressive_Ratio': (x.loc[x['Aggressive Passive Flag'] == 'Aggressive', 'Poids_EUR'].sum()) / x['Poids_EUR'].sum() if x['Poids_EUR'].sum() > 0 else 0,
        'W_Passive_Ratio': (x.loc[x['Aggressive Passive Flag'] == 'Passive', 'Poids_EUR'].sum()) / x['Poids_EUR'].sum() if x['Poids_EUR'].sum() > 0 else 0,
        # Ratios simples (en nombre de fills)
        'Aggressive_Ratio_Fills': (x['Aggressive Passive Flag'] == 'Aggressive').sum() / len(x),
        'Passive_Ratio_Fills': (x['Aggressive Passive Flag'] == 'Passive').sum() / len(x),
        # PnL pondéré par Gross Amount €
        'W_PnL_Fartouch': wavg_fill(x, 'P&L Bps vs Fartouch'),
        'W_PnL_Mid': wavg_fill(x, 'P&L Bps vs Mid'),
        'W_Spread_Capture': wavg_fill(x, 'Spread Capture'),
        # PnL simple
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

# ---- Export multi-onglets ----
with pd.ExcelWriter("analyse_descriptive.xlsx", engine="openpyxl") as writer:
    for nom_onglet, dataframe in onglets.items():
        dataframe.to_excel(writer, sheet_name=nom_onglet)

print("\nExport analyse_descriptive.xlsx OK (8 onglets)")
print("\n=== Terminé ===")
print(f"  - fills_enriched.xlsx       : {len(df)} fills avec rang, contribution PnL, quality flag")
print(f"  - trade_level_enriched.xlsx : {len(trade_level)} ordres avec timelines venue/broker")
print(f"  - analyse_descriptive.xlsx  : 8 onglets (pondération Gross Amount €)")
