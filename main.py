import pandas as pd
import numpy as np

# =============================================================================
# CHARGEMENT ET NETTOYAGE
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
# FILTRE : on ne garde que les ordres (Care OU Market) exécutés en ALGO
# =============================================================================

df = df[
    ((df['PM Instruction'] == 'Care') | (df['PM Instruction'] == 'Market'))
    & (df['Exec Mode'] == 'ALGO')
]
print(f"Fills après filtre : {len(df)} | Order Id uniques : {df['Order Id'].nunique()}")

# =============================================================================
# CONVERSIONS DE TYPES
# =============================================================================

# Colonnes numériques — nécessaire car le header décalé peut laisser des types 'object'
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

# Colonnes timestamp
for col in ['Exec Timestamp (UTC)', 'Placement Timestamp', 'Pick Up Timestamp']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# =============================================================================
# PHASE DE MARCHÉ (au niveau fill)
# On classe chaque fill selon le moment où il a été exécuté.
# Permet ensuite de filtrer les métriques de spread qui n'ont de sens
# que pendant le trading continu (pas pendant les auctions ni hors marché).
# =============================================================================

def classer_market_phase(row):
    """Détermine la phase de marché d'un fill à partir de son heure et de son type d'exécution."""
    # Si le Liquidity Indicator Label indique une auction, c'est une auction
    label = str(row.get('Liquidity Indicator Label', ''))
    if 'Auction' in label or 'auction' in label:
        return 'Auction'
    # Sinon on se base sur l'heure UTC
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
# CONSTRUCTION DU DATASET TRADE-LEVEL
# Chaque ligne = 1 Order Id = 1 trade agrégé à partir de ses fills
# =============================================================================

print(f"\nConstruction du trade-level...")
results = []

for order_id, g in df.groupby('Order Id'):

    total_qty = g['Exec Qty'].sum()
    n_fills = len(g)

    # --- Fonction utilitaire : moyenne pondérée par Exec Qty ---
    def wavg(col):
        """Moyenne pondérée par la quantité exécutée. Retourne NaN si pas de données."""
        valid = g[[col, 'Exec Qty']].dropna()
        if valid['Exec Qty'].sum() > 0:
            return (valid[col] * valid['Exec Qty']).sum() / valid['Exec Qty'].sum()
        return np.nan

    # --- Identifiants et contexte ---
    # Champs normalement constants par ordre, on prend la première valeur
    side = g['Side'].iloc[0]
    isin = g['ISIN'].iloc[0]
    instrument = g['Instrument Name'].iloc[0]
    area = g['Area'].iloc[0]
    quotation_country = g['Quotation Country'].iloc[0]
    order_qty = g['Order Qty'].iloc[0]

    # --- VWAP ---
    # Prix moyen pondéré par quantité : le prix "réel" auquel l'ordre a été exécuté
    vwap = wavg('Exec Price')

    # --- PnL pondéré par quantité ---
    # Mesure la performance moyenne de chaque unité exécutée
    # vs Fartouch = par rapport au pire côté du carnet (conservateur)
    # vs Mid = par rapport au milieu du spread (standard)
    wpnl_fartouch = wavg('P&L Bps vs Fartouch')
    wpnl_mid = wavg('P&L Bps vs Mid')

    # --- Spread Capture et % Perf vs Spread : uniquement sur le continuous ---
    # Ces métriques n'ont de sens que quand il y a un vrai spread bid-ask,
    # donc on exclut les auctions et le hors-marché
    g_cont = g[g['Market_Phase'] == 'Continuous']
    cont_qty = g_cont['Exec Qty'].sum()
    if cont_qty > 0:
        valid_sc = g_cont[['Spread Capture', 'Exec Qty']].dropna()
        wavg_spread_capture = (valid_sc['Spread Capture'] * valid_sc['Exec Qty']).sum() / valid_sc['Exec Qty'].sum() if valid_sc['Exec Qty'].sum() > 0 else np.nan
        valid_perf = g_cont[['% Perf vs Spread', 'Exec Qty']].dropna()
        wavg_perf_vs_spread = (valid_perf['% Perf vs Spread'] * valid_perf['Exec Qty']).sum() / valid_perf['Exec Qty'].sum() if valid_perf['Exec Qty'].sum() > 0 else np.nan
        valid_spd = g_cont[['Spread Bps', 'Exec Qty']].dropna()
        wavg_spread_bps = (valid_spd['Spread Bps'] * valid_spd['Exec Qty']).sum() / valid_spd['Exec Qty'].sum() if valid_spd['Exec Qty'].sum() > 0 else np.nan
    else:
        wavg_spread_capture = np.nan
        wavg_perf_vs_spread = np.nan
        wavg_spread_bps = np.nan

    # --- Timing ---
    ts_min = g['Exec Timestamp (UTC)'].min()
    ts_max = g['Exec Timestamp (UTC)'].max()
    duration_s = (ts_max - ts_min).total_seconds() if pd.notna(ts_min) and pd.notna(ts_max) else 0

    # Format lisible HH:MM:SS.mmm
    h = int(duration_s // 3600)
    m = int((duration_s % 3600) // 60)
    s = int(duration_s % 60)
    ms = int((duration_s % 1) * 1000)
    duration_fmt = f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    # Timestamps de placement et pick-up (min par ordre)
    placement_ts = g['Placement Timestamp'].min() if 'Placement Timestamp' in g.columns else pd.NaT
    pickup_ts = g['Pick Up Timestamp'].min() if 'Pick Up Timestamp' in g.columns else pd.NaT

    # Délai placement → première exécution (en secondes)
    # Mesure le temps entre l'envoi de l'ordre et son premier fill
    delay_placement = (ts_min - placement_ts).total_seconds() if pd.notna(ts_min) and pd.notna(placement_ts) else np.nan

    # Délai pick-up → première exécution (en secondes)
    # Mesure le temps entre la prise en charge par le broker et le premier fill
    delay_pickup = (ts_min - pickup_ts).total_seconds() if pd.notna(ts_min) and pd.notna(pickup_ts) else np.nan

    # --- Intensité d'exécution ---
    # Quantité exécutée par minute : mesure le rythme d'exécution
    # Utile pour comparer ordres lents vs rapides
    exec_intensity = total_qty / (duration_s / 60) if duration_s > 0 else np.nan

    # --- Brokers ---
    # Un même ordre peut être exécuté par plusieurs brokers
    broker_volumes = g.groupby('Broker Name')['Exec Qty'].sum()
    num_brokers = len(broker_volumes)

    # Broker_Max_Share : part du volume du broker dominant
    # Si = 1.0, un seul broker a tout exécuté. Si bas, l'ordre est fragmenté entre brokers.
    broker_max_share = broker_volumes.max() / total_qty if total_qty > 0 else np.nan

    # Broker_HHI : indice de concentration Herfindahl (somme des parts² entre 0 et 1)
    # Plus il est élevé, plus l'exécution est concentrée sur peu de brokers
    broker_shares = broker_volumes / total_qty if total_qty > 0 else broker_volumes
    broker_hhi = (broker_shares ** 2).sum() if total_qty > 0 else np.nan

    # --- Ratios de venue (pondérés par Exec Qty) ---
    # Mesurent la répartition du volume exécuté par type de venue
    # Permet d'analyser si exécuter plus sur SI, Dark, Lit etc. impacte la performance
    venue_volumes = g.groupby('Venue Category')['Exec Qty'].sum()
    si_ratio = venue_volumes.get('SI - Systematic Internaliser', 0) / total_qty if total_qty > 0 else 0
    dark_ratio = venue_volumes.get('Dark', 0) / total_qty if total_qty > 0 else 0
    lit_ratio = venue_volumes.get('Lit', 0) / total_qty if total_qty > 0 else 0
    primary_ratio = venue_volumes.get('Primary Exchange', 0) / total_qty if total_qty > 0 else 0
    auction_ratio = venue_volumes.get('Periodic Auction (Lit)', 0) / total_qty if total_qty > 0 else 0
    otc_ratio = venue_volumes.get('OTC', 0) / total_qty if total_qty > 0 else 0
    venue_count = g['Venue (MIC)'].nunique()

    # --- Ratios Aggressive / Passive (pondérés par Exec Qty) ---
    # Mesurent la part du volume exécuté de façon agressive vs passive
    # Un ordre très agressif "prend" la liquidité, un ordre passif "offre" la liquidité
    flag_volumes = g.groupby('Aggressive Passive Flag')['Exec Qty'].sum()
    aggressive_ratio = flag_volumes.get('Aggressive', 0) / total_qty if total_qty > 0 else 0
    passive_ratio = flag_volumes.get('Passive', 0) / total_qty if total_qty > 0 else 0
    neutral_ratio = flag_volumes.get('Neutral', 0) / total_qty if total_qty > 0 else 0

    # --- Répartition du volume par phase de marché ---
    # Permet de savoir si l'ordre a été exécuté surtout en continu, en auction, ou hors marché
    phase_volumes = g.groupby('Market_Phase')['Exec Qty'].sum()
    pct_continuous = phase_volumes.get('Continuous', 0) / total_qty if total_qty > 0 else 0
    pct_auction = phase_volumes.get('Auction', 0) / total_qty if total_qty > 0 else 0
    pct_off = (phase_volumes.get('Off_Hours', 0) + phase_volumes.get('Pre_Market', 0) + phase_volumes.get('Post_Market', 0)) / total_qty if total_qty > 0 else 0

    # --- Assemblage du résultat ---
    result = {
        # Identifiants
        'Order Id': order_id,
        'Side': side,
        'ISIN': isin,
        'Instrument Name': instrument,
        'Area': area,
        'Quotation Country': quotation_country,
        'Order_Qty': order_qty,
        'Total_Exec_Qty': total_qty,
        'Num_Fills': n_fills,

        # Prix et PnL
        'VWAP': vwap,
        'WAvg_PnL_bps_VS_Fartouch': wpnl_fartouch,
        'WAvg_PnL_bps_VS_Mid': wpnl_mid,

        # Spread (uniquement sur fills en continuous trading)
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

        # Brokers
        'Num_Brokers': num_brokers,
        'Broker_Max_Share': broker_max_share,
        'Broker_HHI': broker_hhi,

        # Venues (pondérés par volume)
        'SI_Ratio': si_ratio,
        'Dark_Ratio': dark_ratio,
        'Lit_Ratio': lit_ratio,
        'Primary_Ratio': primary_ratio,
        'Auction_Ratio': auction_ratio,
        'OTC_Ratio': otc_ratio,
        'Venue_Count': venue_count,

        # Aggressive / Passive (pondérés par volume)
        'Aggressive_Ratio': aggressive_ratio,
        'Passive_Ratio': passive_ratio,
        'Neutral_Ratio': neutral_ratio,

        # Phases de marché (pondérés par volume)
        'Pct_Volume_Continuous': pct_continuous,
        'Pct_Volume_Auction': pct_auction,
        'Pct_Volume_Off_Hours': pct_off,
    }
    results.append(result)

# =============================================================================
# EXPORT
# =============================================================================

trade_level = pd.DataFrame(results)
print(f"\nTrade-level construit : {len(trade_level)} ordres, {len(trade_level.columns)} colonnes")

trade_level.to_excel("trade_level_results.xlsx", index=False)
print("Export trade_level_results.xlsx OK")

# =============================================================================
# ÉTAPE 3 — ANALYSES DESCRIPTIVES
# Export dans analyse_descriptive.xlsx (un onglet par analyse)
# Chaque onglet contient les moyennes pondérées (par Exec Qty) ET simples
# pour pouvoir comparer les deux approches.
# =============================================================================

print("\n=== Analyses descriptives ===")

# Fonction utilitaire : moyenne pondérée par Exec Qty au niveau fill
def wavg_fill(group, col, poids='Exec Qty'):
    """Moyenne pondérée d'une colonne par Exec Qty, en ignorant les NaN."""
    valid = group[[col, poids]].dropna()
    if valid[poids].sum() > 0:
        return (valid[col] * valid[poids]).sum() / valid[poids].sum()
    return np.nan

onglets = {}

# ---- 1. Stats générales ----
# Vue d'ensemble des variables clés : ordres de grandeur, dispersion, outliers
cols_stats = [
    'WAvg_PnL_bps_VS_Fartouch', 'WAvg_PnL_bps_VS_Mid',
    'WAvg_Spread_Capture_Cont', 'WAvg_Perf_vs_Spread_Cont', 'WAvg_Spread_Bps_Cont',
    'Duration_Seconds', 'Num_Fills', 'Total_Exec_Qty',
    'Aggressive_Ratio', 'Passive_Ratio', 'Neutral_Ratio',
    'SI_Ratio', 'Dark_Ratio', 'Lit_Ratio',
    'Num_Brokers', 'Broker_Max_Share',
    'Pct_Volume_Continuous', 'Pct_Volume_Auction',
]
onglets['Stats_Generales'] = trade_level[cols_stats].describe().T
print("  Stats générales OK")

# ---- 2. Par broker ----
# Pour chaque broker : volume, performance pondérée et simple, style d'exécution.
# Pondéré = coût réel par action exécutée. Simple = moyenne par fill (pour comparaison).
broker_stats = df.groupby('Broker Name').apply(
    lambda x: pd.Series({
        'Nb_Fills': len(x),
        'Nb_Orders': x['Order Id'].nunique(),
        'Total_Volume': x['Exec Qty'].sum(),
        # Pondéré par Exec Qty (métrique principale)
        'W_PnL_Fartouch': wavg_fill(x, 'P&L Bps vs Fartouch'),
        'W_PnL_Mid': wavg_fill(x, 'P&L Bps vs Mid'),
        'W_Spread_Capture': wavg_fill(x, 'Spread Capture'),
        # Non pondéré (pour comparaison)
        'Avg_PnL_Fartouch': x['P&L Bps vs Fartouch'].mean(),
        'Avg_PnL_Mid': x['P&L Bps vs Mid'].mean(),
        'Avg_Spread_Capture': x['Spread Capture'].mean(),
    })
).sort_values('Total_Volume', ascending=False)
onglets['Par_Broker'] = broker_stats
print("  Par broker OK")

# ---- 3. Par venue category ----
# Pour chaque type de venue : volume, PnL, spread capture (pondéré + simple).
# Permet de voir si certains types de venue sont structurellement plus coûteux.
venue_stats = df.groupby('Venue Category').apply(
    lambda x: pd.Series({
        'Nb_Fills': len(x),
        'Total_Volume': x['Exec Qty'].sum(),
        # Pondéré
        'W_PnL_Fartouch': wavg_fill(x, 'P&L Bps vs Fartouch'),
        'W_PnL_Mid': wavg_fill(x, 'P&L Bps vs Mid'),
        'W_Spread_Capture': wavg_fill(x, 'Spread Capture'),
        'W_Spread_Bps': wavg_fill(x, 'Spread Bps'),
        # Non pondéré
        'Avg_PnL_Fartouch': x['P&L Bps vs Fartouch'].mean(),
        'Avg_PnL_Mid': x['P&L Bps vs Mid'].mean(),
        'Avg_Spread_Capture': x['Spread Capture'].mean(),
        'Avg_Spread_Bps': x['Spread Bps'].mean(),
    })
).sort_values('Total_Volume', ascending=False)
onglets['Par_Venue'] = venue_stats
print("  Par venue OK")

# ---- 4. Par bucket d'agressivité ----
# On découpe les ordres en 4 groupes selon leur taux d'agressivité.
# But : quantifier le coût de l'agressivité sur la performance.
# Est-ce que plus agressif = pire PnL ? Ou est-ce compensé par moins de slippage ?
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
# On découpe les ordres selon leur durée d'exécution avec des bornes fixes.
# But : voir si les ordres patients (longs) ont une meilleure performance,
# ou s'ils subissent plus d'adverse selection (le marché bouge contre eux).
# Note : beaucoup d'ordres ont durée = 0 (1 seul fill), on les isole.
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
# On identifie le type de venue qui a reçu le plus de volume pour chaque ordre.
# But : comparer la performance des ordres selon leur canal d'exécution principal.
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
# Pour chaque broker : son taux d'agressivité pondéré par volume, et le PnL associé.
# But : répondre à "ce broker est-il cher PARCE QU'il est agressif ?"
# Si un broker est très agressif ET a un bon PnL, il est efficace.
# Si un broker est passif ET a un mauvais PnL, il y a un problème.
broker_agg = df.groupby('Broker Name').apply(
    lambda x: pd.Series({
        'Nb_Fills': len(x),
        'Total_Volume': x['Exec Qty'].sum(),
        # Ratios pondérés par volume (combien de volume est agressif/passif)
        'W_Aggressive_Ratio': (x.loc[x['Aggressive Passive Flag'] == 'Aggressive', 'Exec Qty'].sum()) / x['Exec Qty'].sum() if x['Exec Qty'].sum() > 0 else 0,
        'W_Passive_Ratio': (x.loc[x['Aggressive Passive Flag'] == 'Passive', 'Exec Qty'].sum()) / x['Exec Qty'].sum() if x['Exec Qty'].sum() > 0 else 0,
        # Ratios simples (en nombre de fills, pour comparaison)
        'Aggressive_Ratio_Fills': (x['Aggressive Passive Flag'] == 'Aggressive').sum() / len(x),
        'Passive_Ratio_Fills': (x['Aggressive Passive Flag'] == 'Passive').sum() / len(x),
        # PnL pondéré + simple
        'W_PnL_Fartouch': wavg_fill(x, 'P&L Bps vs Fartouch'),
        'W_PnL_Mid': wavg_fill(x, 'P&L Bps vs Mid'),
        'W_Spread_Capture': wavg_fill(x, 'Spread Capture'),
        'Avg_PnL_Fartouch': x['P&L Bps vs Fartouch'].mean(),
        'Avg_PnL_Mid': x['P&L Bps vs Mid'].mean(),
        'Avg_Spread_Capture': x['Spread Capture'].mean(),
    })
).sort_values('Total_Volume', ascending=False)
onglets['Broker_Agressivite'] = broker_agg
print("  Broker × Agressivité OK")

# ---- 8. Venue × Agressivité ----
# Pour chaque type de venue : taux d'agressivité et PnL (pondéré + simple).
# But : certaines venues forcent-elles un style d'exécution ?
# Ex: sur un SI, l'exécution est souvent passive (le SI propose un prix).
# Sur un Lit, on peut être agressif ou passif selon la stratégie.
venue_agg = df.groupby('Venue Category').apply(
    lambda x: pd.Series({
        'Nb_Fills': len(x),
        'Total_Volume': x['Exec Qty'].sum(),
        # Ratios pondérés par volume
        'W_Aggressive_Ratio': (x.loc[x['Aggressive Passive Flag'] == 'Aggressive', 'Exec Qty'].sum()) / x['Exec Qty'].sum() if x['Exec Qty'].sum() > 0 else 0,
        'W_Passive_Ratio': (x.loc[x['Aggressive Passive Flag'] == 'Passive', 'Exec Qty'].sum()) / x['Exec Qty'].sum() if x['Exec Qty'].sum() > 0 else 0,
        'W_Neutral_Ratio': (x.loc[x['Aggressive Passive Flag'] == 'Neutral', 'Exec Qty'].sum()) / x['Exec Qty'].sum() if x['Exec Qty'].sum() > 0 else 0,
        # Ratios simples
        'Aggressive_Ratio_Fills': (x['Aggressive Passive Flag'] == 'Aggressive').sum() / len(x),
        'Passive_Ratio_Fills': (x['Aggressive Passive Flag'] == 'Passive').sum() / len(x),
        'Neutral_Ratio_Fills': (x['Aggressive Passive Flag'] == 'Neutral').sum() / len(x),
        # PnL pondéré + simple
        'W_PnL_Fartouch': wavg_fill(x, 'P&L Bps vs Fartouch'),
        'W_PnL_Mid': wavg_fill(x, 'P&L Bps vs Mid'),
        'W_Spread_Capture': wavg_fill(x, 'Spread Capture'),
        'Avg_PnL_Fartouch': x['P&L Bps vs Fartouch'].mean(),
        'Avg_PnL_Mid': x['P&L Bps vs Mid'].mean(),
        'Avg_Spread_Capture': x['Spread Capture'].mean(),
    })
).sort_values('Total_Volume', ascending=False)
onglets['Venue_Agressivite'] = venue_agg
print("  Venue × Agressivité OK")

# ---- Export multi-onglets ----
with pd.ExcelWriter("analyse_descriptive.xlsx", engine="openpyxl") as writer:
    for nom_onglet, dataframe in onglets.items():
        dataframe.to_excel(writer, sheet_name=nom_onglet)

print("\nExport analyse_descriptive.xlsx OK (8 onglets)")
