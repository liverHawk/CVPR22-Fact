"""Generate FSCIL session files for network-flow data (CICFlowMeter).

Pipeline stage 1: make session file -> (stage 2: train) -> (stage 3: test).

Reads raw CICFlowMeter files (CSV and/or Arrow IPC `.arrow`/`.feather`/`.ipc`,
matched by glob), cleans them, and writes:
  <flow-data-dir>/flows.parquet      # cleaned full data, stable RangeIndex = global row ID
  <flow-data-dir>/train.parquet      # stratified train split (index preserved)
  <flow-data-dir>/test.parquet       # stratified test split (index preserved)
  <flow-data-dir>/feature_cols.json  # numeric feature columns in order
  <flow-data-dir>/label_map.json     # label string -> 0..C-1 (base classes first)
  <flow-data-dir>/scaler.pkl         # StandardScaler fit on base-train ONLY
  <out-dir>/session_1.txt ... session_N.txt  # global row IDs, one per line
  <out-dir>/manifest.json            # class ranges, counts, seed, sha256 per file

Conventions (must match dataloader/cicflowmeter/):
  - session_1.txt: ALL base-train row IDs (full base training).
  - session_t.txt (t>1): way*shot row IDs, evenly sampled per new class.
  - scaler is fit on base-train only; never refit (leakage prevention).

Usage:
  uv run python scripts/make_session.py --config params.yaml
  uv run python scripts/make_session.py --flow-glob 'data/cicflowmeter/csv/*.csv' --dry-run
  uv run python scripts/make_session.py --base-classes BENIGN,'DoS Hulk',PortScan,DDoS,FTP-Patator,SSH-Patator --overwrite-session
"""
import argparse
import glob
import hashlib
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

DEFAULT_DROP_COLS = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_txt(path, lines):
    with open(path, 'w') as f:
        for line in lines:
            f.write(f'{line}\n')


def parse_list_opt(v):
    if v is None:
        return []
    if isinstance(v, list):
        return [str(i).strip() for i in v if str(i).strip()]
    return [p.strip() for p in str(v).split(',') if p.strip()]


def get_parser():
    p = argparse.ArgumentParser(description='Generate session files for CICFlowMeter flow data.')
    p.add_argument('--config', type=str, default=None, help='YAML config (e.g. params.yaml)')
    p.add_argument('--opts', nargs='*', default=[], metavar='KEY=VALUE')
    p.add_argument('--dataset', type=str, default='cicflowmeter')
    p.add_argument('--flow-glob', type=str, default=None, action='append',
                   help="glob for raw flow files (.csv and/or .arrow/.feather/.ipc); "
                        "repeatable. Default: data/cicflowmeter/csv/*.csv")
    p.add_argument('--flow-data-dir', type=str, default='data/cicflowmeter')
    p.add_argument('--index-list-dir', type=str, default='data/index_list')
    p.add_argument('--out-dir', type=str, default=None)
    p.add_argument('--label-col', type=str, default='Label')
    p.add_argument('--drop-cols', type=str, default=','.join(DEFAULT_DROP_COLS))
    p.add_argument('--label-aliases', type=str, default=None,
                   help='optional JSON {raw_label: unified_label} for spelling variants')
    p.add_argument('--base-classes', type=str, default=None,
                   help='comma-separated base labels in id order; default: BENIGN + most frequent')
    p.add_argument('--base-class-num', type=int, default=6)
    p.add_argument('--session-way', type=int, default=2)
    p.add_argument('--session-shot', type=int, default=5)
    p.add_argument('--exclude-labels', type=str, default='',
                   help='comma-separated labels to drop (e.g. Heartbleed,Infiltration)')
    p.add_argument('--test-size', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--session-seed', type=int, default=None)
    p.add_argument('--overwrite-session', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p


def apply_config(ns, cfg):
    from train import parse_opt_value
    known = set(vars(ns))
    for k, v in cfg.items():
        if k in known and k not in ('config', 'opts'):
            ns.__dict__[k] = v
    for item in (ns.opts or []):
        k, v = item.split('=', 1)
        k = k.strip()
        if k not in known:
            raise ValueError(f'--opts unknown key: {k}')
        ns.__dict__[k] = parse_opt_value(v)


def read_flow_file(path):
    """Read one raw flow file. Supports CSV and Arrow IPC (.arrow/.feather/.ipc)."""
    import pandas as pd
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        return pd.read_csv(path, low_memory=False)
    if ext in ('.arrow', '.feather', '.ipc'):
        return pd.read_feather(path)
    raise ValueError(f'unsupported flow file extension: {path} (use .csv or .arrow/.feather/.ipc)')


def load_clean_frames(patterns, label_col, drop_cols, aliases):
    import pandas as pd
    if isinstance(patterns, str):
        patterns = [patterns]
    paths = sorted({p for pat in patterns for p in glob.glob(pat)})
    if not paths:
        raise FileNotFoundError(f'no flow files matched: {patterns}')
    frames = []
    for path in paths:
        df = read_flow_file(path)
        df.columns = [c.strip() for c in df.columns]
        if label_col not in df.columns:
            raise ValueError(f'{path}: missing label column {label_col!r} (cols: {list(df.columns)[:5]}...)')
        df[label_col] = df[label_col].astype(str).str.strip()
        if aliases:
            df[label_col] = df[label_col].replace(aliases)
        frames.append(df)
        print(f'  loaded {path}: {len(df)} rows')
    df = pd.concat(frames, ignore_index=True)
    drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop)
    df = df.replace([float('inf'), float('-inf')], float('nan'))
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f'  concat {before} rows -> {len(df)} after dropna (dropped cols: {drop})')
    return df


def main(argv=None):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from train import load_config_file

    parser = get_parser()
    ns = parser.parse_args(argv)
    if ns.config:
        apply_config(ns, load_config_file(ns.config))

    seed = ns.session_seed if ns.session_seed is not None else ns.seed
    drop_cols = parse_list_opt(ns.drop_cols)
    exclude = set(parse_list_opt(ns.exclude_labels))
    aliases = json.load(open(ns.label_aliases)) if ns.label_aliases else {}
    out_dir = ns.out_dir or os.path.join(ns.index_list_dir, ns.dataset)
    if not ns.flow_glob:
        ns.flow_glob = ['data/cicflowmeter/csv/*.csv']

    df = load_clean_frames(ns.flow_glob, ns.label_col, drop_cols, aliases)
    if exclude:
        df = df[~df[ns.label_col].isin(exclude)].reset_index(drop=True)
        print(f'  excluded {sorted(exclude)}: {len(df)} rows remain')

    counts = df[ns.label_col].value_counts()
    print('  label distribution:')
    for lab, cnt in counts.items():
        print(f'    {lab}: {cnt}')
    tiny = counts[counts < 2]
    if len(tiny):
        print(f'  WARNING: dropping labels with <2 samples (stratify impossible): {list(tiny.index)}')
        df = df[~df[ns.label_col].isin(set(tiny.index))].reset_index(drop=True)
        counts = df[ns.label_col].value_counts()

    base_labels = parse_list_opt(ns.base_classes)
    if base_labels:
        unknown = [l for l in base_labels if l not in set(counts.index)]
        if unknown:
            raise ValueError(f'--base-classes not in data: {unknown}')
    else:
        ordered = list(counts.index)
        if 'BENIGN' in ordered:
            ordered.remove('BENIGN')
            ordered = ['BENIGN'] + ordered
        base_labels = ordered[:ns.base_class_num]
    new_labels = [l for l in counts.index if l not in set(base_labels)]
    # frequent-first keeps few-shot pools as large as possible
    new_labels = sorted(new_labels, key=lambda l: -counts[l])
    way, shot = ns.session_way, ns.session_shot
    if len(new_labels) % way:
        raise ValueError(f'{len(new_labels)} new classes not divisible by way={way}; '
                         f'use --exclude-labels or --session-way to fix')
    label_order = base_labels + new_labels
    label_map = {lab: i for i, lab in enumerate(label_order)}
    print(f'  base[{len(base_labels)}]: {base_labels}')
    print(f'  new[{len(new_labels)}]: {new_labels}')

    feature_cols = [c for c in df.columns if c != ns.label_col]
    # keep numeric features only; drop constant columns
    import pandas as pd
    num_df = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    nunique = num_df.nunique(dropna=False)
    const_cols = list(nunique[nunique <= 1].index)
    if const_cols:
        print(f'  dropping constant columns: {const_cols}')
        feature_cols = [c for c in feature_cols if c not in const_cols]
    df = df[feature_cols + [ns.label_col]]
    df['label_id'] = df[ns.label_col].map(label_map).astype(int)

    train_df, test_df = train_test_split(df, test_size=ns.test_size,
                                         stratify=df['label_id'],
                                         random_state=seed)
    train_df, test_df = train_df.sort_index(), test_df.sort_index()
    print(f'  train {len(train_df)} / test {len(test_df)}')

    base_ids = list(range(len(base_labels)))
    base_train = train_df[train_df['label_id'].isin(base_ids)]
    scaler = StandardScaler().fit(base_train[feature_cols].to_numpy(np.float32))
    print(f'  scaler fit on base-train ({len(base_train)} rows, {len(feature_cols)} feats)')

    rng = np.random.RandomState(seed)
    plan = {'session_1.txt': list(base_train.index)}
    groups = [new_labels[i:i + way] for i in range(0, len(new_labels), way)]
    for s, grp in enumerate(groups, start=2):
        picks = []
        for lab in grp:
            pool = train_df.index[train_df['label_id'] == label_map[lab]].to_numpy()
            if len(pool) < shot:
                raise ValueError(f'label {lab!r}: train pool {len(pool)} < shot={shot}; '
                                 f'exclude or merge it')
            picks.extend(str(i) for i in rng.choice(pool, size=shot, replace=False))
        plan[f'session_{s}.txt'] = picks
    sessions = len(plan)
    for name, lines in plan.items():
        print(f'  {name}: {len(lines)} rows')

    manifest = {
        'dataset': ns.dataset,
        'seed': seed,
        'base_classes': base_labels,
        'new_classes': new_labels,
        'way': way,
        'shot': shot,
        'sessions': sessions,
        'test_size': ns.test_size,
        'feature_cols': feature_cols,
        'label_map': label_map,
        'flow_glob': ns.flow_glob,
        'files': {},
    }
    if ns.dry_run:
        return manifest

    os.makedirs(ns.flow_data_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    artifacts = ['flows.parquet', 'train.parquet', 'test.parquet']
    df.to_parquet(os.path.join(ns.flow_data_dir, 'flows.parquet'), index=True)
    train_df.to_parquet(os.path.join(ns.flow_data_dir, 'train.parquet'), index=True)
    test_df.to_parquet(os.path.join(ns.flow_data_dir, 'test.parquet'), index=True)
    with open(os.path.join(ns.flow_data_dir, 'feature_cols.json'), 'w') as f:
        json.dump(feature_cols, f, indent=2)
    with open(os.path.join(ns.flow_data_dir, 'label_map.json'), 'w') as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    with open(os.path.join(ns.flow_data_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    artifacts.extend(['feature_cols.json', 'label_map.json', 'scaler.pkl'])
    for name in artifacts:
        manifest['files'][name] = sha256_file(os.path.join(ns.flow_data_dir, name))
    for name, lines in plan.items():
        path = os.path.join(out_dir, name)
        if os.path.exists(path) and not ns.overwrite_session:
            raise FileExistsError(f'{path} exists (use --overwrite-session to regenerate)')
        write_txt(path, lines)
        manifest['files'][name] = {'lines': len(lines), 'sha256': sha256_file(path)}
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f'wrote {ns.flow_data_dir} + {out_dir}/manifest.json')
    return manifest


if __name__ == '__main__':
    main()
