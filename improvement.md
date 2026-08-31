# NSGA-II performance improvements

## 概要

NSGA-II の parent population 処理について、`master`（`HEAD`）との差分を確認しながら、benchmark と CPU sampling でボトルネックを特定した。

benchmark は別 worktree `/private/tmp/rustuna-profile-main` で release build して実行した。CPU profile には macOS の `sample` を使用した。

## 変更内容

### 1. trial number base から trial ID base へ変更

generation cache と parent population を `trial.number` ではなく `trial.id` で扱うようにした。

変更前は次の変換が発生していた。

```text
trial number → trial ID → trial number
```

変更後は NSGA-II 内部で一貫して次の形式になった。

```text
trial ID → trial ID
```

これにより parent cache の encode/decode における number/ID 変換と、`Vec` index 依存を削除した。

### 2. storage から trial を ID で直接取得

non-dominated sort、crowding distance、parent cache の検証で、全 trial の `HashMap` を作らず、必要な ID を `Storage::get_cached_trial()` から直接取得する形にした。

これにより、全 trial を対象にした `trials_by_id` の構築を削除した。

### 3. generation の現在位置を cache

`generation_to_ids` と `child_generation` を `GenerationCache` にまとめた。

変更前は sampling のたびに generation `0` から検索していた。

```text
0, 1, 2, ..., current_generation
```

変更後は前回の `child_generation` から開始し、現在の generation が埋まったときだけ次へ進めるようにした。

再構築時だけ `0` から child generation を計算する。

### 4. parent cache の検証結果を memoize

study attribute に保存された parent population は、generation ごとに初回読み込み時だけ JSON decode と trial 検証を行うようにした。

一度検証した parent population は sampler 内の

```rust
parent_population_cache: RwLock<HashMap<u32, Vec<u32>>>
```

に保存し、以後は storage lookup、JSON decode、trial 検証を繰り返さない。

### 5. `clone()`、sort、dedup を削除

parent ID の重複検出で、次の処理を削除した。

```rust
let mut unique_trial_ids = trial_ids.clone();
unique_trial_ids.sort_unstable();
unique_trial_ids.dedup();
```

現在は `HashSet` による一回走査で、parent ID の順序を維持したまま重複を検出する。

また、study attributes 全体の `HashMap` clone も削除し、必要な attribute だけを取得する形にした。

## benchmark 結果

以下は同じ benchmark を変更段階ごとに実行した結果である。値は各段階の単発測定であり、厳密なベンチマーク比較ではなく、ボトルネック特定用の実測値である。

| 段階 | trials | elapsed |
|---|---:|---:|
| ID base 化・直接 lookup 後、attributes clone あり | 100,000 | 4.804754 s |
| study attributes の `clone()` 削除 | 100,000 | 1.424452 s |
| generation cursor cache 導入 | 100,000 | 0.758353 s |
| 重複検出の clone/sort/dedup 削除 | 100,000 | 0.626526 s |
| parent cache 検証の generation 単位 memoize | 100,000 | 0.347564 s |

最初の測定から最後の測定まで、約92.8%短縮した。

追加の長時間測定では、generation cursor と parent cache memoize 後に次の結果を得た。

```text
5,000,000 trials   39.358140 s
20,000,000 trials  119.666002 s
```

## CPU profile で確認したボトルネック

### clone 削除前

`sample` による 1,000,000 trials の profile では、`study_attrs.clone()` に対応する
`HashMap<AttrKey, String>::clone` が以下だった。

```text
7,444 / 16,694 samples = 約44.6%
```

この結果を根拠に attributes 全体の clone を削除した。

### generation cursor と parent cache memoize 後

20,000,000 trials の実プロセスを20秒間 profile した結果は次の通りだった。

```text
sample_joint                 5,623 / 16,567 = 33.9%（inclusive）
decode_parent_trial_ids      3,019 / 16,567 = 18.2%
Storage::get_cached_trial    1,627 / 16,567 =  9.8%
```

generation の `0` からの走査は、profile 上の主要コストではなくなった。

parent cache memoize 前は `decode_parent_trial_ids()` の検証ループが大きなコストだったが、memoize 後は同じ generation に対する再検証を行わない。

## 検証

```text
cargo test -p rustuna_sampler --lib
23 passed; 0 failed
```

benchmark は毎回、main とは別の worktree で実行した。main の既存の未コミット変更および未追跡ファイルは保持している。

## 注意点

parent population cache は同一 sampler instance 内で memoize する。sampler を再生成した場合は、study attribute から読み込み直して一度検証する。

この方式では、sampler instance の生存中に study attribute が外部から書き換えられた場合、その変更を自動検出しない。study attribute を外部から変更可能な API として厳密に扱う必要がある場合は、保存済み encoded string の fingerprint などを追加して変更検出する必要がある。
