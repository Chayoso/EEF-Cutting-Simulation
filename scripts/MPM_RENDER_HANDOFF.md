# MPM mp4 재렌더 핸드오프 (trajectory → mpm.mp4)

목표: CulinaryCut 데이터셋의 각 에피소드(`mpm_render.npz`)를 로컬 CPIC MPM 시뮬로
재생·렌더해서 **입자 영상 mp4**를 만든다. 75개 = 15 과일 × 5 seed.

검증 환경: Windows + RTX 4090 (CUDA), Taichi 1.7.4. 헤드리스 GGUI(`show_window=False`)
오프스크린 렌더 정상 동작 확인. 서버(리눅스 GPU)에서도 동일하게 동작.

---

## 입력 데이터 (Drive)

각 에피소드 폴더 `<fruit>/auto_<seed>/`:
- `mpm_render.npz` — **`knife` (T,3): 칼끝 궤적, 이미 MPM 좌표계** ← 렌더에 필요한 유일 파일
- `trajectory.json` — `variation.object` (과일명), seed
- `mpm.mp4` — 기존(깨진) 참조본. `ms_base.mp4` — ManiSkill 로봇 장면. `trajectory.h5` — RDT용(렌더엔 불필요)

렌더는 `mpm_render.npz` + 과일 `.yaml`만 있으면 됨. **SAPIEN/ManiSkill 불필요(Taichi만).**

---

## 발견한 문제 4가지 + 수정

1. **기존 mpm.mp4가 깨져 있었음** — 칼도 안 보이고 바나나가 안 잘리는 정지 덩어리.
   → 좌표 문제가 아니라 아래 2,3,4가 원인.
2. **보드 SDF 빌드 크래시** — `board.obj`가 non-watertight라
   `trimesh.proximity.signed_distance`가 네이티브 크래시(exit 127).
   → `render_mpm_episode_portable.py`에서 보드를 **해석적 박스 SDF**(AABB 기반)로 대체. 이식성 ↑.
3. **컷이 안 생김 (핵심)** — `damage_v_threshold = dx·v_hat/dt = 0.0042·0.35/2e-5 ≈ 72.9 m/s`
   인데 재생 칼 속도는 ~0.3 m/s. 임계값이 240배 높아 데미지가 **절대 안 터짐**.
   → `--damage-v-hat` 로 임계값을 재생 속도에 맞춤. `0.0007 ≈ 0.146 m/s` → 컷 발생 확인.
4. **로컬에 melon config/asset 없음** → 저장소 EEF에서 `configs/fruits/melon.yaml` +
   `assets/melon.obj`(+mtl/skin) 포팅 완료.

검증: banana/auto_0 → 칼이 바나나에 정확히 내려와 접촉 지점에 빨간 데미지(컷) 발생. ✓

---

## 산출물

- `scripts/render_mpm_episode_portable.py` — 단일 에피소드 렌더 (경로 하드코딩 제거, 보드 박스SDF, 컷 튜닝 플래그)
- `scripts/render_all_mpm.py` — 데이터셋 전체 배치 + 샤딩 병렬
- `configs/fruits/melon.yaml`, `assets/melon.obj` — 추가됨

---

## 서버 실행

```bash
# 1) 단일 에피소드 (확인용)
python scripts/render_mpm_episode_portable.py \
    --npz /data/culinary_traj/banana/auto_0/mpm_render.npz \
    --config configs/fruits/banana.yaml \
    --repo /data/CPIC \
    --out /data/culinary_traj/banana/auto_0/mpm_fixed.mp4 \
    --damage-v-hat 0.0007 --fps 20 --res 720

# 2) 전체 배치 (단일 프로세스)
python scripts/render_all_mpm.py \
    --dataset-root /data/culinary_traj --repo /data/CPIC \
    --config-dir configs/fruits --damage-v-hat 0.0007 --out-name mpm_fixed.mp4

# 3) 8-way 병렬 (서버가 한 번에 더 많이 돌릴 때)
for s in $(seq 0 7); do
  python scripts/render_all_mpm.py --dataset-root /data/culinary_traj \
    --repo /data/CPIC --config-dir configs/fruits \
    --damage-v-hat 0.0007 --out-name mpm_fixed.mp4 --of 8 --shard $s &
done; wait
```

튜닝 노브:
- `--damage-v-hat` ↓ → 컷 데미지 더 강하게/빨리. (현재 0.0007. 더 진한 단면 원하면 0.0004~0.0005)
- `--knife-speed` → 데미지 모델에 보고되는 칼 속도 override.
- `--extra-descent` → 칼이 보드까지 더 깊이 통과(아래쪽 절단면까지).
- `--anchor` → 칼 xz를 실제 시드된 과일 중심으로 보정(좌표 안 맞을 때 비상용).
- `--diag` → 시드 입자 AABB vs 칼 궤적 AABB, 프레임별 max_damage 출력.

---

## 75개로 확장 시 남은 것

- Drive 파일럿엔 **banana/melon/peach 3종 × 5 = 15개**만 있음(현재 banana 5 + melon 5 + peach 5).
- 나머지 12 과일 trajectory(`mpm_render.npz`)가 있어야 75개 완성.
- **과일별 config/asset 일치 필요**: 데이터셋 과일 목록(frame.py FRUIT_CANONICAL =
  banana, apple, cucumber, melon, orange, peach, strawberry, cherry, plum, lemon, kiwi,
  shine_muscat, tomato, grape, golden_strawberry, pear)과 로컬 `configs/fruits/`가 일부 다름
  (로컬엔 avocado/mango/persimmon/pineapple_slice/watermelon_slice가 있고 cucumber/cherry/plum/
  shine_muscat/golden_strawberry/strawberry config는 없음). 해당 과일은 EEF/Cutting_rebuttal에서
  config+obj를 포팅하고, `--diag`로 시드 위치가 칼 궤적과 겹치는지 확인할 것.
- 과일마다 `--damage-v-hat` 미세조정이 필요할 수 있음(재질/속도 차이).
```
