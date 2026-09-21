# St-INTEL

**Stackelberg-Intent Enhanced Learning based Resource Allocation in 5G Open RAN**

[English](README.md) · [연구 소개](https://wsshinskku.github.io/research/St-INTEL/) · [방법·수식](docs/METHOD.md) · [재현성](docs/REPRODUCIBILITY.md)

St-INTEL은 사업자의 처리량·지연 의도와 UE별 QoS를 함께 고려하는 Open RAN 자원 제어 프레임워크입니다. 느린 주기의 MILP가 기준 배분을 계산하고, LP 완화의 자원 제약 쌍대변수에서 희소성 가격을 구합니다. UE는 이 정보를 활용해 DDQN으로 스케줄링 요청을 학습하고, gNB가 실제 가능한 자원 승인을 결정합니다.

이 저장소는 **St-INTEL의 저자 공식 구현**입니다. Python 시뮬레이션 환경에서 최적화·학습·스케줄링·평가 전체 파이프라인을 실행할 수 있으며, 외부 에뮬레이터 연결 인터페이스도 문서화했습니다.

## 포함된 기능

- UE·사업자 의도 slack을 포함하는 이진 MILP와 LP 완화 기반 shadow price 산출
- 패킷 큐, 타임스탬프 기반 HOL 지연, eMBB·URLLC·mMTC 트래픽, 네 종류의 셀 구성
- 실제 teacher rollout으로 초기화한 replay, DDQN target network, FedProx 로컬 학습
- 셀 내·셀 간 모델 집계, 주기적·이벤트 기반 기준 배분 갱신과 cooldown
- 같은 request–grant 인터페이스를 사용하는 비교 방법과 구성요소 ablation
- 학습과 분리한 가중치 고정 평가, 체크포인트, 다중 seed 집계, 자동 검증

## 빠른 실행

Python **3.10 이상**을 사용합니다. 기본 예제는 CPU에서 실행되며 NumPy, SciPy, PyTorch를 사용합니다. CPLEX나 외부 네트워크 시뮬레이터를 설치할 필요가 없습니다.

```bash
git clone https://github.com/wsshinskku/St-INTEL.git
cd St-INTEL
python -m venv .venv
```

Linux/macOS에서는 `source .venv/bin/activate`, PowerShell에서는 `.venv\Scripts\Activate.ps1`로 가상환경을 활성화한 뒤 실행합니다.

```bash
python -m pip install -e ".[dev]"
python -m st_intel run --config configs/smoke.json --output runs/demo
python -m st_intel evaluate --checkpoint runs/demo/final.pt --output runs/eval
pytest
```

`run`은 온라인 학습 후 별도 환경에서 학습 가중치를 고정해 평가합니다. `evaluate`는 저장된 모델을 불러와 gradient 갱신 없이 평가합니다. `smoke` 설정은 전체 동작을 확인하는 짧은 예제이며, 수렴 성능을 판단하는 설정이 아닙니다. 실제 검증 범위는 [검증 기록](docs/VALIDATION.md)을 확인하세요.

`smoke`는 셀 2개·UE 8개를 사용합니다. `configs/paper_reference.json`은 셀 4개·UE 400개, 학습 600,000 step, 평가 600,000 step을 설정합니다. 자원 요구량은 [실험 설정](docs/REPRODUCIBILITY.md), 완료된 실행은 [검증 기록](docs/VALIDATION.md)에서 확인할 수 있습니다.

## 비교·ablation 실험

```bash
python -m st_intel suite --config configs/smoke.json --output runs/suite --methods st-intel ddqn fl-rl milp --seeds 1 2 3 4 5
python -m st_intel run --config configs/smoke.json --output runs/unstable --method st-intel --scenario unstable --seed 1
python -m st_intel run --config configs/smoke.json --output runs/no-warmstart --method no-warmstart --seed 1
```

| 이름 | 구성 |
| --- | --- |
| `st-intel` | 전체 St-INTEL 파이프라인 |
| `ddqn` | 가격·teacher replay·연합학습 없는 독립 DDQN |
| `fl-rl` | 가격·teacher replay 없는 연합 DDQN 비교 방법 |
| `milp` | 신경망 학습 없이 MILP 기준 요청 사용 |
| `no-price` | 관측·보상에서 가격 제외 |
| `no-warmstart` | 최초·이후 기준 배분 갱신 시 teacher replay 생성 제외 |
| `no-fl` | 연합 집계와 proximal 기준 모델 제외 |
| `no-events` | 주기적 갱신을 유지하고 이벤트 갱신 제외 |

모든 방법은 같은 request–grant 인터페이스를 사용합니다. `fl-rl`은 연합 DDQN, `milp`는 기준 배분 요청을 사용하고, `no-*` 방법은 St-INTEL의 개별 구성요소를 비활성화합니다.

`suite`는 `<output>/<method>/seed-<seed>/`에 실행을 저장하고, 고정 정책 평가 결과를 `summary.json`으로 집계합니다. `summarize`는 중복 method/seed나 서로 다른 설정·시나리오를 섞은 입력을 거부합니다.

## 결과 확인

실행 시 최종 설정과 지표를 저장하고, 학습 방법은 `final.pt`에 모델을 저장합니다. 서로 독립적인 seed의 `metrics.json`을 모아 요약할 수 있습니다.

```bash
python -m st_intel run --config configs/smoke.json --output runs/seed1 --seed 1
python -m st_intel run --config configs/smoke.json --output runs/seed2 --seed 2
python -m st_intel summarize runs/seed1/metrics.json runs/seed2/metrics.json --output runs/summary.json
```

비교 시 학습·평가 구간, offered load·delivered traffic, HOL 지연·전달 완료 패킷 지연을 구분해야 합니다. 짧은 단일 실행은 기능 검증에 해당하며 성능 우위를 입증하지 않습니다.

외부 에뮬레이터의 관측, 요청, 자원 할당, 시간 단위 연결 방법은 [연동 문서](docs/INTEGRATION.md)에, 자원 수·대역폭·정책 네트워크 구성은 [실험 설정](docs/REPRODUCIBILITY.md)에 정리했습니다.

## 문서

- [방법과 수식의 대응](docs/METHOD.md)
- [재현 설정과 구현상 가정](docs/REPRODUCIBILITY.md)
- [외부 에뮬레이터 연동 범위](docs/INTEGRATION.md)
- [테스트·실행 검증 기록](docs/VALIDATION.md)
- [인용 정보](CITATION.cff)

관련 논문은 *Computer Communications* **심사 중**입니다. 코드는 [MIT License](LICENSE)를 따릅니다.