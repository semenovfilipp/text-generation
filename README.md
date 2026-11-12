# DialogSum: LoRA-дообучение маленькой LLM для суммаризации диалогов

Задача: улучшить абстрактивную суммаризацию диалогов ([DialogSum](https://www.kaggle.com/datasets/marawanxmamdouh/dialogsum)) с помощью LoRA-дообучения небольшой decoder-only модели. Сравниваем:

1. **Бейзлайн без ML**: первое + последнее предложение диалога.
2. **Qwen2.5-0.5B-Instruct zero-shot**: инференс через **vLLM**.
3. **Qwen2.5-0.5B-Instruct + LoRA**: обучение через **unsloth + trl `SFTTrainer`** на 2 000 примеров, логи в **wandb**, инференс смёрженной модели через vLLM.

Качество меряем по **ROUGE-1/2/L и BLEU** (multi-reference, в test по 3 референса на диалог) и через **LLM-as-judge** (`openai/gpt-oss-120b` на Groq). Отдельно смотрим примеры глазами.

## Данные и выводы EDA

Ноутбук: [`notebooks/exploratory/eda.ipynb`](notebooks/exploratory/eda.ipynb).

| split | диалогов | референсов на диалог |
|---|---:|---:|
| train | 12 460 | 1 |
| validation | 500 | 1 |
| test | 500 | 3 |

- Диалоги короткие: ~130 слов, ~9–10 реплик, почти всегда 2 спикера (`#Person1#`, `#Person2#`).
- Саммари ~20 слов, это ~15–18% длины диалога. Задача сильно абстрактивная.
- ~85–90% референсов используют теги `#Person1#`/`#Person2#` и пишутся от третьего лица. Это особый стиль, который zero-shot модель не знает, а LoRA выучивает.
- В токенах Qwen: диалог p99 ≈ 540, саммари p99 ≈ 85. Отсюда `max_seq_length=1024`, `max_new_tokens=128`, `max_model_len=2048`.
- Референсы сильно расходятся между собой: leave-one-out ROUGE-L ≈ 50, ROUGE-2 ≈ 34. Это ориентир потолка, поэтому одних n-граммных метрик мало.

## Результаты

Test: 500 диалогов, 3 референса. GPU: Colab T4.

| система | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | длина, слов | скорость, примеров/с |
|---|---:|---:|---:|---:|---:|---:|
| lead: первое + последнее предложение | 28.22 | 7.11 | 23.96 | 11.65 | 15.9 | ~10⁴ (CPU) |
| Qwen2.5-0.5B-Instruct zero-shot (vLLM) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Qwen2.5-0.5B-Instruct + LoRA (vLLM) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

LLM-as-judge (`openai/gpt-oss-120b`, 150 случайных диалогов из test, шкала 1–5):

| система | faithfulness | coverage | conciseness | fluency | overall | win rate |
|---|---:|---:|---:|---:|---:|---:|
| _TBD_ | | | | | | |

Обучение LoRA: _TBD_ (время, train/eval loss, ссылка на wandb).

## Выбор модели и судьи

**Qwen2.5-0.5B-Instruct**:
- меньше 1B параметров (0.5B), поэтому обучение на T4 занимает минуты, а инференс в vLLM идёт сотнями примеров в секунду;
- instruct-версия: сразу понимает задачу из промпта, так что zero-shot бейзлайн осмысленный и сравнение «до/после LoRA» честное;
- первоклассная поддержка в vLLM и unsloth, chat template с явными маркерами `<|im_start|>assistant`, что удобно для лосса только по ответу;
- контекст 32k, хотя нам хватает 2k.

**Судья `openai/gpt-oss-120b` (Groq)**:
- самая сильная модель из доступных на бесплатном тарифе Groq: `llama-3.3-70b-versatile` на момент работы недоступна, а у `qwen3` и `llama-3.1-8b` есть минусы (см. ниже);
- другое семейство моделей, чем Qwen, поэтому меньше self-preference bias. Судью из семейства Qwen не брали сознательно;
- reasoning-модель, `reasoning_effort: low` даёт баланс качества и лимита токенов.

Как устроена оценка: судья видит диалог и **все три саммари в одном запросе** под метками A/B/C в **случайном порядке** (против позиционного bias и ради экономии лимитов). Каждое саммари оценивается по шкале 1–5 по критериям faithfulness, coverage, conciseness, fluency и overall, ответ приходит в JSON. Референс судье не показываем: оценка идёт относительно самого диалога. Ответы кешируются в `results/judge/judge_raw.jsonl`, поэтому прерванный запуск продолжается с места остановки.

## Выводы

_TBD после прогона на GPU._

### Какая модель лучшая по качеству и скорости?

_TBD_

### Как ещё можно улучшить результаты?

_TBD_

## Структура репозитория

```
configs/
  data.yaml              # seed, пути, размер train/val подвыборки
  infer_zeroshot.yaml    # vLLM: модель, dtype, max_model_len, sampling
  infer_lora.yaml        # то же для смёрженной LoRA-модели
  train_lora.yaml        # LoRA r/alpha/modules, гиперпараметры SFT, wandb
  judge.yaml             # Groq endpoint, модель-судья, rpm, n_samples
notebooks/
  exploratory/eda.ipynb             # EDA
  experiments/colab_pipeline.ipynb  # запуск всего GPU-пайплайна в Colab
scripts/
  prepare_data.py   # raw CSV -> data/processed/{train,val,test}.jsonl
  baseline_lead.py  # первое + последнее предложение
  infer.py          # генерация через vLLM (или HF для отладки на CPU)
  train_lora.py     # LoRA SFT: unsloth (GPU) или чистые trl+peft
  evaluate.py       # ROUGE/BLEU по всем предсказаниям -> таблица
  judge.py          # LLM-as-judge через Groq
src/dialogsum/
  data.py           # загрузка, группировка референсов, промпт
  metrics.py        # multi-reference ROUGE + sacreBLEU
  utils.py          # конфиги, seed, jsonl
results/            # таблицы метрик, оценки судьи, графики (коммитятся)
```

Данные (`data/`), веса и предсказания (`outputs/`) и `.env` в git не попадают.

## Как запустить

### Окружение

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # CPU: EDA, бейзлайн, метрики, судья
pip install -e .
# GPU (Linux/Colab): pip install unsloth vllm && pip install -r requirements-gpu.txt
cp .env.example .env               # вписать GROQ_API_KEY
```

Данные скачиваются автоматически через `kagglehub` при первом запуске `prepare_data.py`. Можно положить CSV вручную в `data/raw/`.

### Пайплайн

```bash
python scripts/prepare_data.py --config configs/data.yaml
python scripts/baseline_lead.py
python scripts/infer.py --config configs/infer_zeroshot.yaml          # GPU, vLLM
python scripts/train_lora.py --config configs/train_lora.yaml         # GPU, unsloth
python scripts/infer.py --config configs/infer_lora.yaml              # GPU, vLLM
python scripts/evaluate.py
python scripts/judge.py --config configs/judge.yaml
```

Отладка на CPU, на паре примеров и с бэкендами HF/peft вместо vLLM/unsloth:

```bash
python scripts/infer.py --backend hf --debug
python scripts/train_lora.py --backend peft --debug
python scripts/judge.py --debug
```

На GPU проще всего запустить [`notebooks/experiments/colab_pipeline.ipynb`](notebooks/experiments/colab_pipeline.ipynb) в Colab (T4): он клонирует репозиторий, ставит зависимости, вызывает те же скрипты и сохраняет всё на Google Drive.


## Run the PR branch in Colab

Open the [GPU notebook in Colab](https://colab.research.google.com/github/semenovfilipp/text-generation/blob/feature/lora-summarization/notebooks/experiments/colab_pipeline.ipynb).
No merge is needed: the notebook checks out `feature/lora-summarization`.

1. Select a T4 GPU and connect Google Drive.
2. Optionally add `WANDB_API_KEY` to Colab Secrets for W&B logging. Groq is evaluated locally after the GPU run; no `GROQ_API_KEY` or local `.env` is needed in Colab.
3. Execute cells sequentially. Stop at an error and resolve it before continuing.
4. Short debug inference and training runs precede the full runs. Debug predictions are excluded from the final metric table.
5. Predictions, metrics, weights and installed package versions persist in `MyDrive/text-generation/outputs`. The last cell saves and downloads `colab_results.zip` without weights; put it in the local project folder for Codex to run Groq judging and finish the analysis.
6. Codex will use the returned predictions for local Groq judging, review examples in `manual_review.json`, copy final metrics and the loss plot into the repository, and replace README `TBD` entries with measured conclusions.

Estimated Colab T4 times: setup 5-15 minutes, debug runs 2-5 minutes, main GPU stages 10-30 minutes, judge 8-20 minutes. Actual times depend on the environment and API quotas.
GPU dependency compatibility must be verified in Colab; local training and vLLM were not run. Compare speed on the same hardware with the same generation settings.
If predictions or judge settings change, move the old judge cache before restarting. The script rejects stale caches to avoid mixing experiments.
