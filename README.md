# Autoregressive next-word generation with recurrent neural networks

**ICT4403 Foundations of Generative AI — Assignment 2**  
Gouri Naik · Registration number 230953086

## Problem and objective

Train and compare Simple RNN, Bidirectional RNN, LSTM, and GRU systems for next-word prediction and sequential text generation. Measure held-out cross-entropy, perplexity, top-1 accuracy and CPU training time. Generate from identical seeds at temperatures 0.6, 1.0 and 1.4, then compare coherence, diversity and repetition.

## Dataset

[Tiny Shakespeare, from Andrej Karpathy's char-rnn repository](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt); the [raw source](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) is downloaded by the code. The included `data/tinyshakespeare.txt` is 1,115,394 characters (SHA-256 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`). To keep CPU training quick, the experiment uses its first 320,000 characters, with contiguous 70/15/15 percent character splits before tokenization. The train/validation/test portions contain 49,760 / 10,674 / 10,513 tokens. Text is lowercased and tokenized into words (including simple contractions) and `.,!?;:` punctuation. Speaker formatting, newlines and other symbols are discarded. The vocabulary is learned from training only; words occurring fewer than twice map to `<unk>`. There are 2,329 vocabulary entries, with 1,077 unknown tokens in validation and 1,336 in test.

Each supervised example contains the previous 16 tokens and the immediately following target token. Windows advance by four positions. Validation and test windows are constructed separately inside their own contiguous partitions, so targets cannot cross split boundaries. The bidirectional RNN processes only the 16-word *past* input window in both directions and predicts the word after that window. It therefore does not use future target words. This model has more parameters because it contains two directional recurrent streams.

## Models and comparable setup

All models share a 48-dimensional embedding, 64 hidden units per direction, a linear vocabulary head, Adam with learning rate 0.002, batch size 64, gradient norm clipping at 1, seed 42 and eight training epochs. The best validation-loss epoch selects each model for the held-out test and generation; all models still receive the same eight-epoch budget. The distinct model parameter counts are shown below. Training seconds measure the complete eight-epoch fit plus validation at each epoch on this workspace CPU; a Colab CPU will have different timings. No pretrained weights are used.

Run on Google Colab (CPU): open `FGAI_Assignment2.ipynb`, choose **Runtime → Run all**. The notebook has the complete implementation embedded and downloads the dataset automatically if it is absent. Colab includes PyTorch, NumPy and Matplotlib; if an import fails, run `!pip install -r requirements.txt` after uploading that file, or install `torch numpy matplotlib` directly. Download `results/` from the Colab Files sidebar when finished. The notebook starts with empty output cells so that a fresh run reports results for its actual runtime.

Alternative source-file run: `python experiment.py` from this directory. Dependencies are in `requirements.txt`. Output files are `results/metrics.json`, `results/validation_loss.png`, and `results/generated_samples.txt`.

## Held-out results from an actual CPU run

| Model | Parameters | Best epoch | Test loss ↓ | Perplexity ↓ | Top-1 accuracy ↑ | Training seconds ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Simple RNN | 270,473 | 4 | 5.210 | 183.06 | 14.9% | 8.6 |
| Bidirectional RNN | 426,825 | 2 | 5.271 | 194.57 | 13.0% | 20.0 |
| LSTM | 292,361 | 6 | **5.202** | **181.62** | **16.3%** | 11.2 |
| GRU | 285,065 | 5 | 5.203 | 181.75 | 15.6% | 17.2 |

Cross-entropy is the mean natural-log loss per next-word example and perplexity is `exp(loss)`. Accuracy is exact top-1 match, including unknown-token targets. The test has 2,625 stride-four windows. Timing reflects machine load and is illustrative, not a portable speed ranking. [Validation-loss plot](results/validation_loss.png) and [all exact metrics](results/metrics.json) are included. Several models begin to overfit after their best validation epoch, especially the bidirectional RNN.

## Sampling-temperature analysis

For each model, the same three seeds (`my lord`, `i pray you`, `the king is`) generate 30 additional tokens at temperatures 0.6, 1.0 and 1.4. Each model/seed/temperature draw resets a deterministic random seed. Generation repeatedly conditions on the most recent 16 output tokens. The `<unk>` token is blocked at sampling time. [All 36 complete outputs](results/generated_samples.txt) are included.

| Model | Mean unique-token ratio at 0.6 / 1.0 / 1.4 | Mean repeated-bigram ratio at 0.6 / 1.0 / 1.4 |
|---|---|---|
| Simple RNN | .711 / .867 / .956 | .046 / .000 / .000 |
| Bidirectional RNN | .611 / .844 / .967 | .126 / .000 / .000 |
| LSTM | .733 / .844 / .944 | .057 / .000 / .000 |
| GRU | .756 / .844 / .956 | .046 / .000 / .000 |

Ratios are averaged across the three 30-token continuations for each cell. Unique-token ratio is the share of distinct generated tokens; repeated-bigram ratio is the share of repeated adjacent token pairs among the 29 generated bigrams. These small samples are descriptive, not statistical proof. At 0.6, continuations favor frequent connective words and recognizable speaker-like phrases, but can circle back to repeated patterns. At 1.0, wording varies more, while syntax and scene continuity remain uneven. At 1.4, rare or abrupt choices increase lexical diversity and often weaken grammar and narrative continuity. In the LSTM sample from `my lord`, the low-temperature continuation repeats familiar phrases around “the people”; at higher temperature it introduces more unusual transitions. A speaker label or punctuation can make a sample look locally plausible without sustaining a coherent conversation. None of the models consistently maintains long-range context on this small training subset.

## Conclusion and limitations

**LSTM is the best overall next-word model here**, with the lowest held-out loss and perplexity and the highest accuracy. GRU is very close in loss (5.203 versus 5.202), so this single run does not establish a reliable general advantage. Moderate temperature 1.0 offers a practical compromise between repetitive low-temperature text and disjoint high-temperature text. The bidirectional RNN has more parameters and is valid for a fixed past context window, but its reversed reading within that window does not translate into better generalization here. The vocabulary omits rare words, the corpus subset is small, text formatting is simplified, and the results use one split and seed. More runs and a larger dataset would be needed for stronger claims.

## Repository contents

- `FGAI_Assignment2.ipynb`: self-contained Colab notebook.
- `experiment.py`: equivalent Python source.
- `data/tinyshakespeare.txt`: original public-source text (the run uses its first 320,000 characters).
- `results/`: exact metrics, plot and generated samples from the recorded CPU run.
- `requirements.txt`: direct dependencies.
- `Gouri_Naik_230953086_FGAI_Assignment2.docx` and `.pdf`: submission cover document, once the repository link is finalized.
