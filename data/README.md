## On Referee's data

### Download

Data can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1BODA-JqYY-j_6SjAchzlcjA4FNPDDy_R?usp=sharing).

### Source data description
`source-sentences/realnews_100k` contains the source sentences extracted from RealNews. Each chunk has 10000 sentences,  and we filter out sentences shorter than 50 characters.

- `realnews_s1_chunk_{i}.txt` corresponds the chunk `i` of the source sentences, `realnews_s2_chunk_{i}.txt` are the immediate next sentences of each sentence in the corresponding `s1` file. `s0` are the previous sentences of each `s1` sentence, unused in the present work.

### Summaries description
* `chunk_1` is the evaluation set. Therefore, `summaries_chunk_1` were never used during training.
* `summaries_chunk_{i}.txt` correspond to the summaries of sentences found in `source-sentences/realnews_100k/realnews_s1_chunk_{i}.txt`
* `num_beam_{j}` reflects the outputs of the `j`-th beam and were used to compute the comments about increased range accuracy when including additional beams (although possibly at a cost in summary quality).

* To train Referee-Distil iteration 1, GPT3-Curie summaries from chunks 2,3,4,100,101,102,103,104,105,106 were used in training.

Warning: Directory naming may not reflect those used in the codebase!
