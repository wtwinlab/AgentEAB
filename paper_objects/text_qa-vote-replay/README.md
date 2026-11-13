# Return rules on fixed QA answers

Labels: the QA part of `sec:qa-vote-replay` (Section VII-D) and of supplementary `sec:s-return` (S7-F).

## What is printed

On 112 QA tasks, each with five fixed sets of three initial responses and three update repeats
(1,680 sets of three post-update answers), the vote is replayed on the same candidates with a
readable-only plurality. The text prints the task and set counts, the 53 recovered abstentions,
that no success is overturned, and the accuracy gain of 3.15 pp with its 95% interval.

## Inputs

`inputs/qa_vote_replay_sets.csv` (3,360 rows): one set of three post-update answers (benchmark,
task, fixed initial responses `prefix_id`, repeat) under one peer-choice rule (`delivery`). The
1,680 rows with `native_ppl_high (released)` use the protocol's own rule (each agent receives the
one peer reply whose perplexity is above the median) and are the population the paper reports.
Columns `released_correct` and `released_abstained` give the original plurality, which counts
unreadable answers as one answer category and abstains if that category wins;
`readable_only_correct` gives the plurality after removing unreadable answers. Both break ties by
agent order. The file was derived once from the logged QA selection records, with both votes defined as
above.

## Command

```sh
python3 code/compute.py --out output
```

Writes `output/values.json`.

## Estimates

The gain is readable-only minus original accuracy over the 1,680 sets. Its interval resamples the
112 tasks 4,000 times with Python's `random.Random(20261013)`, each task weighted by its number of
sets, and takes order statistics 100 and 3,900 of the sorted resampled means.

## Check

`python3 ../check_objects.py --recompute --only text_qa-vote-replay` compares the listed values
with Section VII-D and the supplement. Both interval ends
("1.90--4.58") are checked; the counts printed as words (five, three, no) are under `not_checked`
and are computed as 5, 3 and 0.
