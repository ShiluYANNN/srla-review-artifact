# Tail comparison subset

Join generations.jsonl to questions.jsonl on id. Each generation row identifies one question and zero-based sampling round. Several question IDs can share identical question text or identical Tail input. Text counts below use exact string equality before chat templating.

| Language | Generation keys | Question IDs | Distinct base-question texts (including options) | Distinct Tail texts in the four-model manifest |
|---|---:|---:|---:|---:|
| Chinese | 67 | 40 | 37 | 39 |
| English | 75 | 64 | 60 | 61 |
| Total | 142 | 104 | 97 | 100 |

74 IDs contribute one generation each, 22 contribute two, and 8 contribute three. The follow-up reuses 60 scored generations (41 IDs) and adds 82 generations on reconstructed inputs (63 IDs). For the additions, the same-question unit is repeated four times. Seventeen units already occurred four times consecutively in the source; 46 were completed by copying. The reconstruction fields distinguish observed from added copies. The two cross-ID groups with identical follow-up Tail text are en-2-68 / en-2-98 / en-3-108 / en-5-221 and zh-4-5 / zh-4-11.

The cohort was calibrated using the Qwen DPO-to-GRPO historical Tail outcomes: all 86 positive records and 56 selected negative records out of 960. This conditions that row's RDR on the selection procedure. Qwen DPO and DPO-to-GRPO use historical Tail inputs (104 distinct texts); the four follow-up targets use the Tail manifest. Historical English inputs may include the question and options. Each Tail outcome is coded as 1 for RDR and 0 otherwise. RDR requires exhausting the budget and repeating a 6-gram at least four times in the last quarter. The historical Qwen rows use the 1024-token window. Llama Instruct and Llama DPO use 1024; the two distilled models use native open-thinking prompts and 4096. New generations use T=0.7 and top-p=0.9. tail_manifest_seed identifies the assigned follow-up seed and does not document the RNG seed of independently archived historical runs. llama_dpo_full_rdr is the separate full-input comparator scored on the same 142 question/round keys.
