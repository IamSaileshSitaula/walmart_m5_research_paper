# Pre-modeling checklist

| item | status | checks | notes |
|---|---|---|---|
| Raw files preserved, provenance + schema recorded | PASS (after logged repair) | P01, P02, P03, H01, H02, H03 | H03 |
| Key uniqueness (sales item-store, calendar d/date, price store-item-week) | PASS | S03, S04, C05, C06, R06, R07, R08 |  |
| Value validity (sales whole & nonneg, price positive & finite, SNAP 0/1, hierarchy, event pairing) | PASS | S06, S07, S08, S09, S10, R01, R03, R04, R05, C15_CA, C15_TX, C15_WI, S11, S12, S13, S14, S15, C16_1, C16_2, C17, C18, C19 |  |
| Exact calendar matching (contiguous d, exact d<->date, weekday/week rules, no unmatched sales days) | PASS | C07, C08, C09, C10, C11, C13, S01, J04 |  |
| Row counts and unit totals unchanged after each join (and in the written Parquet) | PASS | J01, J02, J03, J05, J06, J07 |  |
| Missingness classified (price status complete, raw price never filled, event blank vs join failure) | PASS | M01, M02, M03, M04 |  |
| Holiday anomaly review (flags written, no silent edits) | PASS (automated) - HUMAN REVIEW PENDING | A01, A02, A03, A04 |  |
| Feature availability at each forecast origin (no leakage) | FAIL | L01, L02, L03, L04, L05 | L01 |

Holiday anomaly flags require human review before modeling; nothing was auto-corrected.
