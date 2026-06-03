# Soridormi scripted social readiness report

Overall: **NEEDS WORK**
Require live MuJoCo acceptance: `true`
Candidates ready for `available_sim`: **6/7**

| Skill | Status | Dry run | Live | Recommendation | Blockers |
| --- | --- | --- | --- | --- | --- |
| bow | `available_sim_experimental` | PASS | PASS | `candidate_for_available_sim` | — |
| express_attention | `available_sim_experimental` | PASS | PASS | `candidate_for_available_sim` | — |
| look_at_person | `available_sim_experimental` | PASS | PASS | `candidate_for_available_sim` | — |
| look_direction | `available_sim_experimental` | PASS | PASS | `candidate_for_available_sim` | — |
| neutral_head | `available_sim_experimental` | PASS | PASS | `candidate_for_available_sim` | — |
| nod_yes | `available_sim_experimental` | PASS | FAIL | `keep_available_sim_experimental` | live MuJoCo acceptance failed<br>live: nod_yes commanded non-moving axis head_yaw outside tolerance: min=0, max=0.293 |
| shake_no | `available_sim_experimental` | PASS | PASS | `candidate_for_available_sim` | — |

Promotion rule: do not edit a skill from `available_sim_experimental` to `available_sim` unless dry-run acceptance passes and a live MuJoCo acceptance report passes without falls.
