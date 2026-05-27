"""
Build a 100+ SVA dataset entirely from hand-crafted examples (no API calls).
This ensures the TCL pilot runs without OPENAI_API_KEY.

If OPENAI_API_KEY is available, run build_tcl_dataset.py instead for GPT-augmented data.

Output: data/sva_examples.json
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.tcl import classify_tcl

EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(EXPERIMENTS_DIR, "data")
OUT_FILE = os.path.join(DATA_DIR, "sva_examples.json")

# 20 per level × 5 levels = 100 examples
HAND_CRAFTED_ALL = [

    # =========== TCL Level 1: Combinational ===========
    ("assert property (@(posedge clk) valid && ready);", "L1: valid and ready simultaneously"),
    ("assert property (@(posedge clk) $rose(req));", "L1: req rising edge"),
    ("assert property (@(posedge clk) $fell(ack));", "L1: ack falling edge"),
    ("assert property (@(posedge clk) $stable(data));", "L1: data stable"),
    ("assert property (@(posedge clk) !(req && !gnt));", "L1: no req without gnt"),
    ("assert property (@(posedge clk) $past(valid) == valid);", "L1: valid unchanged"),
    ("assert property (@(posedge clk) a == b);", "L1: equality check"),
    ("assert property (@(posedge clk) $rose(req) || $fell(ack));", "L1: rose or fell"),
    ("assert property (@(posedge clk) en |-> ($stable(addr)));", "L1+L4: en implies stable addr"),
    ("assert property (@(posedge clk) (state == IDLE) || (state == BUSY));", "L1: state validity"),
    ("assert property (@(posedge clk) wr_en |-> (addr < 16'h8000));", "L1+L4: write in range"),
    ("assert property (@(posedge clk) !($rose(req) && $rose(ack)));", "L1: no simultaneous rises"),
    ("assert property (@(posedge clk) $fell(cs) |-> $stable(data));", "L1+L4: stable after cs falls"),
    ("assert property (@(posedge clk) cnt == $past(cnt) + 1);", "L1: cnt increments"),
    ("assert property (@(posedge clk) !overflow);", "L1: no overflow"),
    ("assert property (@(posedge clk) valid |-> (data != 8'hFF));", "L1+L4: data not max"),
    ("assert property (@(posedge clk) $rose(clk_en) |-> pll_lock);", "L1+L4: pll locked"),
    ("assert property (@(posedge clk) !(read && write));", "L1: no simultaneous read/write"),
    ("assert property (@(posedge clk) $stable(mode) || $rose(reset));", "L1: mode stable unless reset"),
    ("assert property (@(posedge clk) fifo_full |-> !wr_en);", "L1+L4: no write when full"),

    # =========== TCL Level 2: Single Fixed Delay ===========
    ("assert property (@(posedge clk) ##1 valid);", "L2: valid next cycle"),
    ("assert property (@(posedge clk) ##2 data_out == expected);", "L2: 2-cycle latency"),
    ("assert property (@(posedge clk) ##3 ack);", "L2: 3-cycle ack"),
    ("assert property (@(posedge clk) a [*3]);", "L2: a asserted 3 consecutive cycles"),
    ("assert property (@(posedge clk) a [=2]);", "L2: a asserted exactly 2 times nonconsecutive"),
    ("assert property (@(posedge clk) ##4 (out == in + 1));", "L2: 4-cycle pipeline output"),
    ("assert property (@(posedge clk) req ##1 gnt);", "L2: req then gnt next cycle"),
    ("assert property (@(posedge clk) a [->1]);", "L2: goto repetition once"),
    ("assert property (@(posedge clk) start ##5 done);", "L2: 5-cycle operation"),
    ("assert property (@(posedge clk) in_valid ##2 out_valid);", "L2: 2-stage pipeline"),
    ("assert property (@(posedge clk) ##1 !overflow);", "L2: no overflow next cycle"),
    ("assert property (@(posedge clk) push ##1 !empty);", "L2: fifo non-empty after push"),
    ("assert property (@(posedge clk) rd_en ##1 data_valid);", "L2: data valid 1 cycle after read"),
    ("assert property (@(posedge clk) (a ##1 b) ##1 c);", "L2: three-cycle sequence"),
    ("assert property (@(posedge clk) wr_en ##2 wr_done);", "L2: write completes in 2 cycles"),
    ("assert property (@(posedge clk) clk_en ##1 pll_stable);", "L2: pll stable 1 cycle later"),
    ("assert property (@(posedge clk) a [*5]);", "L2: a high for 5 cycles"),
    ("assert property (@(posedge clk) req ##1 (data != 0));", "L2: data nonzero after req"),
    ("assert property (@(posedge clk) ##6 result_valid);", "L2: result valid after 6 cycles"),
    ("assert property (@(posedge clk) (en && valid) ##1 output_ready);", "L2: output ready next cycle"),

    # =========== TCL Level 3: Variable/Ranged Delay ===========
    ("assert property (@(posedge clk) req ##[1:4] gnt);", "L3: gnt within 1-4 cycles"),
    ("assert property (@(posedge clk) valid ##[2:$] done);", "L3: done eventually after valid"),
    ("assert property (@(posedge clk) a [*2:8]);", "L3: a asserted 2-8 consecutive times"),
    ("assert property (@(posedge clk) a [=1:4]);", "L3: a asserted 1-4 times total"),
    ("assert property (@(posedge clk) req ##[0:3] ack);", "L3: ack within 0-3 cycles"),
    ("assert property (@(posedge clk) start ##[3:10] done);", "L3: done between 3-10 cycles"),
    ("assert property (@(posedge clk) irq ##[1:8] irq_ack);", "L3: irq ack within 1-8 cycles"),
    ("assert property (@(posedge clk) ##[2:5] result_valid);", "L3: result valid in 2-5 cycles"),
    ("assert property (@(posedge clk) en ##[1:$] rdy);", "L3: rdy eventually after en"),
    ("assert property (@(posedge clk) tx_start ##[4:16] tx_done);", "L3: tx in 4-16 cycles"),
    ("assert property (@(posedge clk) a [->1:3]);", "L3: goto 1-3 times"),
    ("assert property (@(posedge clk) rd_req ##[0:2] rd_valid);", "L3: read valid in 0-2 cycles"),
    ("assert property (@(posedge clk) miss ##[8:32] fill_done);", "L3: cache fill in 8-32 cycles"),
    ("assert property (@(posedge clk) cmd ##[1:5] resp);", "L3: response in 1-5 cycles"),
    ("assert property (@(posedge clk) wr ##[2:$] wr_ack);", "L3: write ack eventually"),
    ("assert property (@(posedge clk) a [*1:$]);", "L3: a asserted one or more times"),
    ("assert property (@(posedge clk) fetch ##[1:3] decode);", "L3: decode in 1-3 cycles"),
    ("assert property (@(posedge clk) lock ##[0:4] unlock);", "L3: unlock within 4 cycles"),
    ("assert property (@(posedge clk) flush ##[2:8] flush_done);", "L3: flush done in 2-8 cycles"),
    ("assert property (@(posedge clk) req ##[1:4] (ack || timeout));", "L3: ack or timeout in 1-4"),

    # =========== TCL Level 4: Sequence Operators ===========
    ("assert property (@(posedge clk) req |-> gnt);", "L4: req implies gnt"),
    ("assert property (@(posedge clk) start |=> ##2 done);", "L4: start implies done 2 later"),
    ("assert property (@(posedge clk) valid |-> (data != 0));", "L4: valid implies nonzero data"),
    ("assert property (@(posedge clk) req |-> ##[1:4] ack);", "L4: req implies ack in range"),
    ("assert property (@(posedge clk) en |=> output_valid);", "L4: en implies output_valid next"),
    ("assert property (@(posedge clk) (a ##1 b) |-> c);", "L4: sequence implies c"),
    ("assert property (@(posedge clk) first_match(req ##[0:5] gnt));", "L4: first match of req-gnt"),
    ("assert property (@(posedge clk) wr_en |-> !rd_en);", "L4: no simultaneous wr/rd"),
    ("assert property (@(posedge clk) (a ##1 b) throughout (c ##1 d));", "L4: throughout"),
    ("assert property (@(posedge clk) (req ##[1:3] gnt) within (frame ##[0:8] eof));", "L4: within"),
    ("assert property (@(posedge clk) (a ##1 b) intersect (c ##1 d));", "L4: intersect"),
    ("assert property (@(posedge clk) stall |-> (pc == $past(pc)));", "L4: pc frozen on stall"),
    ("assert property (@(posedge clk) irq |-> ##[1:8] irq_ack);", "L4: irq serviced"),
    ("assert property (@(posedge clk) miss |=> stall);", "L4: miss causes stall next cycle"),
    ("assert property (@(posedge clk) burst_en |-> (len > 0));", "L4: burst has positive length"),
    ("assert property (@(posedge clk) $rose(cs) |-> $stable(addr));", "L4: addr stable on cs rise"),
    ("assert property (@(posedge clk) (wr [*3]) |=> wr_done);", "L4: 3 writes then done"),
    ("assert property (@(posedge clk) full |-> !push);", "L4: no push when full"),
    ("assert property (@(posedge clk) (a ##1 b ##1 c) |-> d);", "L4: three-event sequence"),
    ("assert property (@(posedge clk) request |=> ##[0:3] grant);", "L4: grant within 3 cycles"),

    # =========== TCL Level 5: Liveness ===========
    ("assert property (@(posedge clk) req |-> s_eventually gnt);", "L5: req eventually gets gnt"),
    ("assert property (@(posedge clk) s_always valid);", "L5: valid always holds"),
    ("assert property (@(posedge clk) hungry s_until full);", "L5: hungry until full"),
    ("assert property (@(posedge clk) a until_with b);", "L5: a until b inclusive"),
    ("assert property (@(posedge clk) strong(req ##[1:$] gnt));", "L5: req eventually followed by gnt"),
    ("assert property (@(posedge clk) wr_en |-> s_eventually wr_done);", "L5: writes eventually complete"),
    ("assert property (@(posedge clk) s_always (req |-> s_eventually gnt));", "L5: always eventually fair"),
    ("assert property (@(posedge clk) stall s_until flush);", "L5: stall until flush"),
    ("assert property (@(posedge clk) tx_start |-> s_eventually tx_done);", "L5: tx eventually done"),
    ("assert property (@(posedge clk) err |-> s_eventually err_clear);", "L5: error eventually cleared"),
    ("assert property (@(posedge clk) strong(a [*1:$]));", "L5: a asserted at least once"),
    ("assert property (@(posedge clk) s_always (irq |-> s_eventually ack));", "L5: irq always serviced"),
    ("assert property (@(posedge clk) pending s_until complete);", "L5: pending until complete"),
    ("assert property (@(posedge clk) busy s_until idle);", "L5: busy until idle"),
    ("assert property (@(posedge clk) lock_req |-> s_eventually lock_grant);", "L5: lock eventually granted"),
    ("assert property (@(posedge clk) req |-> (req s_until ack));", "L5: req held until ack"),
    ("assert property (@(posedge clk) s_always (start |-> s_eventually done));", "L5: start-done liveness"),
    ("assert property (@(posedge clk) starvation s_until resource_avail);", "L5: starvation ends"),
    ("assert property (@(posedge clk) strong((a ##[1:$] b) ##[1:$] c));", "L5: strong sequence"),
    ("assert property (@(posedge clk) empty until_with push);", "L5: empty until_with push"),
]


def build_handcrafted_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    dataset = []
    expected_levels = [1]*20 + [2]*20 + [3]*20 + [4]*20 + [5]*20

    for i, ((sva, desc), exp_lv) in enumerate(zip(HAND_CRAFTED_ALL, expected_levels)):
        level, reason = classify_tcl(sva)
        dataset.append({
            "id": i,
            "sva": sva,
            "tcl_level": level,
            "tcl_reason": reason,
            "expected_level": exp_lv,
            "description": desc,
            "source": "hand",
        })

    with open(OUT_FILE, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved {len(dataset)} hand-crafted examples to {OUT_FILE}")

    from collections import Counter
    dist = Counter(d["tcl_level"] for d in dataset)
    exp_dist = Counter(d["expected_level"] for d in dataset)
    print("Classified distribution:", dict(sorted(dist.items())))
    print("Expected distribution:",   dict(sorted(exp_dist.items())))

    return dataset


if __name__ == "__main__":
    build_handcrafted_dataset()
