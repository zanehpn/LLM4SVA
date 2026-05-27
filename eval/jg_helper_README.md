# Cadence JasperGold docker helper contract

`eval/rescore_funcatk_with_cadence_pec.py` calls JasperGold inside a
docker container. It does **not** vendor the JG image (license-bound)
and does **not** fall back to the open SymbiYosys+Z3 PEC — failing
hosts exit with code 2.

This file documents the shape of the helper image so a site admin can
either build the image themselves or wire an existing internal JG
container against the same protocol.

## Image expectations

The image (default tag `cadence-jg:latest`, override with
`--jg-docker-image` or `JG_DOCKER_IMAGE`) must contain:

1. A licensed Cadence Jasper installation that ships `jg`.
2. A helper entrypoint script (default `/work/jg_prop_eq.sh`, override
   with `--jg-helper-script` or `JG_HELPER_SCRIPT`).
3. Whatever license server / `CDS_LIC_FILE` mount is needed at runtime
   (the rescorer passes `--jg-docker-args` through to `docker run`,
   e.g. `--jg-docker-args "-v /opt/cds.lic:/opt/cds.lic"`).

## Stdin/stdout protocol

The helper script is invoked by the rescorer like:

```
docker run --rm -i $JG_DOCKER_ARGS $JG_DOCKER_IMAGE $JG_HELPER_SCRIPT
```

It reads **one JSON object per line** on stdin:

```json
{"id": "<task_id>:<cand_idx>",
 "lm_sva": "assert property (...);",
 "ref_sva": "assert property (...);",
 "rtl": "module foo(...); ... endmodule",
 "depth": 15,
 "timeout": 60}
```

It writes **one JSON object per line** on stdout:

```json
{"id": "<same id>",
 "verdict": "EQUIVALENT" | "IMPLIES_REF_TO_LM" | "IMPLIES_LM_TO_REF"
            | "NOT_EQUIVALENT" | "UNSUPPORTED",
 "fwd_status": "PASS" | "FAIL" | "TIMEOUT" | "UNSUPPORTED",
 "bwd_status": "PASS" | "FAIL" | "TIMEOUT" | "UNSUPPORTED",
 "seconds": 0.92}
```

Notes:

- `verdict` must be one of the paper-facing five. Any internal JG
  failure (parse error, license drop, hang past `timeout` seconds)
  should be reported as `UNSUPPORTED` so the rescorer keeps the
  paper's 5-verdict surface.
- `fwd_status` / `bwd_status` are the two BMC directions used by
  `prop_eq_checker` (App. E Table 5). They are persisted for debugging
  but are not used to recompute pass@k.
- `seconds` is the wall-clock for the pair; used for cost reporting.
- The script is expected to release JG license seats between
  invocations. The rescorer chunks payloads so each `docker run` exits
  cleanly after a small batch.

## Reference TCL skeleton

A minimal Jasper TCL the helper can wrap (paper App. E `prop_eq_checker`
protocol):

```tcl
# /work/jg_prop_eq.tcl — invoked per (ref, lm) pair
clear -all
analyze -sv $::env(RTL_FILE) $::env(REF_FILE) $::env(LM_FILE)
elaborate -top sva_check

# Two-arm property equivalence — paper §4.3 / App. E Table 5.
assume -name ref_arm $::env(REF_PROP)
assert -name lm_arm  $::env(LM_PROP)
prove -bg -engine_mode {N} -time_limit ${::env(TIMEOUT)}s
report -task <fwd_task> -result -file $::env(FWD_RESULT)

clear -all
analyze -sv $::env(RTL_FILE) $::env(REF_FILE) $::env(LM_FILE)
elaborate -top sva_check
assume -name lm_arm  $::env(LM_PROP)
assert -name ref_arm $::env(REF_PROP)
prove -bg -engine_mode {N} -time_limit ${::env(TIMEOUT)}s
report -task <bwd_task> -result -file $::env(BWD_RESULT)
```

The wrapping shell script collates the two `report` outputs and emits
the JSON-per-line described above.
