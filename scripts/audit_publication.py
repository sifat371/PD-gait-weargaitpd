#!/usr/bin/env python3
"""Validate publication source, aggregate tables and privacy exclusions (no clinical data needed)."""
import csv, hashlib, json, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for line in (ROOT/'analysis/reference/code_sha256.txt').read_text().splitlines():
    if not line.strip(): continue
    expected,relative=line.split(maxsplit=1)
    assert hashlib.sha256((ROOT/'analysis'/relative).read_bytes()).hexdigest()==expected,relative
ref=json.loads((ROOT/'analysis/reference/model_reference.json').read_text())
assert len(ref['pooled'])==5 and len(ref['richer_covariate_site_prediction'])==6
def csvrows(p):
    with (ROOT/p).open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))
pool=csvrows('results/analysis_outputs/pooled_baseline_decomposition.csv')
sites=csvrows('results/analysis_outputs/adjusted_site_signature_results.csv')
assert len(pool)==5 and len(sites)==6
for a,b in zip(pool,ref['pooled']):
    assert a['model']==b['model']
    assert abs(float(a['auc_mean'])-b['auc_mean'])<1e-10
    assert abs(float(a['auc_sd'])-b['auc_sd'])<1e-10
for a,b in zip(sites,ref['richer_covariate_site_prediction']):
    assert a['diagnosis_group'] + ': ' + a['model'] == b['model']
    assert abs(float(a['auc_mean'])-b['auc_mean'])<1e-10
pub=csvrows('results/manuscript/table_2_primary.csv')
assert len(pub)==7
for a,b in zip(pub[:5],pool):
    assert round(float(a['auc']),3)==round(float(b['auc_mean']),3)
assert [x['scheme'] for x in csvrows('results/manuscript/table_7_permutation.csv')]==['Global label permutation','Site-conditioned permutation']
restricted=('private-inputs/','/private-inputs/','/outputs/','/logs/','/metadata/','candidate_clinical_data.csv','preflight.json','participant_predictions')
paths=[x for x in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0') if x]
bad=[p for p in paths if any(x in p.lower() for x in restricted) or p.lower().endswith(('.npy','.npz','.pkl','.parquet','.zip','.pem','.key'))]
assert not bad,bad
expected_output=['pooled_baseline_decomposition.csv','adjusted_site_signature_results.csv','nested_cv_results.csv','jhu_to_va_transport.csv','va_participant_uncertainty.csv','site_signature_results.csv','site_diagnosis_composition.csv']
for name in expected_output:
    rows=csvrows('results/analysis_outputs/'+name)
    assert rows,name
    forbidden={'participant_id','subject_id','patient_id','participant','patient_name','name'}
    assert not (forbidden & {k.lower() for k in rows[0]}),name
print('PASS: manuscript tables, 11 model reference estimates, 7 aggregate output CSVs, source hashes and tracked-file privacy scan')
