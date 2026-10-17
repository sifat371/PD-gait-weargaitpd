#!/usr/bin/env python3
"""Evaluate aggregate outcomes only; do not publish patient-level intermediate files."""
import argparse, json, math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXPECTED=json.loads((ROOT/'analysis/reference/model_reference.json').read_text())

def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    work=Path(a.workspace); src=work/'outputs/verification_results.json'
    actual=json.loads(src.read_text());report={'status':'PASS','comparison':[], 'cautions':[
        'Model comparison covers five pooled and six within-diagnosis models.',
        'Feature exclusion uses a cohort-wide (both-task) missingness screen before outer CV.',
        'A requested total-UPDRS covariate in PD is all missing and imputed to zero.',
        'Working directories may contain participant-level data and must remain private.'
    ]}
    for key in ('pooled','richer_covariate_site_prediction'):
        got=actual[key];expected=EXPECTED[key]
        if len(got)!=len(expected):report['status']='FAIL';continue
        for index,(g,e) in enumerate(zip(got,expected)):
            errors=[]
            for attr in ('auc_mean','auc_sd'):
                if not math.isclose(g[attr],e[attr],abs_tol=0.0005,rel_tol=0): errors.append(f'{attr} drift')
            if g['model']!=e['model'] or g['n']!=e['n']: errors.append('model/n mismatch')
            if g['selected_C_frequencies']!=e['selected_C_frequencies']: errors.append('C-frequencies mismatch')
            row={'family':key,'model':g['model'],'n':g['n'],'auc_mean':g['auc_mean'],'auc_sd':g['auc_sd'],'status':'PASS' if not errors else 'FAIL', 'issues':errors}
            if errors:report['status']='FAIL'
            report['comparison'].append(row)
    import csv
    for name in ('nested_cv_results.csv','jhu_to_va_transport.csv','label_permutation_sensitivity.csv','va_participant_uncertainty.csv','revision2_summary.json'):
        path=work/'outputs'/name
        report.setdefault('additional_analysis_outputs',{})[name]=path.exists()
        if not path.exists():report['status']='FAIL'
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print('Final reference check:',report['status'],len(report['comparison']),'verified models')
    print('Scope: five pooled and six within-diagnosis classification models.')
    if report['status']!='PASS':raise SystemExit(2)
if __name__=='__main__':main()
