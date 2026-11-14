"""Report cap and end-token evidence without inferring safety labels."""
import argparse,json
from collections import Counter
from finsafe.data import read_jsonl
from finsafe.progress import atomic_json

def inspect(rows):
    measured=[r for r in rows if 'first_tokenizer_eos_position' in r and 'last_token_id' in r]
    caps=[r for r in rows if isinstance(r.get('max_new_tokens'),int)]
    interior=[r['id'] for r in measured if r['first_tokenizer_eos_position'] is not None
              and r['first_tokenizer_eos_position']<r['output_tokens']-1]
    return {'n':len(rows),'token_diagnostic_records':len(measured),
        'token_diagnostic_missing':len(rows)-len(measured),
        'cap_metadata_records':len(caps),'cap_hits':sum(r['output_tokens']>=r['max_new_tokens'] for r in caps),
        'interior_tokenizer_eos_ids':interior,
        'last_token_counts':dict(Counter(str(r['last_token_id']) for r in measured)),
        'notes':['An output cap hit is not proof of truncation.',
                 'Missing token diagnostics are unknown, not zero end-token problems.',
                 'An interior tokenizer EOS warrants comparing actual token definitions and configured stopping IDs.',
                 'This report supplies generation diagnostics only; it does not label safety.']}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    report=inspect(read_jsonl(a.predictions));atomic_json(a.out,report);print(json.dumps(report,indent=2))
