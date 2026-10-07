"""Read-only checks of class mapping, split integrity and completed training evidence."""
import argparse,csv,hashlib,json
from pathlib import Path
from project_config import DATA
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'wildlife8'
def main():
    p=argparse.ArgumentParser();p.add_argument('--data-only',action='store_true');args=p.parse_args()
    d=json.loads((OUT/'manifest.json').read_text(encoding='utf8'));records=d['records']
    assert len(d['classes'])==8
    assert len({r['sha256'] for r in records})==len(records),'Exact duplicate'
    for s in ['train','val','test']:
        subset=[r for r in records if r['split']==s]
        assert set(c for r in subset for c in r['classes'])==set(range(8)),f'Missing class in {s}'
        for r in subset:
            image=DATA/Path(r['file']).relative_to('data');assert hashlib.sha256(image.read_bytes()).hexdigest()==r['sha256']
            label=DATA/'labels'/s/image.with_suffix('.txt').name
            for line in label.read_text().splitlines():
                c,x,y,w,h=map(float,line.split());assert int(c)==c and 0<=c<8
                assert 0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1
            if s=='test' and r['source'].startswith('COCO'):assert r['source']=='COCO val2017'
    if not args.data_only:
        with (OUT/'runs/train/results.csv').open() as f:rows=list(csv.DictReader(f))
        assert [int(r['epoch']) for r in rows]==list(range(1,31))
        test=json.loads((OUT/'test_results.json').read_text(encoding='utf8'))
        assert len(test['per_class'])==8
        assert {r['Class'] for r in test['per_class']}==set(d['classes'])
        assert all(0<=r['mAP50-95']<=1 for r in test['per_class'])
        assert (OUT/'runs/train/weights/best.pt').stat().st_size>1000000
    print('PASS: all eight classes represented, image hashes and boxes valid, no exact cross-split duplicates.'+('' if args.data_only else ' Thirty epochs and eight-class test results verified.'))
if __name__=='__main__':main()
