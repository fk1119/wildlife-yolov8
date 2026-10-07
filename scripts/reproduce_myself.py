"""Start a fresh, isolated reproduction using the frozen eight-class dataset."""
import argparse,datetime,hashlib,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.yolo')
os.environ['YOLO_AUTOINSTALL']='false'
os.environ['MPLBACKEND']='Agg'
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check-only',action='store_true',help='Check readiness without starting training')
    parser.add_argument('--imgsz',type=int,choices=[416,640],default=416)
    parser.add_argument('--mosaic',type=float,choices=[0.0,1.0],default=1.0)
    args=parser.parse_args()
    import torch,yaml,ultralytics
    from ultralytics import YOLO
    cfg=ROOT/'wildlife8/dataset.yaml';initial=ROOT/'assets/yolov8n.pt'
    if not cfg.exists() or not initial.exists():raise RuntimeError('Run prepare8 first')
    expected='f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36'
    if hashlib.sha256(initial.read_bytes()).hexdigest()!=expected:raise RuntimeError('Initial checkpoint hash mismatch')
    settings=yaml.safe_load(cfg.read_text(encoding='utf8'))
    assert len(settings['names'])==8
    assert torch.cuda.is_available(),'CUDA GPU unavailable'
    for split in ['train','val','test']:
        assert (Path(settings['path'])/settings[split]).is_dir(),f'Missing {split} images'
    print('GPU:',torch.cuda.get_device_name(0),flush=True)
    print('PyTorch:',torch.__version__,'Ultralytics:',ultralytics.__version__,flush=True)
    print(f'Ready: 8 classes; original COCO initial weights; 30 epochs; imgsz={args.imgsz}; mosaic={args.mosaic}; batch=8; seed=42.',flush=True)
    if args.check_only:return
    target=ROOT/'wildlife8/my_runs'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    target.mkdir(parents=True,exist_ok=False)
    record={'started_at':datetime.datetime.now().isoformat(),'initial_weights':str(initial),'initial_sha256':expected,'dataset':str(cfg),'dataset_manifest_sha256':hashlib.sha256((ROOT/'wildlife8/manifest.json').read_bytes()).hexdigest(),'torch':torch.__version__,'ultralytics':ultralytics.__version__,'gpu':torch.cuda.get_device_name(0)}
    record['configuration']={'epochs':30,'imgsz':args.imgsz,'mosaic':args.mosaic,'batch':8,'seed':42}
    (target/'run_record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
    print('YOUR NEW EXPERIMENT:',target,flush=True)
    started=time.time()
    model=YOLO(str(initial))
    model.train(data=str(cfg),epochs=30,imgsz=args.imgsz,mosaic=args.mosaic,batch=8,device=0,workers=0,seed=42,deterministic=True,optimizer='AdamW',lr0=.001,weight_decay=.0005,amp=False,cache=False,patience=31,close_mosaic=0,project=str(target),name='train',plots=True,verbose=False)
    best=YOLO(str(target/'train/weights/best.pt'))
    metrics=best.val(data=str(cfg),split='test',imgsz=args.imgsz,batch=1,device=0,workers=0,project=str(target),name='test',plots=True)
    result={'metrics':metrics.results_dict,'per_class':metrics.summary(),'speed_ms':metrics.speed,'elapsed_seconds':time.time()-started,'classes':best.names}
    (target/'test_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=lambda v:v.item()),encoding='utf8')
    print('DONE. Your results:',target,flush=True)
    print(json.dumps(metrics.results_dict,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
