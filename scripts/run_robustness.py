"""Single-corruption augmentation study. No new architecture; official YOLO training/AP."""
import argparse,datetime,hashlib,json,os,random,shutil,subprocess,sys,time
from live_process import run_logged
from pathlib import Path
from project_config import CONFIG, DATA, INITIAL, OUTPUT, BASELINE, DEVICE, WORKERS, TRAINING, dataset_config, snapshot
ROOT=Path(__file__).resolve().parents[1];OUT=OUTPUT/'robustness'
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.yolo');os.environ['YOLO_AUTOINSTALL']='false';os.environ['MPLBACKEND']='Agg'
import cv2,numpy as np,yaml
NAMES=['buffalo','elephant','rhino','zebra','bear','giraffe','lion','deer']
KINDS=['dark','blur','jpeg']
CONDITIONS=['clean']+[f'{k}{s}' for k in KINDS for s in [1,2,3]]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda x:x.item()),encoding='utf8')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def now():return datetime.datetime.now().isoformat()
def decode(p):
    im=cv2.imdecode(np.fromfile(p,dtype=np.uint8),cv2.IMREAD_COLOR)
    if im is None:raise RuntimeError('Invalid image '+str(p))
    return im
def corrupt(im,kind,severity):
    if kind=='clean':return im.copy()
    if kind=='dark':return np.clip(im.astype(np.float32)*[.6,.3,.12][severity-1],0,255).astype(np.uint8)
    if kind=='blur':
        sigma=min(im.shape[:2])*[.0015,.004,.008][severity-1]
        return cv2.GaussianBlur(im,(0,0),sigmaX=sigma,sigmaY=sigma,borderType=cv2.BORDER_REFLECT_101)
    if kind=='jpeg':
        ok,buf=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,[60,25,8][severity-1]])
        assert ok;return cv2.imdecode(buf,cv2.IMREAD_COLOR)
    raise ValueError(kind)
def save_png(p,im):
    p.parent.mkdir(parents=True,exist_ok=True);ok,buf=cv2.imencode('.png',im,[cv2.IMWRITE_PNG_COMPRESSION,1]);assert ok;buf.tofile(p)
def make_eval(split,condition):
    dest=OUT/'eval_data'/condition;done=dest/f'{split}_manifest.json'
    if done.exists():return dest/'dataset.yaml'
    records=[]
    for src in sorted((DATA/'images'/split).glob('*')):
        im=decode(src);kind='clean' if condition=='clean' else condition[:-1];sev=0 if kind=='clean' else int(condition[-1])
        transformed=corrupt(im,kind,sev);assert transformed.shape==im.shape
        p=dest/'images'/split/src.with_suffix('.png').name;save_png(p,transformed)
        label=DATA/'labels'/split/src.with_suffix('.txt').name;lp=dest/'labels'/split/label.name;lp.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(label,lp)
        assert lp.read_bytes()==label.read_bytes()
        if condition=='clean':assert np.array_equal(decode(p),im)
        records.append({'source':str(src.relative_to(DATA)),'source_sha256':sha(src),'file':str(p.relative_to(OUT)),'sha256':sha(p),'label_sha256':sha(lp)})
    (dest/'dataset.yaml').write_text(yaml.safe_dump({'path':str(dest),'train':str(DATA/'images/train'),'val':'images/val','test':'images/test','names':dict(enumerate(NAMES))}),encoding='utf8')
    write(done,records);print('PREPARED',split,condition,len(records),flush=True);return dest/'dataset.yaml'
def evaluate(weights,split,condition,tag):
    target=OUT/'metrics'/tag/split/f'{condition}.json'
    if target.exists():
        result=read(target);assert result['weights_sha256']==sha(weights);return result
    cfg=make_eval(split,condition)
    from ultralytics import YOLO
    m=YOLO(str(weights)).val(data=str(cfg),split=split,imgsz=CONFIG['experiments']['robustness']['imgsz'],batch=1,device=DEVICE,workers=WORKERS,conf=.001,iou=.7,augment=False,plots=False,save_json=False,project=str(OUT/'evaluations'/tag/split),name=condition,verbose=False)
    result={'tag':tag,'split':split,'condition':condition,'evaluated_at':now(),'weights':str(weights),'weights_sha256':sha(weights),'evaluation_manifest_sha256':sha(cfg.parent/f'{split}_manifest.json'),'metrics':m.results_dict,'per_class':m.summary(),'speed_ms':m.speed,'parameters':{'imgsz':416,'batch':1,'conf':.001,'nms_iou':.7}}
    write(target,result);return result
def plan():
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'protocol.json'
    if p.exists():return
    snapshot(OUT)
    write(p,{'created_at':now(),'question':'Does extra training augmentation for one corruption transfer to other, non-targeted corruptions, and at what clean-image cost?',
      'conditions':CONDITIONS,'dark_multipliers':[.6,.3,.12],'blur_sigma_short_edge_fraction':[.0015,.004,.008],'jpeg_quality':[60,25,8],
      'selection':'Using seed42 baseline validation only: choose type with lowest mean mAP50-95 across its three preset severity levels; ties use KINDS order dark,blur,jpeg. Levels are not perceptually calibrated across types.',
      'training':'Reuse seed42 original baseline; train augmented seed42, original baseline seed43, augmented seed43. 30 epochs each, same COCO initialization/optimizer/batch/resolution/default augmentations. No test-driven retraining.',
      'extra_augmentation':'Offline fixed set: deterministic shuffled half (floor N/2) of train images replaced by corrupted counterparts. Severity 1/2/3 cycles across shuffled selected images. Other half stays clean; sample count unchanged. All augmented-arm files lossless PNG with original dimensions. The same augmentation assignment is used for both training seeds.',
      'augmentation_assignment_seed':20261004,'seeds':[42,43],'checkpoint_selection':'Best clean original validation fitness, same rule for both arms; no corrupted-test checkpoint selection.',
      'limitations':['Two seeds describe sensitivity, not statistical significance.','Prior original test has been inspected in previous work; this is not a fresh blind holdout.','Synthetic corruption is not evidence of real night/weather performance.','Existing HSV/Mosaic etc remain; non-targeted types are not guaranteed unseen in all training augmentations.','Offline fixed augmentation is simpler than online stochastic augmentation and results are specific to this policy.'],
      'initial_sha256':sha(INITIAL),'manifest_sha256':sha(ROOT/'wildlife8/manifest.json')})
def select():
    plan();p=OUT/'selected_corruption.json'
    if p.exists():return read(p)['kind']
    weights=BASELINE/'weights/best.pt'
    scores={c:evaluate(weights,'val',c,'baseline42')['metrics']['metrics/mAP50-95(B)'] for c in CONDITIONS}
    means={k:sum(scores[f'{k}{s}'] for s in [1,2,3])/3 for k in KINDS}
    kind=min(KINDS,key=lambda k:means[k]);write(p,{'selected_at':now(),'kind':kind,'validation_scores':scores,'mean_by_type':means,'rule':'minimum mean corrupted validation mAP50-95, fixed severity grids','baseline_weights_sha256':sha(weights)})
    print('SELECTED',kind,means,flush=True);return kind
def make_training(kind):
    dest=OUT/'augmented_data';done=dest/'manifest.json'
    if done.exists():assert read(done)['kind']==kind;return dest/'dataset.yaml'
    files=sorted((DATA/'images/train').glob('*'));order=list(range(len(files)));random.Random(20261004).shuffle(order)
    assignment={idx:1+j%3 for j,idx in enumerate(order[:len(files)//2])};records=[]
    for i,src in enumerate(files):
        im=decode(src);sev=assignment.get(i,0);transformed=corrupt(im,kind,sev) if sev else im
        p=dest/'images/train'/src.with_suffix('.png').name;save_png(p,transformed)
        source_label=DATA/'labels/train'/src.with_suffix('.txt').name;lp=dest/'labels/train'/source_label.name;lp.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source_label,lp)
        assert transformed.shape==im.shape and lp.read_bytes()==source_label.read_bytes()
        if not sev:assert np.array_equal(decode(p),im)
        records.append({'source':str(src.relative_to(DATA)),'source_sha256':sha(src),'file':str(p.relative_to(OUT)),'sha256':sha(p),'severity':sev,'label_sha256':sha(lp)})
    cfg={'path':str(dest),'train':'images/train','val':str(DATA/'images/val'),'test':str(DATA/'images/test'),'names':dict(enumerate(NAMES))}
    (dest/'dataset.yaml').write_text(yaml.safe_dump(cfg),encoding='utf8');write(done,{'kind':kind,'records':records,'changed_images':len(assignment),'total':len(files)});return dest/'dataset.yaml'
def train(tag):
    import torch,ultralytics
    from ultralytics import YOLO
    kind=read(OUT/'selected_corruption.json')['kind'];seed=int(tag[-2:]);aug=tag.startswith('augmented')
    target=OUT/'training'/tag
    if (target/'record.json').exists() and read(target/'record.json')['state']=='complete':return
    if target.exists():raise RuntimeError('Incomplete run exists; inspect rather than overwrite: '+str(target))
    target.mkdir(parents=True)
    cfg=make_training(kind) if aug else dataset_config()
    initial=INITIAL;assert sha(initial)==read(OUT/'protocol.json')['initial_sha256']
    settings=dict(data=str(cfg),epochs=CONFIG['experiments']['robustness']['epochs'],imgsz=CONFIG['experiments']['robustness']['imgsz'],mosaic=1.0,batch=CONFIG['experiments']['robustness']['batch'],device=DEVICE,workers=WORKERS,seed=seed,deterministic=True,optimizer='AdamW',lr0=.001,weight_decay=.0005,amp=False,cache=False,patience=31,close_mosaic=0,project=str(target),name='train',plots=True,verbose=False)
    record={'state':'training','started_at':now(),'tag':tag,'settings':settings,'initial_sha256':sha(initial),'torch':torch.__version__,'ultralytics':ultralytics.__version__,'opencv':cv2.__version__,'gpu':torch.cuda.get_device_name(DEVICE),'selected_corruption_sha256':sha(OUT/'selected_corruption.json'),'dataset_manifest_sha256':sha(OUT/'augmented_data/manifest.json') if aug else sha(ROOT/'wildlife8/manifest.json')}
    write(target/'record.json',record);started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    try:
        YOLO(str(initial)).train(**settings)
        csvlines=(target/'train/results.csv').read_text().splitlines();assert len(csvlines)==31
        record.update(state='complete',ended_at=now(),training_seconds=time.perf_counter()-started,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,best_sha256=sha(target/'train/weights/best.pt'))
        write(target/'record.json',record)
    except Exception as e:
        record.update(state='failed',error=repr(e));write(target/'record.json',record);raise
def pipeline():
    plan();select();make_training(read(OUT/'selected_corruption.json')['kind'])
    for tag in ['augmented42','baseline43','augmented43']:
        rec=OUT/'training'/tag/'record.json'
        if rec.exists() and read(rec)['state']=='complete':continue
        print('START TRAIN',tag,now(),flush=True)
        print('Live training output; log:', OUT/f'{tag}.log', flush=True)
        run_logged([sys.executable,'-u','-X','utf8',str(Path(__file__).resolve()),
                    '--train',tag,'--output',str(OUT)], OUT/f'{tag}.log', cwd=ROOT)
        print('FINISHED TRAIN',tag,now(),flush=True)
    # Test inspection only after both paired training sets complete.
    for tag in ['baseline42','augmented42','baseline43','augmented43']:
        weights=BASELINE/'weights/best.pt' if tag=='baseline42' else OUT/'training'/tag/'train/weights/best.pt'
        for condition in CONDITIONS:evaluate(weights,'test',condition,tag)
    write(OUT/'completion.json',{'state':'complete','ended_at':now(),'test_evaluations':40,'validation_selection_evaluations':10,'training_runs_added':3})
    print('COMPLETE robustness experiment',flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--train',choices=['augmented42','baseline43','augmented43']);a.add_argument('--output',default=str(OUT),help='Use a new directory to independently rerun this study');args=a.parse_args()
    OUT=Path(args.output).resolve()
    if not OUT.is_relative_to(OUTPUT):raise ValueError('Keep study outputs inside the project directory')
    if args.train:train(args.train)
    else:pipeline()
