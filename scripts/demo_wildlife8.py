"""Run eight-class detection and export per-image counts (not unique animal counts)."""
import argparse,csv,os
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'wildlife8'
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.yolo');os.environ['YOLO_AUTOINSTALL']='false'
from ultralytics import YOLO
def main():
    p=argparse.ArgumentParser();p.add_argument('--source');p.add_argument('--conf',type=float,default=.25)
    p.add_argument('--weights',default=str(OUT/'runs/train/weights/best.pt'))
    p.add_argument('--iou',type=float,default=.7,help='NMS overlap threshold; smaller values suppress more overlapping boxes')
    p.add_argument('--imgsz',type=int,choices=[416,640],default=416);a=p.parse_args()
    if not (0 <= a.conf <= 1 and 0 <= a.iou <= 1):p.error('conf and iou must be between 0 and 1')
    model=YOLO(a.weights)
    source=Path(a.source) if a.source else OUT/'demo_images'
    files=sorted(source.glob('*')) if source.is_dir() else [source]
    files=[f for f in files if f.suffix.lower() in ['.jpg','.jpeg','.png','.bmp']]
    if not files:raise RuntimeError('No input images')
    rows=[];dest=None
    for result in model.predict(source=[str(f) for f in files],stream=True,imgsz=a.imgsz,batch=1,rect=True,conf=a.conf,iou=a.iou,device=0,save=True,project=str(OUT/'runs'),name='demo'):
        counts=Counter(int(c) for c in result.boxes.cls.cpu().tolist())
        rows.append({'image':Path(result.path).name,**{name:counts[i] for i,name in model.names.items()},'total':len(result.boxes)})
        dest=Path(result.save_dir)
    with (dest/'counts.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    print('Saved detection images and counts.csv:',dest)
if __name__=='__main__':main()
