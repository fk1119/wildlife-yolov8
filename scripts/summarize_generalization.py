"""Build a single readable report and audit tables from actual cached predictions."""
import csv,html,json
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'wildlife8/generalization'
NAMES=['buffalo','elephant','rhino','zebra','bear','giraffe','lion','deer']
def load(name):return json.loads((OUT/name).read_text(encoding='utf8'))
def csvwrite(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    data=load('evaluation.json');grid=load('validation_grid.json');review=load('annotation_review.json');manifest=load('external_manifest.json');selected=load('selected_setting.json')
    rows=[]
    for split,settings in data.items():
        for name,r in settings.items():rows.append({'split':split,'setting':name,**{k:v for k,v in r.items() if k not in ['per_image','per_class']}})
    csvwrite('metrics.csv',rows)
    csvwrite('validation_grid.csv',[{k:v for k,v in r.items() if k not in ['per_image','per_class']} for r in grid])
    accepted=[r for r in manifest['records'] if Path(r['file']).name not in review['excluded_by_filename']]
    perclass=[]
    for cid,name in enumerate(NAMES):
        n=sum(any(b[0]==cid for b in r['boxes']) for r in accepted)
        perclass.append({'class':name,'images':n,**data['external']['baseline']['per_class'][str(cid)],'selected_tp':data['external']['validation_selected']['per_class'][str(cid)]['tp'],'selected_fp':data['external']['validation_selected']['per_class'][str(cid)]['fp'],'selected_fn':data['external']['validation_selected']['per_class'][str(cid)]['fn']})
    csvwrite('external_per_class.csv',perclass)
    fig,axs=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for ax,split in zip(axs,['test','external']):
        for metric,label in [('precision','Precision'),('recall','Recall'),('f1','F1')]:
            ax.plot(range(3),[data[split][n][metric]*100 for n in ['baseline','user_conf05','validation_selected']],marker='o',label=label)
        ax.set_xticks(range(3),['conf .25\nNMS .7','conf .5\nNMS .7','conf .5\nNMS .5']);ax.set_ylim(70,100);ax.grid(alpha=.25);ax.set_ylabel('Fixed-threshold micro score (%)');ax.set_title(f'{split}: '+str(data[split]['baseline']['images'])+' images');ax.legend()
    fig.savefig(OUT/'threshold_tradeoff.png',dpi=160);plt.close(fig)
    # Render all accepted external cases from cached predictions, not selected successes.
    gallery=OUT/'external_review';gallery.mkdir(exist_ok=True)
    predictions={n:{r['file']:r for r in load(f'external_nms{i}.json')} for n,i in [('baseline',.7),('selected',selected['nms_iou'])]}
    def draw(r,mode):
        im=Image.open(OUT/r['file']).convert('RGB');im.thumbnail((480,360));d=ImageDraw.Draw(im)
        if mode=='gt':boxes=[(c,1,x1,y1,x2,y2) for c,x1,y1,x2,y2 in r['boxes']]
        else:
            conf=.25 if mode=='baseline' else selected['conf']
            boxes=[p for p in predictions[mode][Path(r['file']).name]['pred'] if p[1]>=conf]
        occupied=[]
        for c,s,x1,y1,x2,y2 in boxes:
            coords=(x1*im.width,y1*im.height,x2*im.width,y2*im.height)
            d.rectangle(coords,outline='lime' if mode=='gt' else 'red',width=2)
            tx,ty=coords[:2]
            while any(abs(tx-x)<140 and abs(ty-y)<13 for x,y in occupied):ty+=14
            occupied.append((tx,ty))
            d.text((tx,ty),NAMES[int(c)]+('' if mode=='gt' else f' {s:.2f}'),fill='yellow',stroke_width=1,stroke_fill='black')
        return im
    cards=[]
    for j,r in enumerate(accepted):
        canvas=Image.new('RGB',(1440,395),'white');d=ImageDraw.Draw(canvas)
        for k,mode in enumerate(['gt','baseline','selected']):
            canvas.paste(draw(r,mode),(480*k,30));d.text((480*k+5,8),mode,fill='black')
        dest=gallery/f'{j:02d}_{Path(r["file"]).stem}.jpg';canvas.save(dest)
        cards.append(f'<section><h3>{html.escape(Path(r["file"]).name)}</h3><img loading="lazy" src="external_review/{dest.name}"></section>')
    (OUT/'图片逐张对比.html').write_text('<!doctype html><meta charset="utf-8"><title>新增图片逐张对比</title><style>body{font-family:system-ui;margin:24px;background:#eee}section{background:white;padding:12px;margin-bottom:16px}img{width:100%;max-width:1440px}</style><h1>37张新增图片：标注 / 原设置 / 验证集选定设置</h1><p>全部保留样本均展示。绿色为官方标注，红色为模型预测。候选筛查是AI视觉复核，仍需人工复查。</p>'+''.join(cards),encoding='utf8')
    lines=['# 新图片重复检测：阈值取舍与新增图片评估','',
      '本次实验已实际运行，使用你在2026年10月2日训练的best.pt，没有重新训练、没有改模型权重，也没有修改原训练/验证/测试标签。','',
      '## 结论先说','',
      '提高置信度阈值能减少额外预测和重复框，但会损失部分检出目标。不能把“框更干净”写成“模型全面变强”。在本次新增小样本中，进一步降低NMS阈值没有带来额外收益。','',
      '## 实验如何防止只为一张图片调参','',
      '1. 预先保存protocol.json，固定置信度0.1/0.25/0.5/0.75与NMS IoU 0.3/0.5/0.7，共12种设置。',
      '2. 只在原383张验证图片上按micro F1选择设置；随后保存selected_setting.json。选中conf=0.5、NMS IoU=0.5，验证F1为89.48%。',
      '3. 设置固定后，评估原360张测试图片及新增复核后的37张图片；没有根据这些结果再次调参。',
      '4. 用户已看过的犀牛和两张狮子图片仅作案例，不进入新增评估集。','',
      '## 指标定义','',
      '按置信度从高到低，同类别、一对一、框IoU≥0.5匹配。Precision=TP/(TP+FP)，Recall=TP/(TP+FN)，F1为两者调和平均；这里都是汇总目标后的micro指标。FP是未匹配预测，FN是未匹配标注，可能涉及错类、定位、重复或标注缺陷，不能全部解释成纯粹误认/完全没看见。',
      '**这不是mAP，也不是原Ultralytics test_results.json里按其评估流程汇总的P/R；数值口径不同，不覆盖旧指标。** NMS IoU是去重参数，匹配IoU是评估规则，两者作用不同。降低NMS IoU也可能压掉两只真实重叠动物。','',
      '## 完整对照结果','',
      '| 图片集 | 设置conf/NMS | 标注框 | TP | FP | FN | P | R | F1 |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {'原测试集' if row['split']=='test' else '新增37张'} | {row['conf']}/{row['nms_iou']} | {row['gt']} | {row['tp']} | {row['fp']} | {row['fn']} | {row['precision']:.2%} | {row['recall']:.2%} | {row['f1']:.2%} |")
    lines += ['', '原测试集：仅提高conf时，FP从127降到48，FN从68增到86；再降低NMS时，FP降到41，FN增到88。新增图片：FP从10降到4，FN从8增到10。新增图片上两种conf=0.5设置的汇总计数相同。', '',
      '![阈值取舍](threshold_tradeoff.png)','',
      '重复框代理指标：同类别未匹配预测与已匹配标注IoU≥0.5。原测试集25→6→0，新增图片2→0→0。这个规则可能漏计局部重复框或混淆拥挤目标，不能说所有重复检测已被彻底解决。','',
      '## 新增图片从哪里来、覆盖到什么范围','',
      '从已有Open Images官方标注池中，按固定随机种子筛选此前课程数据未使用的图片ID；与原2496张图片进行SHA256和64位感知哈希筛查，距离≤6不纳入。也检查新增样本之间的感知近似。保留图片与原数据的哈希筛查不证明绝无近重复或预训练重合。',
      '先冻结48个候选，再在看预测之前做AI视觉筛查，排除11张明显错类、物种不确定或展示物图片，最终37张、58个标注框。沿用官方框，未把模型预测当标签。排除清单及理由见annotation_review.json；所有候选原图仍保留可审计。','',
      '| 类别 | 保留图片 | 标注框 | 原设置TP/FP/FN | 选定设置TP/FP/FN |','|---|---:|---:|---|---|']
    for r in perclass:lines.append(f"| {r['class']} | {r['images']} | {r['gt']} | {r['tp']}/{r['fp']}/{r['fn']} | {r['selected_tp']}/{r['selected_fp']}/{r['selected_fn']} |")
    lines += ['', '**覆盖限制：只有六类，没有水牛和狮子；鹿仅2张。这个实验是小规模新增样本评估，不是完整八类外部基准。** 原数据部分也来自Open Images，新增样本不等于全新来源或彻底分布外数据。类别构成不同，不能直接比较两套图片集总分来证明存在或不存在泛化下降。',
      '复核发现了标注为斑马的霍加狓、标注为熊的小熊猫，以及若干鹿类映射问题。这个发现提示原数据也需要独立审计，但不能据此断言原训练集中错误的比例；本轮没有修改旧数据、重训或重算旧mAP。',
      '复核由AI进行：查看48张候选的标注缩略图，并对4张疑难图额外查看原图；不冒充人工专家逐框标注。图片遮挡、背景目标和定位边界仍可能存在残余问题。Open Images的官方评估还涉及非穷尽标签与类别层级，本项目的统一匹配是课程诊断规则，不是官方榜单协议。','',
      '## 你的犀牛图片实际发生了什么','',
      '实际重跑得到4个犀牛框，置信度为0.9515、0.8979、0.3887、0.2726；conf=0.5后保留前两个。两张狮子图片在三种设置下分别保留2个和1个框。框数符合目视数量不代表每个框的定位都严格正确。','',
      '![原设置](cases/baseline/rhino1-1024x683.jpg)','![conf0.5](cases/user_conf05/rhino1-1024x683.jpg)','',
      '## 必须同时展示的代价案例','',
      '新增图片oi_test_72d914acac313ab8.jpg中，浸在水里的大象原设置可以匹配；提高conf后两个低分预测都被过滤，变成漏检。oi_validation_4975a418f17c0e13.jpg中，左侧受岩石遮挡的大象原得分约0.40，提高conf后也被过滤。这里的“更干净”并不意味着更完整。','',
      '![水中大象被过滤：左标注、中原设置、右选定设置](external_review/02_oi_test_72d914acac313ab8.jpg)',
      '![遮挡大象被过滤](external_review/03_oi_validation_4975a418f17c0e13.jpg)','',
      '## 自己运行','',
      '在项目根目录的PowerShell中运行：','', '```powershell',
      "$projectPython = '.\\.venv\\Scripts\\python.exe'",
      '& $projectPython -X utf8 scripts/evaluate_generalization.py --stage validation',
      '& $projectPython -X utf8 scripts/evaluate_generalization.py --stage evaluation',
      '& $projectPython -X utf8 scripts/summarize_generalization.py',
      '```','',
      '评估脚本默认复用本次缓存，避免无意义重复推理；会核对模型与清单。缓存和冻结清单均在此目录。需要从头重跑时用下文的--fresh-cache参数，保留本次证据。',
      '```powershell',
      '& $projectPython -X utf8 scripts/evaluate_generalization.py --stage validation --fresh-cache',
      '& $projectPython -X utf8 scripts/evaluate_generalization.py --stage evaluation --fresh-cache',
      '& $projectPython -X utf8 scripts/summarize_generalization.py',
      '```','',
      '日常检测可显式添加 `--conf 0.5 --iou 0.5`；更重视少漏检时应评估其他设置，不把该值当成任何场景都最优的常数。原demo默认值未改。','',
      '## 文件与汇报用法','',
      '- `metrics.csv`：两套图片集的三种固定设置；`validation_grid.csv`：完整12格验证结果。',
      '- `图片逐张对比.html`：全部37张图片的标注、原设置、选定设置，无挑选成功案例。',
      '- `external_manifest.json`、`source_attribution.json`：候选清单、图片哈希、来源作者和许可元数据。',
      '- `annotation_review.json`：看模型预测前的筛查记录；`case_predictions.json`：用户案例原始框。',
      '- `run_record.json`、`selected_setting.json`及日志：模型哈希、环境、冻结设置、实际执行记录。','',
      '汇报主线可以说：“原测试集mAP较高，但自己找的犀牛图片出现重复框。我们在验证集选择推理阈值，再用固定测试集和新增样本检查。结果显示额外框减少，但召回率下降，说明实际应用需要明确误检与漏检的取舍；同时发现公开标注仍需复核。”','',
      '尚未完成：完整八类、足够规模、独立人工审核的外部数据集；多随机种子训练稳定性实验。本次结果不代表这些工作已经做过。','',
      '来源：[Open Images标注与评估说明](https://storage.googleapis.com/openimages/web/evaluation.html)。用户提供的三张案例只用于本地演示，未确认转载许可，不应随公开仓库直接再分发。']
    (OUT/'实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('Report, CSVs, plot and complete 37-image comparison generated.')
if __name__=='__main__':main()
