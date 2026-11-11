"""Rebuild raw-image contact sheets without distributing benchmark image pixels.

Supply original benchmark image directories with train/test (NEU, GC10) and
train/val (PCB) subdirectories. See docs/DATASETS.md for distributor provenance.
Example: python experiments/instance_contact_sheets.py --neu-root X/images --output outputs/instance_evidence
"""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
BENCHMARKS={'neu_det':('neu_det_copy_pairs.csv','test_image','test'),
            'gc10_det':('gc10_det_copy_pairs.csv','test_image','test'),
            'pcb_det':('dspcbsd_official_copy_pairs.csv','validation_image','val')}


def main(args):
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    geometry=json.loads((DATA/'additional_analyses/real_pair_geometry.json').read_text())
    try:
        font=ImageFont.truetype('DejaVuSans.ttf',19)
        small=ImageFont.truetype('DejaVuSans.ttf',16)
    except OSError:
        font=ImageFont.load_default(size=19)
        small=ImageFont.load_default(size=16)
    records=[]
    for ds,base in [('neu_det',args.neu_root),('gc10_det',args.gc10_root),('pcb_det',args.pcb_root)]:
        if base is None:continue
        filename,qcol,split=BENCHMARKS[ds]
        with (DATA/'leak_lists'/filename).open() as f:rows=list(csv.DictReader(f))
        gp={(p.get('test',p.get('val')),p['train']):p for p in geometry[ds]['pairs']}
        if ds=='neu_det':selected=list(range(len(rows)))
        else:
            ordered=sorted(range(len(rows)),key=lambda i:float(rows[i]['fused_score_F']))
            selected=sorted({ordered[int(x)] for x in np.linspace(0,len(ordered)-1,9)})
            corroborated=[i for i,r in enumerate(rows) if gp[(r[qcol],r['training_image'])]['corroborated']]
            if corroborated and not set(selected)&set(corroborated):selected.append(corroborated[0])
            selected.sort()
        for page in range(math.ceil(len(selected)/3)):
            ids=selected[page*3:(page+1)*3]
            sheet=Image.new('RGB',(1020,80+len(ids)*420),'#f5f1e8')
            draw=ImageDraw.Draw(sheet)
            draw.text((25,19),f'{ds} | raw evaluation image (left), training image (right)',font=font,fill='#243e43')
            for rownum,i in enumerate(ids):
                row=rows[i];qname=row[qcol];tname=row['training_image'];g=gp[(qname,tname)]
                top=67+rownum*420
                draw.text((25,top),f'Pair {i+1:02d}   {qname} / {tname}',font=font,fill='#243e43')
                draw.text((25,top+28),f'Frozen score {row["fused_score_F"]}; SIFT inliers {g["inliers"]}/{g["matches"]}; corroborated {g["corroborated"]}',font=small,fill='#52656a')
                for col,path in enumerate([base/split/qname,base/'train'/tname]):
                    if not path.is_file():raise FileNotFoundError(path)
                    with Image.open(path) as original:
                        im=ImageOps.contain(original.convert('RGB'),(470,340))
                    sheet.paste(im,(25+col*500+(470-im.width)//2,top+65+(340-im.height)//2))
                records.append({'dataset':ds,'pair_index_one_based':i+1,'evaluation_image':qname,'training_image':tname,
                                'score':float(row['fused_score_F']),'sift_matches':g['matches'],'sift_inliers':g['inliers'],
                                'sift_corroborated':g['corroborated'],'contact_sheet':f'{ds}_pairs_{page+1:02d}.png'})
            sheet.save(out/f'{ds}_pairs_{page+1:02d}.png')
    (out/'visual_selection.json').write_text(json.dumps({'selection':'all 17 NEU; nine evenly spaced score ranks for others, plus a corroborated example when absent','records':records},indent=2)+chr(10),encoding='utf-8')
    print('Rendered',len(records),'raw pairs. No labels were inferred by this script.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--neu-root',type=Path)
    p.add_argument('--gc10-root',type=Path)
    p.add_argument('--pcb-root',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/instance_evidence')
    args=p.parse_args()
    if all(v is None for v in [args.neu_root,args.gc10_root,args.pcb_root]):p.error('supply at least one image root')
    main(args)
