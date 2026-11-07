"""Author Figure 1 directly as SVG and render its vector PDF and PNG."""
from pathlib import Path
import csv
import hashlib
import json
from html import escape

import fitz

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT
FLOW = PREVIOUS / 'data/Figure1_participant_flow.csv'
WIDTH, HEIGHT = 1600, 820
INK, MUTED, RULE = '#26323B', '#6C7C87', '#D7E0E5'
BLUE, OCHRE, PALE = '#286F8E', '#C67A45', '#DFE9EE'
elements = []


def text(x, y, content, size=17, color=INK, weight='normal', anchor='start'):
    elements.append(
        f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" '
        f'text-anchor="{anchor}">{escape(str(content))}</text>'
    )


def line(points, color=RULE, width=1.4, dash=None):
    coordinates = ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)
    dash_attribute = f' stroke-dasharray="{dash}"' if dash else ''
    elements.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="{width}"{dash_attribute}/>' )


def arrow(points, color=MUTED, width=1.5):
    line(points, color, width)
    x, y = points[-1]
    px, py = points[-2]
    dx, dy = x - px, y - py
    length = (dx * dx + dy * dy) ** .5
    ux, uy = dx / length, dy / length
    wing = [(x, y), (x - 7 * ux + 3 * uy, y - 7 * uy - 3 * ux), (x - 7 * ux - 3 * uy, y - 7 * uy + 3 * ux)]
    coordinates = ' '.join(f'{a:.2f},{b:.2f}' for a, b in wing)
    elements.append(f'<polygon points="{coordinates}" fill="{color}"/>')


def rect(x, y, w, h, fill='white', stroke=RULE, width=1.4, radius=0):
    elements.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    flow=read(FLOW)
    navy='#163047';blue_fill='#DCEBF5';green_fill='#E1F0E7'
    rect(10,8,1580,204,stroke='#B8CAD5',radius=22)
    rect(28,25,34,34,fill=navy,stroke='none',radius=9)
    text(45,50,'A',size=24,color='white',weight='bold',anchor='middle')
    text(81,51,'Study population',size=25,weight='bold')
    xs=[405,640,875,1110,1420]
    headings=[('All participants',),('Age ≥30 years',),('Positive MEC weight',),('Complete periodontal','examination'),('Classifiable','CDC/AAP outcome')]
    for x,labels in zip(xs,headings):
        for i,label in enumerate(labels):text(x,43+i*21,label,size=18,anchor='middle')
    for cycle,year,role,y,fill in [('G','2011–2012','Development',112,blue_fill),('H','2013–2014','Validation',178,green_fill)]:
        rows=[row for row in flow if row['cycle']==cycle]
        rect(61,y-39,240,57,fill=fill,stroke='none',radius=23)
        text(81,y-15,'NHANES '+year,size=21,weight='bold')
        text(81,y+7,'('+role+')',size=20)
        arrow([(309,y-10),(330,y-10)],color=navy,width=1.4)
        for i,(x,row) in enumerate(zip(xs,rows)):
            rect(x-60,y-33,120,49,fill=fill if i==4 else '#F5F8FA',stroke='none',radius=24)
            text(x,y,f"{int(row['included_n']):,}",size=25,weight='bold' if i==4 else 'normal',anchor='middle')
            if i<4:arrow([(x+69,y-10),(xs[i+1]-69,y-10)],color=navy,width=1.4)
    rect(10,225,1580,585,stroke='#C4D4DE',radius=22)
    rect(28,240,34,34,fill=navy,stroke='none',radius=9)
    text(45,265,'B',size=24,color='white',weight='bold',anchor='middle')
    text(81,266,'Analytical framework',size=25,weight='bold')
    headers=[
        (15,490,'01','Internal evaluation','Development cohort, n = 3,323',blue_fill,BLUE),
        (540,590,'02','Full-sample attribution and selection','Full development sample, n = 3,323',blue_fill,BLUE),
        (1165,420,'03','Subsequent-cycle validation','Validation cohort',green_fill,'#4D8978')]
    for x,w,num,title,subtitle,fill,accent in headers:
        rect(x,282,w,515,stroke='#C4D7E2',radius=21)
        rect(x+1,283,w-2,69,fill=fill,stroke='none',radius=20)
        rect(x+20,294,62,45,fill=accent,stroke='none',radius=22)
        text(x+51,325,num,size=27,color='white',weight='bold',anchor='middle')
        text(x+96,310,title,size=22,weight='bold')
        text(x+96,338,subtitle,size=20)
    # All four information-selection strategies use the same outer folds.
    text(260,392,'5-fold PSU-grouped nested CV',size=22,weight='bold',anchor='middle')
    for j in range(5):
        text(97+j*52,426,f'F{j+1}',size=19,anchor='middle')
        for i in range(5):rect(76+j*52,440+i*19,43,14,fill='#5488AC' if i==j else '#EAF1F6',stroke='none',radius=2)
    rect(366,467,17,13,fill='#EAF1F6',stroke='none',radius=2);text(392,480,'Training',size=18)
    rect(366,498,17,13,fill='#5488AC',stroke='none',radius=2);text(392,511,'Held out',size=18)
    line([(259,534),(259,553)],color=navy,width=1.1)
    line([(90,553),(445,553)],color=navy,width=1.1)
    methods=[(90,88,'EBM',blue_fill),(210,121,'TreeSHAP',blue_fill),(335,116,'Exhaustive','#F1F5F8'),(445,97,'Random','#F1F5F8')]
    for x,w,label,fill in methods:
        arrow([(x,553),(x,574)],color=navy,width=1.1)
        rect(x-w/2,574,w,42,fill='#D2E6F5',stroke='none',radius=21)
        text(x,601,label,size=19,weight='bold',anchor='middle')
        line([(x,619),(x,640)],color=navy,width=1.1)
    line([(90,640),(445,640)],color=navy,width=1.1)
    arrow([(259,640),(259,675)],color=navy,width=1.8)
    rect(45,675,430,89,fill='#E7F0F7',stroke='none',radius=20)
    text(260,711,'Ridge logistic regression',size=23,weight='bold',anchor='middle')
    text(260,743,'3 inner PSU folds for tuning',size=21,anchor='middle')
    text(260,787,'Pooled OOF AUC · Brier',size=20,weight='bold',anchor='middle')
    # Matched branch-card geometry follows the supplied reference.
    rect(555,365,560,82,fill='#E7F0F7',stroke='none',radius=19)
    text(835,394,'Global attribution ranking',size=23,weight='bold',anchor='middle')
    rect(685,406,130,34,fill='#D2E6F5',stroke='none',radius=17)
    rect(855,406,140,34,fill='#D2E6F5',stroke='none',radius=17)
    text(750,430,'EBM',size=21,weight='bold',anchor='middle')
    text(925,430,'TreeSHAP',size=21,weight='bold',anchor='middle')
    line([(835,449),(835,465)],color=navy,width=1.2)
    line([(690,465),(980,465)],color=navy,width=1.2)
    arrow([(690,465),(690,484)],color=navy,width=1.2)
    arrow([(980,465),(980,484)],color=navy,width=1.2)
    rect(555,485,270,190,fill='#E7F0F7',stroke='none',radius=20)
    rect(845,485,270,190,fill='#E7F0F7',stroke='none',radius=20)
    text(690,518,'Ranking stability',size=22,weight='bold',anchor='middle')
    text(980,518,'Information budgets',size=22,weight='bold',anchor='middle')
    for x,y in [(574,538),(581,545),(588,552)]:
        rect(x,y,48,30,stroke='#8298A7',width=1)
        for j in range(3):rect(x+7+j*12,y+9,8,12,fill='#A8CBDD' if j==1 else '#DFEAF1',stroke='none')
    text(645,568,'500 PSU bootstrap',size=18)
    arrow([(690,595),(690,619)],color=navy,width=1.2)
    rect(630,620,120,45,fill='#D2E6F5',stroke='none',radius=22)
    text(690,650,'3-unit',size=23,weight='bold',anchor='middle')
    text(980,556,'80% cumulative attribution',size=19,anchor='middle')
    line([(980,566),(980,586)],color=navy,width=1.1)
    line([(920,586),(1045,586)],color=navy,width=1.1)
    arrow([(920,586),(920,617)],color=navy,width=1.2)
    arrow([(1045,586),(1045,617)],color=navy,width=1.2)
    for x,w,name,k in [(862,116,'TreeSHAP','8-unit'),(993,104,'EBM','10-unit')]:
        rect(x,618,w,47,fill='#D2E6F5',stroke='none',radius=21)
        text(x+w/2,639,name,size=20,weight='bold',anchor='middle')
        text(x+w/2,660,k,size=20,weight='bold',anchor='middle')
    line([(690,679),(690,698),(980,698),(980,679)],color=navy,width=1.2)
    arrow([(835,698),(835,730)],color=navy,width=1.5)
    text(941,721,'Full 17-unit',size=19,weight='bold')
    arrow([(926,715),(835,715)],color=navy,width=1.2)
    rect(555,730,560,51,fill='#E7F0F7',stroke='none',radius=19)
    text(835,764,'Final Ridge models + frozen cutoffs',size=23,weight='bold',anchor='middle')
    # Stage arrows indicate the study sequence, not reuse of held-out data.
    arrow([(506,537),(537,537)],color=navy,width=2)
    arrow([(1131,537),(1162,537)],color=navy,width=2)
    rect(1190,366,370,46,fill='#F1F2F3',stroke='none',radius=20)
    text(1375,396,'NHANES 2013–2014 (n = 3,617)',size=21,weight='bold',anchor='middle')
    arrow([(1375,421),(1375,464)],color=navy,width=1.8)
    rect(1190,485,370,112,fill='#F1F2F3',stroke='none',radius=20)
    text(1375,528,'Prediction performance',size=23,weight='bold',anchor='middle')
    text(1375,569,'AUC · Brier · Calibration',size=21,anchor='middle')
    arrow([(1375,603),(1375,637)],color=navy,width=1.8)
    rect(1190,647,370,126,fill='#F1F2F3',stroke='none',radius=20)
    text(1375,687,'Screening performance',size=23,weight='bold',anchor='middle')
    text(1375,724,'Sensitivity · Specificity · PPV / NPV',size=18,anchor='middle')
    text(1375,755,'Screen-positive proportion · DCA',size=18,anchor='middle')
    description='Reference-matched NHANES study design. Both development stages use 3,323 participants and subsequent-cycle validation uses 3,617. All four strategies share the outer PSU folds.'
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img">\n<title>NHANES study design</title>\n<desc>{description}</desc>\n<rect width="1600" height="820" fill="white"/>\n'+'\n'.join(elements)+'\n</svg>\n'
    target=ROOT/'figures/Figure1_participants_workflow.svg';target.write_text(svg,encoding='utf-8')
    document=fitz.open(target)
    document[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(target.with_suffix('.png'))
    pdf=fitz.open('pdf',document.convert_to_pdf());output=fitz.open()
    page=output.new_page(width=7.4*72,height=7.4*72*HEIGHT/WIDTH);page.show_pdf_page(page.rect,pdf,0)
    output.save(target.with_suffix('.pdf'),garbage=4,deflate=True)
    (ROOT/'source_manifest.json').write_text(json.dumps({'sources':{FLOW.relative_to(ROOT).as_posix():hashlib.sha256(FLOW.read_bytes()).hexdigest()},'native_svg':True,'periodontal_exam_n_G':3685,'internal_n':3323,'full_development_n':3323,'validation_n':3617,'random_uses_same_five_outer_folds':True},indent=2),encoding='utf-8')
    print(target)


if __name__=='__main__':
    main()
