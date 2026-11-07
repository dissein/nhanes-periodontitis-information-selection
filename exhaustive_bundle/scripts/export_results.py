import argparse,zipfile
import engine as e
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-root',required=True);ap.add_argument('--node',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    node=e.node_name(a.node);root=e.Path(a.output_root)/node;target=e.Path(a.out).resolve()
    assert not (root/'worker.lock').exists(),'Stop the worker before exporting a consistent snapshot'
    assert target.parent!=root and not target.is_relative_to(root.resolve()),'Write the archive outside the node output directory'
    target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in root.rglob('*'):
            if p.is_file():z.write(p,node+'/'+p.relative_to(root).as_posix())
    print('EXPORT',target)
if __name__=='__main__':main()
