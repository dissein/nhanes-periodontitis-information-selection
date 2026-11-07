"""Run with the notebook Python; install into a project-local environment."""
import sys,subprocess,venv
from pathlib import Path
def main():
    if sys.version_info<(3,11):raise SystemExit('Python 3.11+ is required; Python 3.13 is recommended to match the reference.')
    root=Path(__file__).resolve().parent.parent;env=root/'.venv'
    py=env/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    if not py.exists():venv.EnvBuilder(with_pip=True).create(env)
    subprocess.run([str(py),'-m','pip','install','-r',str(root/'requirements.lock.txt')],check=True)
    subprocess.run([str(py),'-c',"import sys;sys.path.insert(0,'scripts');import engine;print('BUNDLE',engine.verify())"],cwd=root,check=True)
    print('TERMINAL_PYTHON',py)
if __name__=='__main__':main()
