from pathlib import Path
import json, os, shutil, subprocess, sys, tempfile
root=Path(__file__).resolve().parents[1]
base=Path(tempfile.mkdtemp(prefix='paiec-publisher-test-'))
realgit=shutil.which('git'); fakebin=base/'bin'; fakebin.mkdir(); bare=base/'remote.git'
subprocess.run([realgit,'init','--bare','--initial-branch=main',str(bare)],check=True,capture_output=True)
state=base/'state.json'; state.write_text(json.dumps({'exists':True,'invited':False,'writes':[]}))
py=sys.executable
(fakebin/'gh').write_text('#!'+py+'\n'+r'''
import json,os,subprocess,sys
from pathlib import Path
s=Path(os.environ['PAIEC_TEST_STATE']); v=json.loads(s.read_text()); a=sys.argv[1:]
if a[:2]==['auth','setup-git']: sys.exit(0)
if a[:2]==['repo','create']:
 v['exists']=True; v['writes'].append('create'); s.write_text(json.dumps(v)); sys.exit(0)
if a[0]!='api': raise RuntimeError(a)
e=a[a.index('--header')+2]; method=a[a.index('--method')+1]; status=200; body={}
if e=='user': body={'login':'ankaggarwal94','type':'User','id':20984056}
elif e=='repos/ankaggarwal94/paiec-2026-rasch-caimira':
 if v['exists']: body={'full_name':'ankaggarwal94/paiec-2026-rasch-caimira','owner':{'login':'ankaggarwal94'},'private':False,'visibility':'public','permissions':{'admin':True},'default_branch':'main'}
 else: status=404
elif e.endswith('/git/ref/heads/main'):
 p=subprocess.run([os.environ['PAIEC_REAL_GIT'],'--git-dir='+os.environ['PAIEC_TEST_BARE'],'rev-parse','refs/heads/main'],capture_output=True)
 if p.returncode: status=409; body={'message':'Git Repository is empty.'}
 else: body={'object':{'sha':p.stdout.decode().strip()}}
elif e.endswith('/collaborators/maharshi95/permission'): status=404
elif '/invitations?' in e: body=[{'id':12345,'invitee':{'login':'maharshi95'},'permissions':'write'}] if v['invited'] else []
elif e.endswith('/collaborators/maharshi95') and method=='PUT':
 v['invited']=True; v['writes'].append('invite'); s.write_text(json.dumps(v)); status=201; body={'id':12345}
else: raise RuntimeError((method,e))
print(f'HTTP/2.0 {status}\nContent-Type: application/json\n\n'+json.dumps(body))
sys.exit(0 if status<400 else 1)
''')
(fakebin/'git').write_text('#!'+py+'\n'+r'''
import os,sys
args=[os.environ['PAIEC_TEST_BARE'] if a=='https://github.com/ankaggarwal94/paiec-2026-rasch-caimira.git' else a for a in sys.argv[1:]]
os.execv(os.environ['PAIEC_REAL_GIT'],[os.environ['PAIEC_REAL_GIT'],*args])
''')
for p in fakebin.iterdir(): p.chmod(0o755)
env=dict(os.environ,PATH=str(fakebin)+os.pathsep+os.environ['PATH'],PAIEC_TEST_STATE=str(state),PAIEC_TEST_BARE=str(bare),PAIEC_REAL_GIT=realgit)
flags=['--expected-owner','ankaggarwal94','--expected-name','paiec-2026-rasch-caimira']
fixture=base/'fixture'; fixture.mkdir(); (fixture/'README.md').write_text('Publication fixture.\n')
def call(label,extra,cwd=fixture,code=0):
 p=subprocess.run([py,str(root/'publishing/publish.py'),*flags,*extra],cwd=cwd,env=env,capture_output=True,text=True,timeout=45)
 (base/(label+'.stdout')).write_text(p.stdout);(base/(label+'.stderr')).write_text(p.stderr)
 assert p.returncode==code,(label,p.returncode,p.stdout,p.stderr)
 return p
call('dry_run_existing_empty',['--dry-run'])
assert json.loads(state.read_text())['writes']==[]
call('publish_existing_empty',[])
assert json.loads(state.read_text())['writes']==['invite']
call('verify_published',['--verify-only'])
call('reject_existing_history',[],code=1)
print(json.dumps({'status':'passed','cases':4,'scope':'Mock GitHub API with real local Git: existing-empty dry run, initial push/invitation, verification, and populated-repository rejection','logs':str(base)},indent=2))
