"""Checksum-verified natural overlapping audio; noncommercial research only."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request,urlopen
import zipfile

RECORD='https://zenodo.org/records/7244360/files/'
FILES={'development_annotation.zip':'e6a3f84f8020725d559b38ccb494ef3d',
       'development_audio.zip':'3de7cb4f92a115a6f5cc077a41ca07b3',
       'LICENSE.txt':'d1111f68bf579834e148db1788ae4cf9',
       'README.md':'a8f77695c5933dcfcaea5018bc43c154'}


def digest(path):
    md5=hashlib.md5()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            md5.update(block)
    return md5.hexdigest()


def fetch(output,audio=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    metadata=[]
    for name,expected in FILES.items():
        if name=='development_audio.zip' and not audio:
            continue
        path=output/name
        if not path.exists() or digest(path)!=expected:
            temporary=path.with_suffix(path.suffix+'.part')
            url=RECORD+name+'?download=1'
            print('Downloading '+name,flush=True)
            with urlopen(Request(url,headers={'User-Agent':'portfolio-noncommercial-audio-research'}),timeout=60) as response,temporary.open('wb') as handle:
                for block in iter(lambda:response.read(1024*1024),b''):
                    handle.write(block)
            if digest(temporary)!=expected:
                raise ValueError('Checksum mismatch: '+name)
            temporary.replace(path)
        metadata.append({'name':name,'md5':expected,'bytes':path.stat().st_size,'url':RECORD+name+'?download=1'})
        if name.endswith('.zip'):
            with zipfile.ZipFile(path) as archive:
                for entry in archive.infolist():
                    destination=(output/entry.filename).resolve()
                    if not destination.is_relative_to(output.resolve()):
                        raise ValueError('Archive path escapes output directory')
                archive.extractall(output)
    result={'dataset':'MAESTRO Real v1','doi':'10.5281/zenodo.7244360',
            'license':'Tampere University custom experimental/noncommercial license; retain full notice',
            'source':'Machine Listening Group at Tampere University','files':metadata,
            'description':'Naturally overlapping real-life recordings with human-aggregated soft temporal labels'}
    (output/'provenance.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='data/external/maestro')
    parser.add_argument('--audio',action='store_true',help='Download 2.6 GB real recordings as well as annotations')
    args=parser.parse_args()
    print(json.dumps(fetch(args.output,args.audio),indent=2))
