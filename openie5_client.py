"""
Client for accessing OpenIE 5 in Python. Skeleton based off StanfordNLP's package.
"""
import os
import requests
import json
import shlex
import subprocess
import time
import sys
from helperutils import *

class UnknownException(Exception):
    """ Exception raised when there is an unknown error. """
    pass

class ShouldRetryException(Exception):
    """ Exception raised if the service should retry the request. """
    pass

class PermanentlyFailedException(Exception):
    """ Exception raised if the service should retry the request. """
    pass

class RobustService(object):
    """ Service that resuscitates itself if it is not available. """
    CHECK_ALIVE_TIMEOUT = 120

    def __init__(self, start_cmd, stop_cmd, endpoint, install_dir, 
                 stdout=sys.stdout, stderr=sys.stderr, be_quiet=False):
        self.start_cmd = start_cmd and shlex.split(start_cmd)
        self.stop_cmd = stop_cmd and shlex.split(stop_cmd)
        self.endpoint = endpoint
        self.install_dir = install_dir
        self.stdout = stdout
        self.stderr = stderr

        self.server = None
        self.is_active = False
        self.be_quiet = be_quiet
        
    def is_alive(self):
        try:
            # placeholder method to check if server is up
            self.extractor.extract('this is a ping')
            return True
        except requests.exceptions.ConnectionError as e:
            raise ShouldRetryException(e)

    def execute(self, command):
        self.server = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # Poll process for new output until finished
        while True:
            nextline = self.server.stdout.readline()
            if nextline == f'Server started at port {self.port} ...\r\n'.encode('utf-8'):
                break
            sys.stdout.write(nextline)
            sys.stdout.flush()  

    def start(self):
        if self.start_cmd:
            if self.be_quiet:
                # Issue #26: subprocess.DEVNULL isn't supported in python 2.7.
                stderr = open(os.devnull, 'w')
            else:
                stderr = self.stderr
            print(f"Starting server with command: {' '.join(self.start_cmd)}")
            cwd = os.getcwd()
            os.chdir(self.install_dir)
            self.server = subprocess.Popen(self.start_cmd, shell=True, stderr=stderr, stdout=stderr)
            #self.execute(self.start_cmd)
            os.chdir(cwd)
            

    def stop(self):
        if self.server:
            self.server.kill()
        if self.stop_cmd:
            subprocess.run(self.stop_cmd, check=True)
        self.is_active = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, _, __, ___):
        self.stop()

    def ensure_alive(self):
        # Check if the service is active and alive\
        if self.is_active:
            try:
                return self.is_alive()
            except ShouldRetryException:
                pass

        # If not, try to start up the service.
        if self.server is None:
            self.start()

        # Wait for the service to start up.
        start_time = time.time()
        while True:
            try:
                if self.is_alive():
                    break
            except ShouldRetryException:
                pass

            if time.time() - start_time < self.CHECK_ALIVE_TIMEOUT:
                time.sleep(1)
            else:
                raise PermanentlyFailedException("Timed out waiting for service to come alive.")

        # At this point we are guaranteed that the service is alive.
        self.is_active = True
        
class OpenIEClient(RobustService):
    """ An OpenIE client to the OpenIE server. """

    DEFAULT_PORT = "8000" 
    DEFAULT_MEMORY = "10G"
    DEFAULT_PATH = ""
    DEFAULT_TIMEOUT = 60000

    def __init__(self, start_server=True,
                 port=DEFAULT_PORT,
                 path=DEFAULT_PATH,
                 stdout=sys.stdout,
                 stderr=sys.stderr,
                 memory=DEFAULT_MEMORY,
                 timeout=DEFAULT_TIMEOUT,
                 be_quiet=True,
                 **kwargs):
        
        # check if path given is a full path or a ./ path
        if (":/" in path) or (":\\" in path):
            pass
        else :
            path = os.path.join(os.getcwd(),path)
            
        install_dir = os.path.join(path, 'openie5') 
        # if openie5 not installed
        if not os.path.exists(install_dir):
            # clone github
            print('Install not found, installing in path')
            
            remote_url = 'https://github.com/dair-iitd/OpenIE-standalone/archive/master.zip'
            
            import urllib.request
            from tqdm import tqdm
            from zipfile import ZipFile
            import shutil

            class DownloadProgressBar(tqdm):
                def update_to(self, b=1, bsize=1, tsize=None):
                    if tsize is not None:
                        self.total = tsize
                    self.update(b * bsize - self.n)

            def download_url(url, output_path):
                with DownloadProgressBar(unit='B', unit_scale=True,
                                        miniters=1, desc=url.split('/')[-1]) as t:
                    urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
                    

            def download_file_from_google_drive(id, destination):
                URL = 'https://docs.google.com/uc?export=download'

                session = requests.Session()
                response = session.get(URL, params = { 'id' : id }, stream = True)

                token = None
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        token = value
                        break

                if token:
                    params = { 'id' : id, 'confirm' : token }
                    response = session.get(URL, params = params, stream = True)

                CHUNK_SIZE = 32*1024
                # TODO: this doesn't seem to work; there's no Content-Length value in header?
                total_size = int(response.headers.get('content-length', 0))

                with tqdm(desc=destination, total=total_size, unit='B', unit_scale=True) as pbar:
                    with open(destination, 'wb') as f:
                        for chunk in response.iter_content(CHUNK_SIZE):
                            if chunk:
                                pbar.update(CHUNK_SIZE)
                                f.write(chunk)

            print('Downloading to %s.' % install_dir)
            download_url(remote_url, 'openie5.zip')
            print('\nExtracting to %s.' % install_dir)
            zf = ZipFile('openie5.zip')
            zf.extractall(path=install_dir)
            zf.close()
            os.remove('openie5.zip')
            # move files up one level
            source_dir = os.path.join(install_dir,'OpenIE-standalone-master/')
            for f in os.listdir(source_dir):
                shutil.move(os.path.join(source_dir,f),install_dir)
            os.rmdir(os.path.join(install_dir,'OpenIE-standalone-master/'))
            
            # download language model
            lang_model_dir = os.path.join(install_dir,'data')
            print('Downloading language model to %s.' % lang_model_dir)
            download_file_from_google_drive('0B-5EkZMOlIt2cFdjYUJZdGxSREU', os.path.join(lang_model_dir,'languageModel'))
            
            # download pre-compiled OpenIE standalone jar
            print('Downloading pre-compiled jar to %s.' % install_dir)
            download_file_from_google_drive('19z8LO-CYOfJfV5agm82PZ2JNWNUPIB6D', os.path.join(install_dir, 'openie-assembly.jar'))

        # create path to jar file for openie server
        # get server address
        endpoint = "".join(["http://localhost:",port])
        sys.path.append(install_dir)
        # start the server
        if start_server:
            start_cmd = f"java -Xmx{memory} -cp '{install_dir}/*' -XX:+UseG1GC -jar openie-assembly.jar --httpPort {port}"
            stop_cmd = None
        else:
            start_cmd = stop_cmd = None
            self.server_start_info = {}
            
        self.port = port
        self.timeout = timeout
        
        super(OpenIEClient, self).__init__(start_cmd, stop_cmd, endpoint,
                                            install_dir, stdout, stderr, be_quiet)
        
        self.extractor = OpenIE5(endpoint)
    
    def extract(self, text):
        self.ensure_alive()
        print("server ensured alive")
        try:
            print(f"attempting to extract: {text}")
            data = self.extractor.extract(text)
            print("extracted")
            return data
        except requests.HTTPError as e:
            print("error")
            raise UnknownException()

class OpenIE5:

    def __init__(self, server_url):
        if server_url[-1] == '/':
            server_url = server_url[:-1]
        self.server_url = server_url
        self.extract_context = '/getExtraction'

    def extract(self, text, properties=None):
        assert isinstance(text, str)
        if properties is None:
            properties = {}
        else:
            assert isinstance(properties, dict)

        requests.get(self.server_url)
        print("url get. encoding")
        # try:
        #     requests.get(self.server_url)
        # except requests.exceptions.ConnectionError:
        #     raise Exception('Check whether you have started the OpenIE5 server')

        data = text.encode('utf-8')
        print("text encoded, sending post req")

        r = requests.post(
            self.server_url + self.extract_context, params={
                'properties': str(properties)
            }, data=data, headers={'Connection': 'close'})
        print("reply received")
        
        return json.loads(r.text)