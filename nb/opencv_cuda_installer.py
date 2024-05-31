import os
import utils

# when a new version of the colab/kaggle environment is released
# some of the dependencies may no longer be the versions that opencv was built with
def double_check_opencv_deps(cv_md5_file, opencv_lib='opencv/install/lib/libopencv_world.so'):
    script_file = '/tmp/double_check_opencv_deps.sh'
    script=r'''
    OPENCV_LIB=$1
    MD5_FILE=$2
    DEPS=`ldd $OPENCV_LIB | grep "=>" | sed "s/.*=> //" | sed "s/ (.*//"`
    MD5=`md5sum -b $DEPS | md5sum | head -c 32`
    NEED_MD5=`cat $MD5_FILE | md5sum | head -c 32`
    if [ "$MD5" != "$NEED_MD5" ]; then
        echo "A new OpenCV binary for this specific environment probably needs to be built." 
        exit 1
    fi
    '''
    with open(script_file, "w") as f:
        f.write(script)
    return 0 == os.system(f'chmod +x {script_file} && {script_file} "{opencv_lib}" "{cv_md5_file}"')

# this simpler check can be run before downloading the archive
# but doesn't work if the library paths change
def check_opencv_deps(cv_md5_file):
    ok = (0 == os.system(f'md5sum -c {cv_md5_file} > /dev/null'))
    if not ok:
        print('A new OpenCV binary for this specific environment probably needs to be built.')
    return ok

def test_opencv_cuda(which):
    print(f'testing the {which} opencv ...')
    # Unloading a module already imported into the Jupyter kernel doesn't seem to work.
    # Colab can even restart the kernel if it detects that an imported module was modified.
    # But it seems to work if a separate interpreter runs the test:
    script_file = '/tmp/test_opencv_cuda.py'
    script=r'''
    import cv2
    print(cv2.getBuildInformation())
    import numpy as np
    a = np.full((480,480), 60, np.uint8)
    ga = cv2.cuda.GpuMat(a)
    '''.replace('    ', '') # python cares about the indentation
    with open(script_file, "w") as f:
        f.write(script)
    return 0 == os.system(f'python3 {script_file} 2> /dev/null | grep -m1 -A2 "NVIDIA CUDA"')

def can_use_opencv_cuda():
    if 0 != os.system('nvidia-smi'):
        print('no GPU present')
        return False
    return True

def download_opencv_cuda_from_huggingface(repo_id):
    if utils.get_platform() == 'unknown':
        print('no pre-built OpenCV CUDA binaries available for this platform')
        return ''
    # TODO: enable pre-checking the dependencies when it's more reliable
    if False:
        print('checking if the dependencies have changed ...')
        cv_md5_file = f'opencv/{utils.get_platform()}/opencv_cuda.md5'
        utils.download_from_huggingface(f'{repo_id}/{cv_md5_file}')
        if not check_opencv_deps(cv_md5_file):
            return ''
    # the opencv cuda binaries might be built for a specific sets of gpu architectures
    # TODO: check that, but for now just show the current arch
    os.system('nvidia-smi --query-gpu=compute_cap --format=csv')
    print('downloading the opencv archive ...')
    cv2_archive = f'opencv/{utils.get_platform()}/opencv_cuda.tar.gz'
    utils.download_from_huggingface(f'{repo_id}/{cv2_archive}')
    return cv2_archive

def install_opencv_cuda_impl(download_func, repo_id):
    if not can_use_opencv_cuda():
        return False
    if test_opencv_cuda('current'):
        return True
    cv2_archive = download_func(repo_id)
    if not os.path.exists(cv2_archive):
        return False
    print('extracting the opencv archive ...')
    os.system(f'tar -xzf {cv2_archive} && rm {cv2_archive} && mv install opencv')
    print('checking if the dependencies have changed ...')
    if not double_check_opencv_deps('opencv/install/lib/opencv_cuda.md5'):
        return False
    print('uninstalling the old opencv ...')
    os.system('pip uninstall -y opencv_python_headless opencv_python opencv_contrib_python')
    print('installing the new opencv ...')
    from distutils.sysconfig import get_python_lib
    os.system(f'cp -r cv2 {get_python_lib()} && rm -rf cv2')
    return test_opencv_cuda('new')

def install_opencv_cuda(download_func = download_opencv_cuda_from_huggingface, repo_id = 'logicbear/cache'):
    use_opencv_cuda = install_opencv_cuda_impl(download_func, repo_id)
    if not use_opencv_cuda:
        print("WARNING: We cannot use OpenCV with CUDA support!")
    return use_opencv_cuda
