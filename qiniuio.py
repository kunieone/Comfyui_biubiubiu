from server import PromptServer
from aiohttp import web
import pathlib
import os
import time
import requests
import folder_paths
import copy
import glob
from qiniu import put_file, etag

def get_dir_by_type(dir_type):
    if dir_type is None:
        dir_type = "input"

    if dir_type == "input":
        type_dir = folder_paths.get_input_directory()
    elif dir_type == "temp":
        type_dir = folder_paths.get_temp_directory()
    elif dir_type == "output":
        type_dir = folder_paths.get_output_directory()

    return type_dir, dir_type

def download_url(url, retry=5):
    download_OK = False
    attempts = 0
    content = None


    try:
        while not download_OK:
            assert attempts <= retry
            try:
                response = requests.get(url)

            except Exception as e:
                print(f"Download {url} network error :{e}")

            if response.status_code == 200:
                content = copy.deepcopy(response.content)
                return content
            time.sleep(0.1)
            attempts += 1
    except:
        raise RuntimeError(f"Download image {url} failed after {attempts} attempts.")
    
    # return content


def download_image(post):
    image_url = post.get("image_url", None)
    image_name = post.get("image_name", None)

    if image_url is None or image_name is None:
        return web.Response(status=400)
    
    overwrite = post.get("overwrite")
    image_upload_type = post.get("type", "input")
    upload_dir, image_upload_type = get_dir_by_type(image_upload_type)
    subfolder = post.get("subfolder", "")
    full_output_folder = os.path.join(upload_dir, os.path.normpath(subfolder))
    filepath = os.path.abspath(os.path.join(full_output_folder, image_name))
    
    if os.path.isdir(filepath):
        return web.Response(status=400)

    pathlib.Path(filepath).parent.mkdir(exist_ok=True, parents=True)
    if os.path.commonpath((upload_dir, filepath)) != upload_dir:
        return web.Response(status=400)

    if not os.path.exists(full_output_folder):
        os.makedirs(full_output_folder)
    filename = image_name
    split = os.path.splitext(filename)

    if overwrite is not None and (overwrite == "true" or overwrite == "1"):
        pass
    else:
        i = 1
        while os.path.exists(filepath):
            filename = f"{split[0]} ({i}){split[1]}"
            filepath = os.path.join(full_output_folder, filename)
            i += 1

    image_content = download_url(image_url)

    if image_content:
        with open(filepath, "wb") as f:
            f.write(image_content)

        return web.json_response({"name" : filename, "subfolder": subfolder, "type": image_upload_type})
    else:
        return web.Response(status=400)

def upload2qiniu(post):
    token = post.get("token")
    key = post.get("key")
    localfile = os.path.join(folder_paths.get_output_directory(), post.get("file_path"))
    if not os.path.exists(localfile):
        return web.Response(status=404)
    if not os.path.isfile(localfile):
        return web.Response(status=404)
    try:
        ret, info = put_file(token, key, localfile, version='v2')
        assert ret['key'] == key
        assert ret['hash'] == etag(localfile)
    except:
        return web.Response(status=400)
    
    return web.Response(status=200)


def list_output_image(filename_prefix, filename_suffix):
    output_dir = folder_paths.get_output_directory()
    fns = glob.glob(os.path.join(output_dir, filename_prefix, '*'+filename_suffix))
    fns = [f.replace(output_dir + '/', "") for f in fns if os.path.isfile(f)]
    return fns


def list_input_image(filename_prefix, filename_suffix):
    output_dir = folder_paths.get_input_directory()
    fns = glob.glob(os.path.join(output_dir, filename_prefix, '*'+filename_suffix))
    fns = [f.replace(output_dir + '/', "") for f in fns if os.path.isfile(f)]
    return fns


@PromptServer.instance.routes.post("/imageio/upload_image_from_qiniu")
async def download_image_from_qiniu(request):
    print(request)
    post = await request.json()
    return download_image(post)


@PromptServer.instance.routes.get("/imageio/list_output_image")
async def download_image_from_qiniu(request):
    if "prefix" in request.rel_url.query:
        prefix = request.rel_url.query["prefix"]
    else:
        prefix = ''

    if "suffix" in request.rel_url.query:
        suffix = request.rel_url.query["suffix"]
    else:
        suffix = ''
    fns = list_output_image(prefix, suffix)
    return web.json_response({"names" : fns, })

@PromptServer.instance.routes.get("/imageio/list_input_image")
async def download_image_from_qiniu(request):
    if "prefix" in request.rel_url.query:
        prefix = request.rel_url.query["prefix"]
    else:
        prefix = ''

    if "suffix" in request.rel_url.query:
        suffix = request.rel_url.query["suffix"]
    else:
        suffix = ''
    fns = list_input_image(prefix, suffix)
    return web.json_response({"names" : fns, })


@PromptServer.instance.routes.post("/imageio/download_image_to_qiniu")
async def download_image_from_qiniu(request):
    post = await request.json()
    return upload2qiniu(post)
