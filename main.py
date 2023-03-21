from path_tool import get_all
import shutil
import easyocr
import cv2
import numpy as np
import re
import time
from multiprocessing import Process
import config

save_res = config.save_res
crop_img = config.crop_img
crop_top = config.crop_top
crop_bot = config.crop_bot

def draw_bb(pts: list, img: np.ndarray):
    cv2.line(img, tuple(pts[0]), tuple(pts[1]), color=(0,0,255))
    cv2.line(img, tuple(pts[1]), tuple(pts[2]), color=(0, 0, 255))
    cv2.line(img, tuple(pts[2]), tuple(pts[3]), color=(0, 0, 255))
    cv2.line(img, tuple(pts[3]), tuple(pts[0]), color=(0, 0, 255))

def same_line(pt1, pt2):
    return abs(pt1[0][1] - pt2[0][1]) < 10

def save_csv(all, imgs):
    with open("统计结果-%s.csv"%(time.strftime("%Y-%m-%d", time.localtime())), "w") as f:
        f.write("姓名, 时间, 地点,项目,结果\n")
        for res, img in zip(all, imgs):
            try:
                name_str = res.get("name", '')
                time_str = "".join(re.findall(r'\d{4}.\d{2}.\d{2}', res.get("time",'')))
                where_str = res.get("where", '')
                what_str = res.get("what", '')
                res_str = " ".join(re.findall(r'[\u4e00-\u9fa5]性', res.get("res", '')))
                if len(time_str) == 0 or len(where_str) == 0 or len(what_str) == 0 or len(res_str) == 0:
                    raise NameError("some is empty!")
                f.write("%s,%s,%s,%s,%s\n"%(
                                name_str,
                                time_str,
                                where_str,
                                what_str,
                                res_str,
                            )
                        )
            except Exception as e:
                print(f"Error {e} in {name_str}")
                shutil.copy2(img, config.err_path)



def parse_str(all_info: list):

    kv_list = [
        (r'.{0,}姓.{0,}名.{0,}', r'[\u4e00-\u9fa5]+', "name"),
        (r'.{0,}采.{0,}样.{0,}时.{0,}间.{0,}', r'\d{4}.\d{2}.\d{2}', "time"),
        # (r'.{0,}检.{0,}测.{0,}机.{0,}构.{0,}', r'[\u4e00-\u9fa5]+', "where"),
        (r'.{0,}检.{0,}测.{0,}项.{0,}目.{0,}', r'[\u4e00-\u9fa5]+', "what"),
        # (r'.{0,}检.{0,}测.{0,}结.{0,}果.{0,}', r'[\u4e00-\u9fa5]性', "res"),
    ]

    res = {}
    l = len(all_info)
    for idx, info in enumerate(all_info):
        # print(info)
        for k,v,name in kv_list:
            if re.findall(k, info[0]):
                # 如果已经获取过消息，那么返回，防止获取旧的信息
                if res.get(name, None):
                    continue
                count = 1
                while (idx + count ) < l and (not re.findall(v, all_info[idx + count][0])): count += 1
                if (idx + count ) < l: res[name] = all_info[idx + count][0]

        if re.findall(r'.{0,}检.{0,}测.{0,}机.{0,}构.{0,}', info[0]):
            if res.get("where", None):
                continue

            count = 1
            while (idx + count) < l and (not re.findall(r'[\u4e00-\u9fa5]+', all_info[idx + count][0])): count += 1
            if (idx + count) < l: res["where"] = all_info[idx + count][0]

            if abs(all_info[idx + count + 1][2][0][0] - info[2][0][0]) > 10:
                res["where"] = res.get("where", '') + all_info[idx + count + 1][0]

        if re.findall(r'.{0,}检.{0,}测.{0,}结.{0,}果.{0,}', info[0]):
            if res.get("res", None):
                continue

            count = 1
            while (idx + count) < l and (not re.findall(r'.+性', all_info[idx + count][0])): count += 1
            if (idx + count) < l: res["res"] = re.findall(r'.+性', all_info[idx + count][0])[0]

            if (idx + count + 1) < l and same_line(all_info[idx + count + 1][2], all_info[idx + count][2]):
                # (abs(all_info[idx + count + 1][2][0][0] - all_info[idx + count][2][1][0]) > 10) and
                # (abs(all_info[idx + count + 1][2][0][0] - all_info[idx + count][2][1][0]) < 10):
                res["res"] = res.get("res", '') + ' ' + re.findall(r'.+性', all_info[idx + count + 1][0])[0]

    print(res)
    return res


def do_ocr(img_path: str, all_res, all_img, bs=8):
    if isinstance(img_path, str):
        all = get_all(img_path)
    else:
        all = img_path

    reader = easyocr.Reader(['ch_sim','en'], model_storage_directory="./model/")
    # all_res = []
    for img in all:
        try:
            imga = cv2.imdecode(np.fromfile(img, dtype=np.uint8), cv2.COLOR_BGR2RGB)
            _h, _w, _ = imga.shape
            whr = _h / _w
            if whr >= (16 / 9) and crop_img:
                _s = int(_h * crop_top)
                _e = int(_h * (1 - crop_bot))
                imga = imga[_s:_e,:,:]

            # imga = cv2.imread(img)
            # imga = np.array(Im.open(img))
            # imga = imga[:,:,::-1]
            res = reader.readtext(imga, detail=1, batch_size=bs)
            all_info = []
            for one in res:
                pts = one[0]
                all_info.append((one[1], one[2], one[0]))
                if save_res: draw_bb(pts, imga)

            all_res.append(parse_str(all_info))
            all_img.append(img)
            # if save_res: cv2.imwrite("./res/%s"%(img.split("/")[-1]), imga)
            if save_res: cv2.imencode('.jpg', imga)[1].tofile("%s/%s"%(config.res_path, img.split("/")[-1]))

        except Exception as e:
            print(e)
            print("%s read fail!"%img)
            shutil.copy2(img, config.err_path)


def do_process(img_p: str):

    get = get_all(img_p)
    core_nums = 1
    signle_num = len(get) // core_nums
    pss = []
    all_res = []
    all_img = []
    print("Split all img:")
    for i in range(core_nums):
        # split imgs
        start_idx = i * signle_num
        end_idx = (i + 1) * signle_num if i < (core_nums - 1) else len(get)
        temp_arg = get[start_idx: end_idx]

        # do multiprocess
        p = Process(target=do_ocr, args=(temp_arg, all_res, all_img, ))
        pss.append(p)

    for p in pss:
        p.start()

    for p in pss:
        p.join()

    save_res(all_res, all_img)

if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')

    img_path = config.img_path
    all_res = []
    all_img = []
    do_ocr(img_path, all_res, all_img)
    save_csv(all_res, all_img)