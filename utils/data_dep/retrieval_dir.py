import os
from os import listdir
from os.path import isfile, join


def ls_files(dir, ext=None, key_with_ext=True):
    f_dict = {}
    if ext is not None:
        ext = '.'+ext if ext[0] != '.' else ext
    for f in os.listdir(dir):
        if f[0] == '.':
            continue

        b, e = os.path.splitext(f)
        if ext is not None:
            if e != ext:
                continue
        
        file_path = os.path.join(dir, f)
        if os.path.isfile(file_path):
            if key_with_ext:
                f_dict[f] = file_path
            else:
                f_dict[b] = file_path
        
    return f_dict


# def ls_files_with_suffix(dir, suffix, ext=None, join_base=True):
#     files = ls_files(dir, ext=ext, join_base=False)
#     filtered_files = []
#     for f in files:
#         name, _ = os.path.splitext(f)
#         if name.endswith(suffix):
#             file_path = os.path.join(dir, f) if join_base else f
#             filtered_files.append(file_path)
#     return filtered_files


# def ls_files_with_prefix(dir, prefix, ext=None, join_base=True):
#     files = ls_files(dir, ext=ext, join_base=False)
#     filtered_files = []
#     for f in files:
#         name, _ = os.path.splitext(f)
#         if name.startswith(prefix):
#             file_path = os.path.join(dir, f) if join_base else f
#             filtered_files.append(file_path)
#     return filtered_files


# def ls_dirs(root_dir):
#     dirs = []
#     # ids = [d for d in os.listdir(train_dir)]
#     for d in os.listdir(root_dir):
#         dir = os.path.join(root_dir, d)
#         if os.path.isdir(dir):
#             dirs.append(dir)
#     return dirs


if __name__ == '__main__':
    # dir = 'D:/Datasets/DSB2018'
    # print(ls_dirs(dir))
    # print(ls_files(dir, ext='.csv'))

    dir = 'D:\Datasets\BBBC006_U2OScell\ground_truth'
    d = ls_files(dir, ext='.png', key_with_ext=False)
    print(d)

