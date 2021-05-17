import os
file_dir = './'
def file_name(file_dir):
    FileList = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.out':
                FileList.append( file)
    print(FileList)
    return FileList

x = file_name(file_dir)